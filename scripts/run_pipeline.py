"""End-to-end reproducible FER2013 train/validation/test pipeline."""
from __future__ import annotations
import argparse, json, logging, os, sys, time, traceback
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import MODEL_DIR, RESULTS_DIR, RANDOM_SEED
from src.data.dataset import prepare_data
from src.data.analysis import generate_eda
from src.ml.models import candidate_models, fit_and_score, final_metrics
from src.tracking import new_run, save_json, plot_confusion, save_comparison

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("full","quick"),default="full")
    args=parser.parse_args(); RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    log_path=RESULTS_DIR/"pipeline.log"
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(log_path,encoding="utf-8")])
    lock=ROOT/".pipeline.lock"
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("Another pipeline appears active (.pipeline.lock exists); inspect and remove stale lock if needed.")
    run=new_run(args.mode); (run/"mode.txt").write_text(args.mode+"\n"+"development subset" if args.mode=="quick" else args.mode,encoding="utf-8")
    try:
        logging.info("Checking the active environment")
        import subprocess
        subprocess.run([sys.executable,str(ROOT/"scripts"/"check_environment.py")],cwd=ROOT,check=True,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logging.info("Preparing FER2013 data (mode=%s)",args.mode)
        data=prepare_data(quick=args.mode=="quick",run_dir=run)
        validation=json.loads((run/"metadata"/"data_validation.json").read_text(encoding="utf-8")
                              if args.mode=="quick" else (ROOT/"data"/"metadata"/"data_validation.json").read_text(encoding="utf-8"))
        generate_eda(data,validation,run/"dataset_analysis")
        (run/"status.txt").write_text("training",encoding="utf-8")
        Xtr,ytr=data["train"]; Xv,yv=data["val"]; Xt,yt=data["test"]
        rows=[]
        for name,model in candidate_models(RANDOM_SEED):
            if name.startswith("knn") and args.mode=="full" and len(ytr)>8000:
                ix,_=train_test_split(np.arange(len(ytr)),train_size=8000,random_state=RANDOM_SEED,stratify=ytr)
                Xfit,yfit=Xtr[ix],ytr[ix]
            else:
                Xfit,yfit=Xtr,ytr
            logging.info("Training %s on %d examples",name,len(yfit))
            result=fit_and_score(name,model,Xfit,yfit,Xv,yv); rows.append(result)
            model_dir=run/name; model_dir.mkdir()
            joblib.dump(result["estimator"],model_dir/"model.joblib",compress=3)
            safe={k:v for k,v in result.items() if k!="estimator"}
            save_json(model_dir/"metrics.json",safe); plot_confusion(result["confusion_matrix"],f"{name} validation",model_dir/"confusion_matrix.png")
            if hasattr(model,"loss_curve_"):
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(); ax.plot(model.loss_curve_); ax.set(xlabel="Iteration",ylabel="Log loss",title="MLP training loss"); fig.tight_layout(); fig.savefig(model_dir/"learning_curve.png",dpi=150); plt.close(fig)
        selected=max(rows,key=lambda r:(r["macro_f1"],r["accuracy"]))
        save_comparison(rows,run)
        best=next(r["estimator"] for r in rows if r["name"]==selected["name"])
        logging.info("Selected %s on validation macro-F1 %.4f",selected["name"],selected["macro_f1"])
        test=final_metrics(best,Xt,yt); save_json(run/"final_test_metrics.json",test)
        plot_confusion(test["confusion_matrix"],f"{selected['name']} final test",run/"final_test_confusion_matrix.png")
        MODEL_DIR.mkdir(parents=True,exist_ok=True)
        bundle={"model":best,"name":selected["name"],"class_names":["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"],"mode":args.mode,"validation_macro_f1":selected["macro_f1"]}
        joblib.dump(bundle,MODEL_DIR/"selected_model.joblib",compress=3)
        joblib.dump(bundle,run/"selected_model.joblib",compress=3)
        save_json(MODEL_DIR/"selected_model.json",{k:v for k,v in bundle.items() if k!="model"}|{"test_metrics":test,"run":str(run.relative_to(ROOT))})
        report=f"# Model comparison\n\nMode: **{args.mode}** ({'development-only subset' if args.mode=='quick' else 'full dataset'})\n\nSelected by validation macro-F1, then accuracy: **{selected['name']}**. Test partition evaluated once after selection.\n\nAvailable training samples: {len(ytr)}; validation: {len(yv)}; test: {len(yt)}. Each row in `validation_comparison.csv` records its exact fit sample count; full-mode KNN uses a stratified, seeded cap of 8,000 training examples for computational feasibility, while the MLP uses the full training partition.\n\nFinal test accuracy: {test['accuracy']:.4f}; macro-F1: {test['macro_f1']:.4f}.\n\nSee `validation_comparison.csv` and model-specific reports for per-class scores.\n"
        (run/"model_comparison_report.md").write_text(report,encoding="utf-8")
        summary=RESULTS_DIR/"experiments_summary.csv"
        import pandas as pd
        table=pd.DataFrame([{k:r[k] for k in ("name","accuracy","macro_f1","train_seconds","prediction_latency_ms_per_sample","train_samples")} | {"mode":args.mode,"run":run.name} for r in rows])
        table.to_csv(summary,index=False,mode="a",header=not summary.exists())
        (run/"status.txt").write_text("complete",encoding="utf-8")
        logging.info("Pipeline complete. Run directory: %s",run)
    except Exception as exc:
        logging.exception("Pipeline failed")
        (run/"status.txt").write_text("failed\n"+str(exc)+"\n"+traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        os.close(fd); lock.unlink(missing_ok=True)
if __name__=="__main__": main()
