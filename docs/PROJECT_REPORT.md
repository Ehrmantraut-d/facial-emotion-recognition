# Facial expression recognition project report

This report records the actual `full` run in `results/runs/20261121T222130Z_full_6297687c/`. Quick-mode values are development-only.

## Execution status (2026-11-21)

The `facial_emotion_ai` Conda environment uses Python 3.11.18. Required imports and the environment diagnostic passed. An earlier offline test run passed 15 tests. Re-running from the Desktop copy stopped on a Windows native Matplotlib exception during the EDA test, so the repeat run is inconclusive. FER2013 was imported from the local `Desktop/dataset.zip` archive after verifying its ZIP CRC. The complete run finished, saved all experiment outputs, and selected an MLP model. A prior quick run is retained separately and marked development-only.

The environment diagnostic opened and read frames from camera indices 0 and 1, then released both. MediaPipe Tasks loaded Google's official short-range face detector model successfully; a blank-frame smoke check returned no boxes as expected. After the visual refresh, the Tkinter interface was opened and its preview, face box, prediction panel, probability bars, and FPS display were observed in the live UI. Frames are processed locally and are not saved or transmitted.

## 1. Introduction and problem

The project studies supervised classification of seven visible facial-expression categories from grayscale facial images. The desktop application demonstrates local webcam inference. It does not claim access to a person's internal emotional state.

## 2. Objectives

Acquire and validate FER2013; preprocess it consistently; compare KNN and an MLP with held-out partitions; persist metrics and plots; and connect the selected trained model to a local Tkinter application.

## 3. Dataset

FER2013 contains 35,887 48x48 grayscale images labeled Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral. Microsoft FER+ crowd-vote annotations are included at `data/raw/ferplus/fer2013new.csv`; these improve target-label quality but reuse the same images rather than adding demographic or image diversity. The CSV records 28,709 Training, 3,589 PublicTest, and 3,589 PrivateTest examples; validation found zero invalid rows. The source archive is not included in this repository. See actual counts in `data/metadata/data_validation.json` when generated locally. The project does not assume a blanket right to redistribute these facial images.

## 4. Preprocessing and analysis

Rows with invalid labels, unknown split names, malformed pixel counts, or non-finite/out-of-range intensity values are reported and excluded. Images are normalized to [0,1] and flattened to 2,304 features. The source's public validation and private test partitions are kept intact. Quick mode selects stratified development subsets with a fixed seed and is explicitly marked.

## 5. Architecture and algorithms

See `ARCHITECTURE.md` and `AI_THEORY.md`. KNN evaluates k=3, 5, 7, and 11 using distance weighting with an explicitly recorded 8,000-sample cap in full mode. The MLP uses hidden layers (128,64), ReLU, Adam, and a softmax classifier. Selection uses validation macro-F1, then accuracy. The selected model is evaluated on the test set once.

## 6. Experiments and results

The full run compared four distance-weighted KNN values using a reproducible stratified subset of 8,000 training examples and an MLP on all 28,709 training examples. Validation results were:

| Model | Fit samples | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|
| KNN, k=3 | 8,000 | 0.2990 | 0.2867 |
| KNN, k=5 | 8,000 | 0.3026 | 0.2834 |
| KNN, k=7 | 8,000 | 0.3090 | 0.2866 |
| KNN, k=11 | 8,000 | 0.3174 | 0.2869 |
| MLP (128,64) | 28,709 | 0.4046 | 0.3220 |

The original MLP was selected by validation macro-F1. Its single-view baseline achieved 0.4046 validation accuracy and, in the original full run, 0.3931 accuracy / 0.3144 macro-F1 on PrivateTest. A separate validation-only comparison then selected horizontal-flip probability averaging: it averages predictions for the original face and its mirror without changing model weights. Validation accuracy rose to **0.4099** (macro-F1 0.3227); the selected strategy was evaluated once on PrivateTest and achieved **0.3979 accuracy** and **0.3178 macro-F1**. The active model bundle records this inference strategy, and the original model is preserved as `models/selected_model_before_flip_average.joblib`. Comparison artifacts are saved in `results/runs/20261121T225141Z_flip_average_64697dee/`; original run artifacts remain intact. Flip averaging doubles classifier evaluations per face but remains lightweight for this MLP. The retained quick development run used 700/350/350 samples and must not be reported as full-dataset performance.

### FER+ consensus-label training

Microsoft FER+ supplies multiple crowd votes for each image, aligned row-for-row with FER2013. For this seven-class project, training retained rows with a unique highest vote among the seven target emotions and at least four votes; tied, weak, or no-target rows were excluded. The candidate trained on 26,593 images. On 3,356 PublicTest rows with accepted FER+ labels, it reached **0.5498 accuracy** and **0.3636 macro-F1**, compared with **0.4231** and **0.3057** for the prior model evaluated on those same consensus targets. After selection on PublicTest, the model scored **0.5500 accuracy** and **0.3414 macro-F1** on 3,329 accepted PrivateTest consensus rows. These FER+ results use a different ground-truth definition and a filtered evaluation set; compare them only with the prior model on the same FER+ PublicTest rows, not directly with the original-label FER2013 scores. The original images and original-label experiment remain preserved. See `results/runs/20261121T231741Z_ferplus_majority_96ec438f/` and cite Microsoft’s [FER+ annotations](https://github.com/microsoft/FERPlus).

### Spatial CNN on FER+ consensus labels

The active model is a three-stage CNN trained on the same 26,593 accepted FER+ Training rows. The 48×48 grayscale image shape is preserved so learned 3×3 filters can detect spatial facial patterns. Each stage uses convolution, batch normalization, ReLU, max pooling, and dropout, followed by adaptive pooling and a 128-unit dense head. Random horizontal flips augment training only; horizontal-flip probability averaging is used at inference. Early stopping selected the lowest PublicTest validation-loss checkpoint after 18 epochs on CPU.

| Model | PublicTest consensus accuracy | Macro-F1 |
|---|---:|---:|
| FER+ MLP baseline | 0.5498 | 0.3636 |
| FER+ spatial CNN | **0.7378** | **0.5159** |

The validation-selected CNN was evaluated once on 3,329 accepted PrivateTest consensus rows: **0.7188 accuracy** and **0.5081 macro-F1**. Its compact per-class report, confusion matrix values, and training history are saved in `results/cnn_ferplus/`. The model is active at `models/selected_model.joblib`; the prior FER+ MLP is retained locally as `models/selected_model_before_cnn.joblib`. These metrics measure agreement with filtered FER+ consensus labels; they do not establish generalization to webcam conditions or measurement of internal emotion.

## 7. Face detection and application

MediaPipe locates faces when available; OpenCV Haar cascade is a local fallback. A detected crop is converted to grayscale, resized to 48x48, normalized identically to training, and passed to the selected CNN. Tkinter displays bounding boxes, French class names, per-class scores, and camera controls in a dark two-panel layout. The CNN averages class probabilities from the original face and its horizontal mirror. No webcam image is saved or transmitted.

## 8. Challenges and limitations

FER2013 has low-resolution images, class imbalance, label noise, and domain limitations. KNN can be slow and sensitive to lighting/alignment. The CNN learns spatial patterns, but webcam faces still differ from dataset crops and the model can perform unevenly across classes. Predicted labels are expression categories and should not be treated as reliable measurement of mental state.

## 9. Teamwork and conclusion

Use `TEAM_TASKS.md` as a starting point and replace it with the students' actual contributions. The project demonstrates supervised classification, preprocessing, log-loss optimization, backpropagation, gradient-based learning, validation, and confusion-matrix evaluation. Conclusion must be written using completed real metrics and a live application demonstration.
