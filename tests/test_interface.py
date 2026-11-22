import pytest
from unittest.mock import Mock
from src.ml.predictor import EmotionPredictor
from src.gui.app import EmotionApp

def test_missing_model_handled_without_camera(tmp_path):
    with pytest.raises(FileNotFoundError): EmotionPredictor(tmp_path/"absent.joblib")

def test_camera_failure_is_reported_and_controls_reenabled(monkeypatch):
    import src.gui.app as app_module
    cap=Mock(); cap.isOpened.return_value=False
    monkeypatch.setattr(app_module.cv2,"VideoCapture",lambda index: cap)
    monkeypatch.setattr(app_module.messagebox,"showerror",Mock())
    app=EmotionApp.__new__(EmotionApp); app.camera_index=0; app.cap=None
    app.start_btn=Mock(); app.stop_btn=Mock()
    app.status=Mock()
    app.start()
    assert app.cap is None and cap.release.called
    app.status.configure.assert_called_once_with(text="Caméra 0 indisponible")
    app_module.messagebox.showerror.assert_called_once()

def test_invalid_frame_schedules_retry_without_crashing():
    app=EmotionApp.__new__(EmotionApp); app.running=True; app.cap=Mock()
    app.cap.read.return_value=(False,None); app.status=Mock(); app.root=Mock()
    app._tick()
    app.status.configure.assert_called_once()
    app.root.after.assert_called_once_with(80,app._tick)

def test_camera_stop_releases_capture():
    app=EmotionApp.__new__(EmotionApp); app.running=True; app.cap=Mock()
    app.start_btn=Mock(); app.stop_btn=Mock(); app.video=Mock(); app.photo=object()
    app.stop()
    assert app.cap is None and not app.running and app.photo is None
    app.start_btn.configure.assert_called_with(state="normal")
    app.stop_btn.configure.assert_called_with(state="disabled")
