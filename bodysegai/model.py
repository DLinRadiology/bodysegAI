import os
import numpy as np
import onnxruntime as ort

_session = None
_input_name = None


def get_model_path():
    base = os.environ.get('BODYSEGAI_BASE',
                          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "xRobotstuffx", "model.onnx")


def load_model():
    global _session, _input_name
    if _session is None:
        path = get_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ONNX model not found at {path}. Run convert_model.py first."
            )
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
    return _session


def predict_slice(image_512: np.ndarray) -> np.ndarray:
    """Run inference on a single 512x512 slice. Returns (512,512,3) float32."""
    session = load_model()
    inp = image_512.astype(np.float32)[np.newaxis, :, :, np.newaxis]
    result = session.run(None, {_input_name: inp})
    return result[0][0]  # (512, 512, 3)
