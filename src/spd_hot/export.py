"""ONNX export of the fine-tuned YOLO (needs the `train` extra; import explicitly).

The file is Ultralytics' own export at the model input size: input `images` (1, 3, S, S) float32 in [0, 1], output
`output0` (1, 4 + nc, anchors). The confidence filter and NMS stay outside it (`decode.py`), as for the detector.
"""

import tempfile
from pathlib import Path

from .params import MODEL_INPUT_SIZE


def export_onnx_bytes(model, *, imgsz: int = MODEL_INPUT_SIZE) -> bytes:
    """Export a YOLO model to ONNX at `imgsz`, check the graph, and return the file bytes."""
    import onnx
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        # Ultralytics writes the ONNX next to the checkpoint it loads, so export from a copy in a temporary folder.
        checkpoint = Path(tmp) / "export.pt"
        model.save(checkpoint)
        onnx_path = Path(YOLO(str(checkpoint)).export(format="onnx", imgsz=imgsz))
        onnx.checker.check_model(str(onnx_path))
        return onnx_path.read_bytes()
