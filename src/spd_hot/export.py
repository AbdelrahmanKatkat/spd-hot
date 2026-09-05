"""ONNX export wrapper for Ultralytics models."""

from pathlib import Path


def export_onnx_bytes(model, *, imgsz: int | None = None) -> bytes:
    """Export a YOLO model to ONNX, validate it, and return file bytes."""
    import onnx

    size = imgsz
    if size is None:
        raw = getattr(getattr(model, "args", None), "imgsz", 640)
        size = int(raw[0] if isinstance(raw, (list, tuple)) else raw)
    onnx_path = Path(model.export(format="onnx", imgsz=size))
    proto = onnx.load(onnx_path)
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx_path)
    try:
        return onnx_path.read_bytes()
    finally:
        onnx_path.unlink(missing_ok=True)

