import numpy as np

from spd_hot.decode import decode_yolo_onnx_output


def test_decode_yolo_output_returns_one_detection():
    # Shape: (1, 4+nc, anchors), with nc=1 and anchors=1
    # cx, cy, w, h, class_score
    output = np.array([[[320.0], [320.0], [100.0], [120.0], [0.9]]], dtype=np.float32)
    detections = decode_yolo_onnx_output(output, confidence_threshold=0.25, iou_threshold=0.45)

    assert len(detections) == 1
    det = detections[0]
    assert det["class_id"] == 0
    assert det["confidence"] > 0.5

