import numpy as np

from spd_hot.postprocess import postprocess_onnx_output


def test_postprocess_onnx_output_decodes_one_detection():
    output = np.array([[[320.0], [320.0], [100.0], [120.0], [0.9]]], dtype=np.float32)
    detections = postprocess_onnx_output(output)

    assert len(detections) == 1
    assert detections[0]["class_id"] == 0
    assert detections[0]["confidence"] > 0.5


def test_postprocess_onnx_output_respects_confidence_threshold():
    output = np.array([[[320.0], [320.0], [100.0], [120.0], [0.1]]], dtype=np.float32)
    detections = postprocess_onnx_output(output, {"confidence_threshold": 0.25})
    assert detections == []
