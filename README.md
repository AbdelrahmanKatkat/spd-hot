# `spd-hot`

Python library for the HOT **swimming pool detector** on fAIr. It owns the model logic — preprocessing, ONNX decoding, georeferencing, GeoJSON output, and YOLO training helpers — so the fAIr model pack (`fAIr-models/models/yolo11m_swimming_pools/`) stays a thin STAC/ZenML/Docker adapter.

**Python:** ≥ 3.12

---

## What it is

`spd_hot` is a reusable package, not a standalone service. It exposes typed helpers that fAIr (or local scripts) call for:

| Area | Modules | Purpose |
| --- | --- | --- |
| **Inference** | `preprocess`, `decode`, `georef`, `postprocess`, `serve` | GeoTIFF chips → ONNX → EPSG:4326 GeoJSON |
| **Training** | `dataset`, `geo_to_yolo_converter`, `train`, `evaluate`, `export` | GeoJSON + chips → YOLO fine-tune → metrics → ONNX bytes |
| **Config** | `params` | Defaults and dict → dataclass parsing (STAC hyperparameters) |

Core dependencies are lightweight (`numpy`, `Pillow`, `pyproj`, `rasterio`). Ultralytics and ONNX are optional extras for training/export only — the inference image does not need them.

---

## Model

| Item | Value |
| --- | --- |
| Architecture | **YOLO11m** (Ultralytics) |
| Base weights | [mozilla-ai/swimming-pool-detector](https://huggingface.co/mozilla-ai/swimming-pool-detector) on HuggingFace |
| Classes | 1 — `swimming_pool` |
| ONNX input | `(1, 3, 640, 640)` float32 NCHW, values in `[0, 1]` |
| ONNX output | YOLO detection tensor `(1, 4+nc, anchors)` or transposed variant |

Fine-tuned `.pt` weights are exported to ONNX via `export_onnx_bytes`. At serve time, fAIr loads the ONNX session and passes it to `predict_session`.

---

## Functionality overview

### Inference path

```text
GeoTIFF chip dir
    │
    ▼
preprocess_chip_for_onnx()     RGB bands → resize 640×640 → normalize → (1,3,640,640)
    │                          + PreprocessMeta (scale, transform, crs)
    ▼
onnxruntime session.run()      (provided by fAIr, not imported here)
    │
    ▼
decode_yolo_onnx_output()      threshold + NMS → list of xyxy boxes + scores
    │
    ▼
detections_to_feature_collection()  unscale → clip → EPSG:4326 polygons
    │
    ▼
GeoJSON FeatureCollection
```

Entry point for fAIr: `spd_hot.serve.predict_session(session, input_images, params)`.

### Training path

```text
chips/ (GeoTIFF) + labels/ (one .geojson)
    │
    ▼
prepare_yolo_detection_dataset_from_geojson()
    │  spatial block split (train/val)
    │  GeoJSON polygons → YOLO .txt labels (real writes)
    │  chips symlinked (or copied) into images/{train,val}/
    ▼
train_yolo()                   Ultralytics fine-tune from base weights
    │
    ▼
evaluate_yolo()                mAP, precision, recall → fAIr metric keys
    │
    ▼
export_onnx_bytes()            validated ONNX file bytes for promotion
```

---

## Preprocess

`spd_hot.preprocess.preprocess_chip_for_onnx(path, cfg)` converts one chip to model input.

**Steps**

1. Open with `rasterio`, read bands 1/2/3 (RGB) as HWC uint8.
2. Resize to `model_input_size` (default **640**) with Pillow bilinear — **no letterbox** (chips are square; resize is square → square).
3. Normalize to float32 `[0, 1]` when `normalize_01=True`.
4. Return tensor shape `(1, 3, S, S)` — **always NCHW**.

**Metadata (`PreprocessMeta`)**

| Field | Use |
| --- | --- |
| `width`, `height` | Original chip size (e.g. 256×256) |
| `scale_x`, `scale_y` | Model pixels → chip pixels |
| `transform`, `crs` | Chip pixels → map coordinates |

Configurable via `PreprocessParams` / `parse_preprocess_params()`.

---

## Postprocess

Postprocessing is split across decode (model space) and georef (map space).

### Decode (`spd_hot.decode`)

`decode_yolo_onnx_output(output, confidence_threshold, iou_threshold)`:

- Parses YOLO ONNX output (handles `(anchors, 4+nc)` layout).
- Filters by confidence (default **0.25**).
- Applies **NMS** in model pixel space (default IoU **0.45**).
- Returns detections: `{x1, y1, x2, y2, confidence, class_id}`.

Pure NumPy — no Ultralytics at inference time.

### Georef + GeoJSON (`spd_hot.postprocess` + `spd_hot.georef`)

`detections_to_feature_collection(detections, meta, cfg)`:

1. **Unscale** boxes from 640×640 back to original chip pixels (`unscale_from_model_space`).
2. **Clip** to chip bounds (`clip_xyxy_to_image`).
3. Drop boxes below `min_box_area_px` (default 0).
4. Convert each bbox to an **EPSG:4326** polygon using the chip’s raster transform and CRS (`xyxy_to_epsg4326_polygon`).
5. Emit a GeoJSON `FeatureCollection` with `confidence`, `class_id`, optional `source` (chip filename) and `class_name`.

Configurable via `PostprocessParams` / `InferenceParams` / `parse_inference_params()`.

---

## Parameters (defaults)

Defined in `spd_hot.params`:

| Dataclass | Key knobs | Defaults |
| --- | --- | --- |
| `PreprocessParams` | `model_input_size`, `chip_size_hint`, `normalize_01` | 640, 256, true |
| `PostprocessParams` / `InferenceParams` | `confidence_threshold`, `iou_threshold`, `min_box_area_px` | 0.25, 0.45, 0.0 |
| `SplitParams` | `val_ratio`, `split_seed`, `block_size` | 0.2, 42, 4 |
| `TrainParams` | `epochs`, `batch_size`, `learning_rate`, `freeze_encoder`, `sample_fraction` | 10, 8, 0.01, true, 1.0 |

fAIr passes plain dicts from STAC; use the `parse_*` helpers to get typed configs. Defaults apply when keys are missing — STAC can override any value.

`freeze_encoder=True` maps to Ultralytics `freeze=10` (first 10 layers frozen).

---

## Package layout

```text
src/spd_hot/
  params.py                 # defaults + parsing
  preprocess.py             # chip → NCHW tensor + PreprocessMeta
  decode.py                 # ONNX output → detections + NMS
  georef.py                 # chip pixels → EPSG:4326 (inference)
  postprocess.py            # detections → GeoJSON
  serve.py                  # predict_session() — fAIr predict entry
  dataset.py                # GeoJSON + chips → YOLO dataset tree
  geo_to_yolo_converter.py  # polygon → YOLO txt (training)
  train.py                  # Ultralytics train wrapper
  evaluate.py               # validation metrics
  export.py                 # ONNX export bytes
tests/
  test_preprocess.py
  test_decode.py
  test_georef.py
  test_dataset_yolo.py
```

---

## Install (uv)

From `cd fair_models/swimming_pool/spd-hot
`:

```bash
# Create and activate a virtualenv
uv venv
source .venv/bin/activate          # Linux / macOS / WSL
# .venv\Scripts\activate           # Windows PowerShell

# Core + tests (inference helpers, pytest, shapely)
uv pip install -e ".[test]"

# Add training/export (ultralytics, onnx)
uv pip install -e ".[train,test]"
```

Run commands inside the venv without activating:

```bash
uv run pytest -q
uv run python -c "from spd_hot import predict_session; print(predict_session)"
```

Alternative without uv:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[test]"
```

---

## Testing

```bash
uv run pytest -q
# or
python -m pytest -q
```

| Test file | What it checks |
| --- | --- |
| `test_preprocess.py` | Output shape `(1,3,S,S)`, float32, values in `[0,1]`, meta width/height |
| `test_decode.py` | Minimal YOLO-like tensor decodes to one detection after NMS |
| `test_georef.py` | Bbox clipping and coordinate mapping |
| `test_dataset_yolo.py` | GeoJSON + chips → YOLO `data.yaml`, `.txt` labels, train/val split |

Tests use synthetic GeoTIFFs and in-memory arrays — no real ONNX file or HuggingFace download required.

**Suggested layers**

1. **Unit** (above) — preprocess, decode, georef, dataset in isolation.
2. **Contract** — fake `session.run()` returning a fixed tensor; assert `predict_session` GeoJSON shape and properties.
3. **Integration** (optional) — real ONNX + `onnxruntime` + sample chip.

---

## fAIr integration (thin adapter)

In `fAIr-models/models/yolo11m_swimming_pools/pipeline.py`, inference delegates to this library:

```python
from spd_hot.serve import predict_session

def predict(session, input_images, params):
    return predict_session(session, input_images, params)
```

Same pattern as `dinov3s_buildings` → `dinov3_hot.serve.predict_session`. fAIr owns STAC, ZenML steps, Docker stages, and ONNX session loading; `spd-hot` owns everything after the session is ready.

Training steps resolve remote chips/labels locally, call `prepare_yolo_detection_dataset_from_geojson`, then `train_yolo` / `evaluate_yolo` / `export_onnx_bytes`.

---

## Public API

Re-exported from `spd_hot`:

`predict_session`, `preprocess_chip_for_onnx`, `decode_yolo_onnx_output`, `detections_to_feature_collection`, `spatial_split`, `train_yolo`, `evaluate_yolo`, `export_onnx_bytes`, and parameter dataclasses / parsers.

Dataset preparation is imported from `spd_hot.dataset` (`prepare_yolo_detection_dataset_from_geojson`).
