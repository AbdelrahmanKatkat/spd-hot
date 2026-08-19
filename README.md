# `spd-hot` (Swimming Pool Detector for HOT fAIr)

`spd-hot` is a **small Python library** that holds the reusable ML logic for the
HOT fAIr base model **YOLOv11m Swimming Pool Detection**.

It is designed to be imported by a thin `fAIr-models` contribution wrapper such as:
- `fAIr_repos/fAIr-models/models/yolo11m_swimming_pools/pipeline.py`

That wrapper is responsible for the **fAIr contract** (ZenML steps/pipelines,
STAC entrypoints, Docker stages, tests). `spd-hot` is responsible for the
**model logic** (preprocess, ONNX decode, georeferencing, training dataset prep,
Ultralytics train/eval/export helpers).

## What `spd-hot` is (and is not)

- **Is**: a reusable implementation library (easy to unit-test without ZenML).
- **Is**: parameterized via a single `params.py` module (defaults + parsing).
- **Is not**: a ZenML pipeline package (no `@pipeline`, no `@step`).
- **Is not**: a STAC package (no STAC JSON generation/validation).
- **Is not**: tied to `fair` imports (so it stays portable).

`fAIr-models` remains the single source of truth for the base model contribution
(`models/<name>/Dockerfile`, `stac-item.json`, step tests, etc.).

## Hyperparameters philosophy (STAC stays in control)

fAIr controls hyperparameters through the STAC item (`mlm:hyperparameters` +
`fair:hyperparameters_spec`). `spd-hot` must **not** block that.

The pattern we use is:
- `spd_hot/params.py` defines **defaults** and **parsers** (dataclasses / helpers)
- the caller (the `pipeline.py` wrapper) passes in `hyperparameters` / `params`
- `spd-hot` merges `incoming params` over `defaults` and uses the result

So STAC overrides always win; defaults only apply when a key is missing.

## Recommended package layout (minimal + intuitive)

```text
spd-hot/
  pyproject.toml
  README.md
  src/spd_hot/
    __init__.py

    params.py                 # one place for defaults + parsing (train/infer/pre/post)

    preprocess.py             # GeoTIFF chip -> ONNX input tensor (+ meta for mapping back)
    decode.py                 # YOLO ONNX output -> bboxes + scores (+ NMS)
    postprocess.py            # detections -> GeoJSON FeatureCollection (EPSG:4326)
    georef.py                 # pixel/CRS transforms, clipping to chip bounds, min-area filters
    serve.py                  # predict_session(session, chips_dir, params) for fAIr serving

    dataset.py                # label prep + splitting for finetune (no ZenML, no STAC)
    train.py                  # ultralytics train wrapper (optional extra)
    evaluate.py               # ultralytics val wrapper -> metrics dict
    export.py                 # ultralytics export -> ONNX bytes + onnx.checker

  tests/
    test_preprocess.py
    test_decode.py
    test_georef.py
```

### Why each file exists

- **`params.py`**: you asked for “one file” to define knobs. This holds:
  - training params (epochs, batch_size, lr, val split, etc.)
  - preprocess params (model input size, normalization, channel order)
  - postprocess params (confidence threshold, IoU/NMS threshold, min area, etc.)
  - inference params (a stable subset of postprocess + any runtime toggles)

- **`preprocess.py`**: makes preprocessing explicit and shared across:
  - training-time preprocessing (if you ever need it)
  - ONNX serving (`fair.serve.base` passes chip paths)

  Preprocess should:
  - read RGB into **HWC** once (not per-channel)
  - normalize to **float32 [0..1]**
  - resize once to the model’s expected input (e.g. 640×640)
  - output ONNX input tensor as **NCHW**: `(1, 3, 640, 640)`
  - return **meta** needed for mapping outputs back (transform, crs, scale)

  **Do we georeference in preprocess?** No. Preprocess should *carry* the chip’s
  `transform/crs` forward as metadata. Georeferencing happens when we convert
  predicted pixel bboxes into EPSG:4326 geometries in postprocess.

- **`decode.py`**: holds “YOLO ONNX output decoding” and NMS. Keeping it isolated:
  - is required because fAIr serving calls `predict(session, ...)` with an
    `onnxruntime.InferenceSession` (see `fAIr-models/docs/contributing/model.md`)
  - avoids pulling the full Ultralytics+Torch stack into the distroless inference image
  - keeps the unavoidable “ONNX raw outputs → detections” glue code small and testable

  **Important design note**: this is not “reinventing Ultralytics”.
  Ultralytics is still the right tool for **training** and **exporting** to ONNX.
  At inference time inside fAIr, the model is executed via **onnxruntime** and returns
  numpy arrays; something must translate those arrays into boxes and apply NMS unless
  NMS is embedded in the exported ONNX graph.

  **Alternative (optional later)**: export ONNX with NMS embedded (if supported for
  YOLOv11). If that route works reliably, `decode.py` can shrink dramatically, but it
  tightens coupling to a specific export configuration and output tensor format.

- **`postprocess.py` + `georef.py`**: converts decoded detections into GeoJSON:
  - rescale boxes from model-input pixel space back to chip pixel space
  - clip to chip bounds
  - optional min-area filtering
  - transform from chip CRS to **EPSG:4326**

- **`serve.py`**: model-specific serving *logic*, not the HTTP server.
  The actual web server is `fair.serve.base` (Starlette app factory) which:
  - downloads chips (via `geomltoolkits`)
  - loads an `onnxruntime.InferenceSession`
  - imports the model pipeline module and calls `predict(session, chips_dir, params)`

  `spd-hot` still needs a clean place to hold the “predict over a directory of chips”
  logic so `pipeline.py` stays tiny. `serve.py` is that place (typically exposing a
  `predict_session(session, chips_dir, params)` function). It also enables:
  - local unit tests without ZenML
  - small scripts/CLIs to run inference on folders of chips

- **`dataset.py`**: prepares finetune datasets *without* a COCO detour.
  You explicitly said you don’t want the GeoJSON→COCO→YOLO path.
  A simpler approach is:
  - for each chip, intersect label polygons with chip bounds (clip)
  - convert clipped polygons to pixel bboxes via rasterio transform
  - write YOLO txt labels directly

  This is also the right place for **spatial splitting** (like `dinov3_hot.dataset.spatial_split`)
  so the ZenML `split_dataset` step can stay tiny.

- **`train.py` / `evaluate.py` / `export.py`**: small wrappers around Ultralytics:
  - training returns a trained `.pt` (or bytes) for the ZenML artifact
  - eval returns a **plain dict** with metric keys that the fAIr wrapper logs
  - export returns **ONNX bytes** (never a path string)

## Dependencies (what belongs in `spd-hot`)

Keep `spd-hot` **inference-first** and add training deps as an extra.

- **Core (inference)**:
  - `numpy`, `rasterio`, `pyproj`, `Pillow`
  - (optional) `shapely` if you want robust clipping/geometry ops

  Notes:
  - `numpy` is here because **onnxruntime returns numpy arrays**. Using numpy for
    decode/postprocess is the lightest way to keep the inference image small.
  - `onnxruntime` itself is already a dependency of `fair-py-ops`; `spd-hot` may
    still depend on it for local (non-fAIr) inference utilities if desired.

- **Training extra** (only installed in the training/batch image):
  - `ultralytics` (brings torch), plus `onnx`, `onnxscript` for export/checks

## About `geomltoolkits`

`geomltoolkits` is a **geospatial utility** library used by `fair-py-ops[serve]`:
- `fair.serve.base` uses it to download TMS imagery chips (georeferenced)
- `fair dataset build` uses it to fetch OSM labels + imagery

It does **not** provide an ONNX pipeline (no `onnxruntime` sessions / export helpers).

You may use `geomltoolkits` in a future “v2 footprints” workflow (mask→polygons via
`vectorize_mask`, `orthogonalize`, `validate_polygon_geometries`), but for v1 YOLO
detection labels and ONNX serving it’s not required.
