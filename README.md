# `spd-hot`

Python helper library for the HOT **swimming-pool detector** on fAIr. It wraps the YOLO11m model
[`mozilla-ai/swimming-pool-detector`](https://huggingface.co/mozilla-ai/swimming-pool-detector) (ONNX export at 512:
[`AbdelrahmanU3/swimming-pool-detector-OAM`](https://huggingface.co/AbdelrahmanU3/swimming-pool-detector-OAM)) so a fAIr
model pack (`fAIr-models/models/<pack>/`) can stay a thin STAC/ZenML/Docker adapter.

**Python:** >= 3.12. One class, `swimming_pool`; boxes in, GeoJSON polygons out.

## Inference path

```text
chip dir (OAM-<x>-<y>-<z>.tif/.png/.jpg, RGB)
  -> preprocess_chip()           RGB, one bilinear resize to 512 x 512, divide by 255 -> (3, 512, 512) float32 in [0, 1]
  -> onnxruntime session.run()   ONNX: network, anchors, box decoding -> (1, 5, 5376); NO normalisation inside
  -> decode_detections()         confidence filter, NMS, cap 300
  -> detections_to_chip_pixels() back to chip pixels
  -> detections_to_feature_collection()   EPSG:4326 box polygons
```

Entry point: `spd_hot.predict_session(session, input_images, params)` returns a GeoJSON `FeatureCollection`
(`confidence`, `class_id`, `class_name`, `source` per feature). All `params` are optional: `confidence_threshold`
(alias `score_thresh`, default 0.25), `nms_threshold` (aliases `iou_threshold`, `nms_thresh`, default 0.45) and
`model_input_size` (alias `chip_size`, default 512).

**Chip names must be `OAM-<x>-<y>-<z>.<ext>`.** PNG/JPEG chips have no georeferencing, so their polygons stay in pixel
coordinates.

### Base model, normalisation and scale

| | |
|---|---|
| Base weights | `mozilla-ai/swimming-pool-detector` `model.pt`, YOLO11m, one class; used unchanged |
| Trained on | [`mozilla-ai/osm-swimming-pools`](https://huggingface.co/datasets/mozilla-ai/osm-swimming-pools): 512 x 512 px Mapbox images, each one tile at zoom 18; OSM `leisure=swimming_pool` boxes |
| Trained how | `imgsz=512`, 10 epochs, patience 3, AdamW `lr0=0.01`, cosine schedule, `scale=0`, `translate=0`, `flipud=0.5`; the rest Ultralytics defaults (from Mozilla's notebook and the checkpoint's stored arguments) |
| Normalisation | Only the division by 255 in `preprocess.py`. The ONNX graph has none (its first node is a convolution) and Ultralytics uses no mean or std |
| Input size | 512, the size of training (`params.MODEL_INPUT_SIZE`) |

**Use zoom 18.** One training pixel is `40075016.686 * cos(lat) / 2^18 / 512`, about `0.2986 * cos(lat)` m. A 256 px
chip of a zoom-18 tile covers the same ground as a training image, so resizing it to 512 gives the training scale. The
resize does not read the chip's zoom, so chips of another zoom are **not** brought to that scale: at zoom 20 pools look
about 4 times larger than the model saw. `spd_hot.tile_gsd_m(name, width_px)` gives the true ground size of a chip pixel
and `scripts/report_chip_scale.py` prints it for a folder.

## Install, extras and who installs what

| Extra | Contents | Who installs it |
|---|---|---|
| *(none)* | numpy, Pillow, pyproj, rasterio. **No torch.** | the CPU inference image (plus `onnxruntime`, which the fAIr serving base provides) |
| `train` | ultralytics (brings torch), onnx: fine-tuning, evaluation, ONNX export | the fAIr **GPU** training image |
| `infer` | onnxruntime | running an exported ONNX locally (`scripts/predict.py`, `scripts/evaluate.py`) |
| `test` | pytest | CI |

Training runs on a GPU, but the CUDA build of torch is chosen by the image (a CUDA wheel index), not by this package's
metadata. The ONNX export through Ultralytics also wants `onnxslim` and `onnxruntime`; if they are missing Ultralytics
installs them itself during the export, so install them in the training image instead of letting it do so at run time.

```bash
uv sync                                # inference core
uv sync --extra train --extra infer --extra test     # development
uv run python scripts/export_onnx.py weights/model.pt weights/model.onnx    # base-model ONNX at 512 (needs `train`)
```

A fAIr model pack installs it from a pinned ref, for example
`uv pip install "spd-hot[train] @ git+https://github.com/<owner>/spd-hot@<tag-or-sha>"` in the training stage and
`"spd-hot @ git+https://github.com/<owner>/spd-hot@<tag-or-sha>"` in the inference stage.

ONNX I/O: input `images (1, 3, 512, 512)` float32, RGB, NCHW, values in [0, 1]; output `output0 (1, 4 + nc, anchors)`,
that is `(1, 5, 5376)`: per anchor `cx, cy, w, h` in model pixels, then one sigmoid score per class. The anchors, the
network and the box decoding are inside the graph; the confidence filter and NMS (both tunable) are in
`decode_detections`. An ONNX with another layout is refused, not guessed.

## Layout

| Module | Role |
|---|---|
| `params.py` | defaults + `parse_*` (tolerate `None`, aliases `score_thresh`, `nms_thresh`, `chip_size`, `lr`) |
| `preprocess.py` | read RGB, divide by 255, the one resize, `PreprocessMeta` |
| `tiles.py` | OAM chip name -> ground resolution (zoom and latitude, no CRS) |
| `decode.py` | confidence filter, NMS, cap on the detections (no CRS) |
| `georef.py`, `postprocess.py` | model pixels -> chip pixels -> EPSG:4326, GeoJSON |
| `serve.py` | `predict_session`, `predict_chip`, `detect_chip` |
| `dataset.py` | `spatial_split` (block split on OAM tile coordinates), GeoJSON polygons -> per-chip pixel boxes |
| `yolo_dataset.py` | pixel boxes -> the Ultralytics folder (chips linked, not copied) and its removal (`train`) |
| `train.py` | Ultralytics fine-tuning from a checkpoint (`train`) |
| `evaluate.py` | precision / recall / F1 at one or several IoUs, confidence sweep (no torch: scores an ONNX session) |
| `export.py` | ONNX bytes of a trained model at the model size (`train`) |

`PreprocessMeta` (source chip size, model size, transform, CRS) is what `preprocess` returns, not a setting: it is
built per chip and is not in `params.py`. `train`, `yolo_dataset` and `export` are imported explicitly (`from
spd_hot.train import train`); `import spd_hot` never loads torch.

## Scripts

`scripts/` holds thin command-line entry points; each one only parses arguments and calls `spd_hot`.

| Script | Calls | Extra |
|---|---|---|
| `scripts/finetune.py` | `train` + `export`: fine-tunes, writes `<name>.pt` and `<name>.onnx`, `split.json`, `train.log`, `runs/` | `train` |
| `scripts/evaluate.py` | `evaluate`: P/R/F1 of an ONNX on chips + GeoJSON, optional confidence sweep | `infer` |
| `scripts/predict.py` | `predict_session`: an ONNX over a folder of chips to GeoJSON, on CPU | `infer` |
| `scripts/export_onnx.py` | `export`: a `.pt` checkpoint to ONNX at 512 | `train` |
| `scripts/report_chip_scale.py` | `tiles`: true ground size of each chip and of one model pixel | none |
| `scripts/select_global_pool_aois.py` | AOI preflight: OSM pools under OAM imagery of an eligible licence, per candidate area | none |

The dataset downloader (OAM chips + OSM pool outlines for the selected areas) and the outdoor-pool filter of the OSM
outlines live in a local `research/` folder (not committed): dataset-specific code stays out of the package.

## From a pipeline step (what a fAIr-models pack calls)

A step trains from the original chips and GeoJSON. The Ultralytics folder it needs holds **links to the chips** (a
symlink, else a hardlink; it fails instead of copying) and one small label file per chip, so it costs a few kilobytes
per chip, not the chips again. Create it in a temporary folder so nothing outlives the step.

```python
import tempfile
from pathlib import Path

from spd_hot import build_chip_labels, parse_preprocess_params, parse_split_params, spatial_split
from spd_hot.evaluate import score_session                 # no torch: scores the exported ONNX
from spd_hot.export import export_onnx_bytes               # `train` extra, imported inside the step
from spd_hot.params import parse_train_params
from spd_hot.train import train
from spd_hot.yolo_dataset import write_yolo_dataset

split = parse_split_params(hyperparameters)                # val_ratio, split_seed, block_size
pre = parse_preprocess_params(hyperparameters)             # model_input_size (512)
labels = build_chip_labels(chips_dir, labels_geojson)      # chip name -> pixel boxes (empty list = negative chip)
train_names, val_names = spatial_split(sorted(labels), split.val_ratio, split.split_seed, block_size=split.block_size)
with tempfile.TemporaryDirectory() as work:                # the links, labels and Ultralytics runs go with it
    yolo_dir = write_yolo_dataset(chips_dir, labels, train_names=train_names, val_names=val_names,
                                  out_dir=Path(work) / "yolo")
    model = train(weights=base_weights, dataset_yaml=yolo_dir / "data.yaml", cfg=parse_train_params(hyperparameters),
                  pre=pre, project_dir=Path(work) / "runs",
                  on_epoch_end=lambda epoch, train_loss, val_loss: ...)   # -> MLflow
    onnx_bytes = export_onnx_bytes(model, imgsz=pre.model_input_size)
metrics = score_session(onnx_session, images_dir=chips_dir, labels=labels, names=val_names)[0.5]
```

`scripts/finetune.py` does the same on disk and removes `<out>/yolo` after a successful run (`--keep-dataset` keeps
it); what stays in the run folder are `split.json`, `train.log`, `<name>.pt`, `<name>.onnx` and Ultralytics' `runs/`,
whose `best.pt` and `last.pt` are the part to clean up between runs. `remove_yolo_dataset` refuses a folder that
`write_yolo_dataset` did not write.

At serving time the pack calls `spd_hot.predict_session(session, input_images, params)`; `spd_hot.serve.detect_chip`
returns the detections in chip pixels when a step needs them instead of GeoJSON.

## Tests

```bash
uv run ruff check src tests
uv run pytest -q                      # offline unit tests: synthetic chips, a fake ONNX session, no downloads
```
