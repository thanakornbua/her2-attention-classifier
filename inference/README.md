# Inference Utilities

## Patch PNG vs .pth Tester

Two thin entrypoints share the same preprocessing + post-processing helpers in
`inference/shared/patch_test_utils.py` so you can sanity-check single patches
against any PyTorch `.pth` checkpoint.

### CLI (non-interactive)

```bash
python -m inference.cli.test_patch_pth \
  --model-path outputs/best_model.pth \
  --arch resnet50 \
  --patch-path samples/patch_001.png \
  --image-size 224 \
  --normalization imagenet \
  --activation auto \
  --class-names configs/classes.json \
  --device cuda:0 \
  --top-k 5 \
  --output-json outputs/patch_test.json
```

Key flags:

- `--model-path`: Required `.pth` checkpoint saved via `torch.save`.
- `--arch`: Backbone architecture (ResNet-18/50 today; extend via
  `inference/shared/load_model.py`).
- `--patch-path`: Single PNG/JPEG/TIFF patch to score.
- `--normalization`: `imagenet`, `tf` ([-1, 1]), or `none`.
- `--activation`: `auto`, `softmax`, `sigmoid`, `none` depending on your head.
- `--class-names`: Optional `.json`/`.txt`/`.csv` label list; otherwise provide
  `--num-classes` so the CLI can size the classification head.
- `--output-json`: Saves raw probabilities + ranked top-k to disk.

### GUI (Tkinter)

```bash
python -m inference.gui.test_patch_pth_gui
```

1. Click **Browse → Load** to select the `.pth` checkpoint.
2. Pick a patch image (PNG/JPG/TIFF). The preview panel shows the resized
   thumbnail used for inference.
3. (Optional) Provide a class-names file (JSON or plain text, one label per
   line); the GUI auto-updates the `# Classes` spinner.
4. Adjust image size, backbone, device string, normalization preset, activation
   mode, and top-k.
5. Press **Run Test** to see the ranked predictions table.

The GUI is intentionally dependency-light (Tkinter + Pillow + PyTorch). Use it
for quick manual QA while the CLI version stays scriptable for CI or batch
checks.

> **Note:** Both entrypoints expect PyTorch + torchvision to be installed in the
> active environment (`pip install torch torchvision --index-url ...`).

## Generating class-name files

When your dataset follows the `torchvision.datasets.ImageFolder` layout, you
can export the label order with:

```bash
python -m inference.cli.generate_class_names \
  --data-dir /path/to/imagefolder/root \
  --output configs/classes.json
```

Use `--sort` to alphabetize if you prefer stable ordering independent of folder
creation time. The resulting JSON/TXT file slots straight into both the CLI and
GUI testers.
