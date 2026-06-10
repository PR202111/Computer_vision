## CRNN (Convolutional Recurrent Neural Network) OCR

This folder contains two Jupyter notebooks that implement a complete OCR pipeline based on a small CRNN followed by optional lexicon-based correction:

- `crnn.ipynb` — dataset preparation, model definition, training loop, decoding and offline evaluation on the provided CSV splits. Saves trained weights to `crnn_trained.pth`.
- `ocr.ipynb`  — inference pipeline that: detects word boxes using EasyOCR's detector (CRAFT), crops words, runs the trained CRNN to recognize text, and (optionally) applies lexicon-based edit-distance correction.

## Quick summary

- Model: small CRNN with a 4-layer CNN backbone that squeezes the height and produces a sequence of feature vectors, followed by a 2-layer bidirectional LSTM and a linear classifier. Training uses CTC loss.
- Input size: grayscale images resized to (32, 256) before feeding the network.
- Decoding: simple greedy CTC decoding (collapse repeated characters and remove the blank index).
- Lexicon correction: optional edit-distance (Levenshtein) correction against a small lexicon per sample.

## Files in this folder

- `crnn.ipynb` — notebook with dataset loading, transforms, `OCRDataset`, `CRNN` model class, training loop and evaluation.
- `ocr.ipynb`  — notebook with the detection+recognition pipeline (uses EasyOCR detector, crops patches, runs the same CRNN model at inference time).
- `crnn_trained.pth` — (expected) weights file produced by `crnn.ipynb` training (may be in the repo or produced after training).
- `train/`, `test/` — image folders referenced by the CSV files in this dataset (relative paths in the CSVs).
- `traindata.csv`, `testdata.csv` — expected CSVs listing the image paths, ground truth string and optional lexicon columns.

## Dataset format

The notebooks expect CSV files with at least following columns (used in the notebooks):

- `ImgName` — path to the image file (relative path works). Example: `CRNN/train/5186_8.png`.
- `GroundTruth` — the ground truth transcription for the image (string).
- `smallLexi` (optional) — a string representation of a Python list holding candidate words for lexicon correction (used during evaluation/inference for lexicon-correction experiments).

If you reuse the code, ensure these columns exist and image paths are correct relative to the notebook working directory.

## Dependencies

Suggested Python packages (create a venv before installing):

- Python 3.8+
- torch, torchvision
- pillow
- numpy, pandas
- matplotlib
- opencv-python
- easyocr (for the detector in `ocr.ipynb`)
- python-Levenshtein

Install with pip (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision pillow pandas matplotlib opencv-python easyocr python-Levenshtein
```

Note: install the correct `torch` wheel for your CUDA / CPU setup following the PyTorch installation instructions if you want GPU acceleration.

## How to train (notebook)

1. Open `CRNN/crnn.ipynb` in Jupyter Lab / Notebook.
2. Make sure `traindata.csv` and `testdata.csv` point to the correct image paths.
3. Inspect and adjust hyperparameters near the top (e.g. `NUM_EPOCHS`, learning rate, batch size inside DataLoader creation).
4. Run the cells in order. The notebook includes a complete training loop and will save weights to `crnn_trained.pth` when finished.

The training cell defines a `run_epoch` function and a simple loop:

- Loss: CTC (torch.nn.CTCLoss)
- Optimizer: Adam (lr ~ 3e-4 in the notebook)

After training the notebook saves `crnn_trained.pth` to the current working directory.

## Inference / OCR pipeline

Two inference helpers are included across the notebooks:

- `predict_image(path)` — loads one image, runs the resizing/normalization transforms and returns the decoded string.
- `pipeline_ocr(image_path)` — two-stage pipeline using EasyOCR's detector (CRAFT) to find boxes, crops each detected box, preprocesses it and runs the CRNN to produce a list of detected box + transcription pairs.

To run inference in a Python script (outline):

```python
import torch
from PIL import Image
# Define or import CRNN class exactly as in the notebooks
# Construct model and load weights:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('crnn_trained.pth', map_location=device))
model.to(device)
model.eval()

# Then use the notebook's `predict_image(image_path)` or `pipeline_ocr(image_path)` helper logic
```

The `ocr.ipynb` notebook shows a ready-to-run `pipeline_ocr()` which uses `easyocr.Reader` to detect text regions and passes each cropped patch to the CRNN model.

## Evaluation & metrics

The training notebook evaluates word-level accuracy on the test split. It computes:

- Raw model word accuracy: exact match between predicted string and ground truth.
- Lexicon-corrected accuracy: after applying edit-distance correction between the predicted string and a provided per-sample lexicon (if available in the CSV). The notebook uses `python-Levenshtein` to compute distances.

Example evaluation output (printed by the notebook):

- Total Evaluated Words: N
- Raw Model Word Accuracy: correct_raw/N = XX.XX%
- Lexicon-Corrected Word Accuracy: correct_lexicon/N = YY.YY%

## Tips and next steps

- If you want to reuse the code in production, extract the `CRNN` class and inference functions into a Python module (e.g. `crnn/model.py`, `crnn/infer.py`) so you can import them from scripts.
- Consider replacing greedy decoding with a beam-search decoder plus a language model for better accuracy.
- If your dataset contains long sequences, increase the width (256) or adapt pooling to preserve enough time steps.
- Optionally integrate a stronger detector (CRAFT) or tune the detector threshold for your use-case.

## Contact / Attribution

This README was generated from the notebooks `crnn.ipynb` and `ocr.ipynb` present in this folder. See those notebooks for the exact code and for interactive cells that visualize inputs/outputs.
