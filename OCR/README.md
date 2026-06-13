# OCR (End-to-end Detection + Recognition)

This folder contains an end-to-end OCR pipeline assembled from two components:

- DBNet (text detector) for locating text regions in an image.
- CRNN (text recognizer) for recognizing cropped word patches.

Primary artifact:

- `ocr.ipynb` — an interactive notebook that implements:
  - A DBNet detector implementation (lightweight FPN backbone in the notebook).
  - CRNN recognizer loading and inference (expects `CRNN/crnn_trained.pth` or uses the CRNN notebook's model structure).
  - A `pipeline_end_to_end(image_path, dbnet_model, crnn_model, device)` convenience function which:
    1. Resizes the input image to a standard size for DBNet.
    2. Runs DBNet to produce polygon/box detections.
    3. Extracts and rectifies cropped word patches.
    4. Runs the CRNN recognizer on each patch and optionally applies lexicon correction.

Checkpoints

- The notebook references `DBNet/dbnet_icdar_best.pth` for detector weights and `CRNN/crnn_trained.pth` for recognizer weights. Place your trained checkpoints at these paths or modify the notebook cell that loads them.

Dependencies

- See `CRNN/README.md` for CRNN-specific dependencies. Additionally:
- easyocr (optional) — the notebook references EasyOCR-style DBNet utilities in some helper cells.
- albumentations / OpenCV for image cropping and geometric transforms (used in the notebook).

Example usage (inside the notebook):

- Initialize models and device in a cell:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dbnet = DBNet(pretrained_backbone=False).to(device)
dbnet.load_state_dict(torch.load('DBNet/dbnet_icdar_best.pth', map_location=device))
crnn = CRNN(num_classes=NUM_CLASSES).to(device)
crnn.load_state_dict(torch.load('CRNN/crnn_trained.pth', map_location=device))
```

- Then run the pipeline:

```python
results = pipeline_end_to_end('/path/to/image.jpg', dbnet, crnn, device)
for r in results:
    print(r['box'], r['text'])
```

Notes & next steps

- For production use, extract the DBNet model and the CRNN model into modules and add a small script `run_ocr.py` that accepts image paths and returns JSON results.
- Consider using a stronger text detector (CRAFT / EAST) or tuning the DBNet postprocessing thresholds for recall-vs-precision tradeoffs.
