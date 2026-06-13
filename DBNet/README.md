# DBNet text detector

This folder contains resources and utilities for the DBNet text detection model used by the repository's OCR pipeline.

What you'll find here:

- `dbnet_icdar_best.pth` — pretrained weights used by the OCR notebook (`OCR/ocr.ipynb`).
- `Loader/`, `util/` — helper scripts for dataset loading and preprocessing (used for training / evaluation experiments).
- `ch4_training_images/`, `ch4_test_images/` and ground-truth folders — sample datasets arranged in ICDAR-style folders (if present).
- `model.ipynb` — exploratory notebook for DBNet training/visualization.

Brief usage notes

- The `OCR/ocr.ipynb` notebook expects the DBNet checkpoint path at `DBNet/dbnet_icdar_best.pth` when running the end-to-end pipeline.
- DBNet prepared inputs are typically resized to a square or 1280x720 scale before forwarding through the network. See `OCR/ocr.ipynb` for the exact resizing used in the pipeline.

Dependencies

- PyTorch
- OpenCV (cv2)
- numpy

If you need to retrain DBNet or compile DCN ops (deformable conv), refer to EasyOCR's DBNet implementation and the `easyocr/scripts/compile_dbnet_dcn.py` helper (this repo uses EasyOCR-style DBNet assets in the environment).

Notes

- The repo includes a small DBNet checkpoint for quick experiments; for production/large-scale detection consider using the official EasyOCR DBNet or other detectors (CRAFT, TextSnake) and tune thresholds for your dataset.
