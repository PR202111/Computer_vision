import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms

import pyclipper
import shapely.geometry as plg

import util

random.seed(123456)

ic15_train_data_dir = ROOT / "ch4_training_images"
ic15_train_gt_dir = ROOT / "ch4_training_localization_transcription_gt"


def get_img(img_path):
    img = cv2.imread(str(img_path))
    img = img[:, :, [2, 1, 0]]
    return img


def get_bboxes(img, gt_path):
    h, w = img.shape[:2]

    lines = util.io.read_lines(gt_path)

    bboxes = []
    tags = []

    for line in lines:

        line = line.strip().replace("\ufeff", "")

        gt = line.split(",")

        if gt[-1].startswith("#"):
            tags.append(False)
        else:
            tags.append(True)

        box = [int(gt[i]) for i in range(8)]

        box = np.asarray(box).reshape(4, 2)

        bboxes.append(box)

    return np.array(bboxes), tags


def perimeter(poly):
    peri = 0

    for i in range(len(poly)):
        peri += np.linalg.norm(
            poly[i] - poly[(i + 1) % len(poly)]
        )

    return peri


def polygon_to_mask(shape, polygons):

    mask = np.zeros(shape[:2], dtype=np.uint8)

    for poly in polygons:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

    return mask


def generate_threshold_map(shape, polygons):

    thresh_map = np.zeros(shape[:2], dtype=np.float32)

    for poly in polygons:

        mask = np.zeros(shape[:2], dtype=np.uint8)

        cv2.fillPoly(
            mask,
            [poly.astype(np.int32)],
            255
        )

        dist = cv2.distanceTransform(
            mask,
            cv2.DIST_L2,
            5
        )

        if dist.max() > 0:
            dist = dist / dist.max()

        thresh_map = np.maximum(
            thresh_map,
            dist
        )

    return thresh_map


class DBNetLoader(data.Dataset):

    def __init__(self):

        self.img_paths = []
        self.gt_paths = []

        img_names = util.io.ls(
            ic15_train_data_dir,
            ".jpg"
        )

        img_names.extend(
            util.io.ls(
                ic15_train_data_dir,
                ".png"
            )
        )

        for img_name in img_names:

            self.img_paths.append(
                str(ic15_train_data_dir / img_name)
            )

            gt_name = (
                "gt_"
                + img_name.split(".")[0]
                + ".txt"
            )

            self.gt_paths.append(
                str(ic15_train_gt_dir / gt_name)
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        img = get_img(
            self.img_paths[index]
        )

        bboxes, tags = get_bboxes(
            img,
            self.gt_paths[index]
        )

        prob_map = polygon_to_mask(
            img.shape,
            bboxes
        ).astype(np.float32)

        thresh_map = generate_threshold_map(
            img.shape,
            bboxes
        )

        training_mask = np.ones(
            img.shape[:2],
            dtype=np.float32
        )

        for poly, tag in zip(
            bboxes,
            tags
        ):

            if not tag:

                cv2.fillPoly(
                    training_mask,
                    [poly.astype(np.int32)],
                    0
                )

        img = Image.fromarray(img)

        img = transforms.ToTensor()(img)

        img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        prob_map = torch.from_numpy(
            prob_map
        ).float()

        thresh_map = torch.from_numpy(
            thresh_map
        ).float()

        training_mask = torch.from_numpy(
            training_mask
        ).float()

        return (
            img,
            prob_map,
            thresh_map,
            training_mask
        )