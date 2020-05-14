"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import cv2
import numpy as np


def read_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # following two cases are not explained in the original repo
    if img.shape[0] == 2:
        img = img[0]
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def normalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def denormalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, long_size, interpolation):
    height, width, channel = img.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
