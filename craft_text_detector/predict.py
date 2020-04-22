import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.imgproc as imgproc
import craft_text_detector.file_utils as file_utils
from craft_text_detector.models.craftnet import CRAFT

from collections import OrderedDict

CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
REFINENET_GDRIVE_URL = (
    "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def load_craftnet_model(cuda: bool = False):
    # get craft net path
    home_path = str(Path.home())
    weight_path = os.path.join(
        home_path, ".craft_text_detector", "weights", "craft_mlt_25k.pth"
    )
    # load craft net
    craft_net = CRAFT()  # initialize

    # check if weights are already downloaded, if not download
    url = CRAFT_GDRIVE_URL
    if os.path.isfile(weight_path) is not True:
        print("Craft text detector weight will be downloaded to {}".format(weight_path))

        file_utils.download(url=url, save_path=weight_path)

    # arange device
    if cuda:
        craft_net.load_state_dict(copyStateDict(torch.load(weight_path)))

        craft_net = craft_net.cuda()
        craft_net = torch.nn.DataParallel(craft_net)
        cudnn.benchmark = False
    else:
        craft_net.load_state_dict(
            copyStateDict(torch.load(weight_path, map_location="cpu"))
        )
    craft_net.eval()
    return craft_net


def load_refinenet_model(cuda: bool = False):
    # get refine net path
    home_path = str(Path.home())
    weight_path = os.path.join(
        home_path, ".craft_text_detector", "weights", "craft_refiner_CTW1500.pth"
    )
    # load refine net
    from craft_text_detector.models.refinenet import RefineNet

    refine_net = RefineNet()  # initialize

    # check if weights are already downloaded, if not download
    url = REFINENET_GDRIVE_URL
    if os.path.isfile(weight_path) is not True:
        print("Craft text refiner weight will be downloaded to {}".format(weight_path))

        file_utils.download(url=url, save_path=weight_path)

    # arange device
    if cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(weight_path)))

        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
        cudnn.benchmark = False
    else:
        refine_net.load_state_dict(
            copyStateDict(torch.load(weight_path, map_location="cpu"))
        )
    refine_net.eval()
    return refine_net


def get_prediction(
    image,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    long_size: int = 1280,
    poly: bool = True,
    show_time: bool = False,
):
    """
    Arguments:
        image: image to be processed
        output_dir: path to the results to be exported
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        canvas_size: image size for inference
        long_size: desired longest image size for inference
        poly: enable polygon type
        show_time: show processing time
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
    preprocessing_time = time.time() - t0
    t0 = time.time()

    # forward pass
    with torch.no_grad():
        y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]

    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    boxes_as_ratio = np.array(boxes_as_ratio)

    # calculate poly coords as ratios to image size
    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])
    polys_as_ratio = np.array(polys_as_ratio)

    text_score_heatmap = imgproc.cvt2HeatmapImg(score_text)
    link_score_heatmap = imgproc.cvt2HeatmapImg(score_link)

    postprocess_time = time.time() - t0

    times = {
        "resize_time": resize_time,
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    if show_time:
        print(
            "\ninfer/postproc time : {:.3f}/{:.3f}".format(
                refinenet_time + refinenet_time, postprocess_time
            )
        )

    return {
        "boxes": boxes,
        "boxes_as_ratios": boxes_as_ratio,
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "heatmaps": {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap,
        },
        "times": times,
    }

