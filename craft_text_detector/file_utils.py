import copy
import os

import cv2
import gdown
import numpy as np

from craft_text_detector.image_utils import read_image


def download(url: str, save_path: str):
    """
    Downloads file from gdrive, shows progress.
    Example inputs:
        url: 'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip'
        save_path: 'data/file.zip'
    """

    # create save_dir if not present
    create_dir(os.path.dirname(save_path))
    # download file
    gdown.download(url, save_path, quiet=False)


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                ext == ".jpg"
                or ext == ".jpeg"
                or ext == ".gif"
                or ext == ".png"
                or ext == ".pgm"
            ):
                img_files.append(os.path.join(dirpath, file))
            elif ext == ".bmp":
                mask_files.append(os.path.join(dirpath, file))
            elif ext == ".xml" or ext == ".gt" or ext == ".txt":
                gt_files.append(os.path.join(dirpath, file))
            elif ext == ".zip":
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def rectify_poly(img, poly):
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int(
            (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        )
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32(
            [[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32(
            [
                [width_step, 0],
                [width_step + w - 1, height - 1],
                [width_step, height - 1],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(
            warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1
        )
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
    return output_img


def crop_poly(image, poly):
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    # create mask with shape of image
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)

    # method 1 smooth region
    cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    # crop around poly
    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(poly)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    return cropped


def export_detected_region(image, poly, file_path, rectify=True):
    """
    Arguments:
        image: full image
        points: bbox or poly points
        file_path: path to be exported
        rectify: rectify detected polygon by affine transform
    """
    if rectify:
        # rectify poly region
        result_rgb = rectify_poly(image, poly)
    else:
        result_rgb = crop_poly(image, poly)

    # export corpped region
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, result_bgr)


def export_detected_regions(
    image,
    regions,
    file_name: str = "image",
    output_dir: str = "output/",
    rectify: bool = False,
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        regions: list of bboxes or polys
        file_name (str): export image file name
        output_dir: folder to be exported
        rectify: rectify detected polygon by affine transform
    """

    # read/convert image
    image = read_image(image)

    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # create crops dir
    crops_dir = os.path.join(output_dir, file_name + "_crops")
    create_dir(crops_dir)

    # init exported file paths
    exported_file_paths = []

    # export regions
    for ind, region in enumerate(regions):
        # get export path
        file_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        # export region
        export_detected_region(image, poly=region, file_path=file_path, rectify=rectify)
        # note exported file path
        exported_file_paths.append(file_path)

    return exported_file_paths


def export_extra_results(
    image,
    regions,
    heatmaps,
    file_name: str = "image",
    output_dir="output/",
    verticals=None,
    texts=None,
):
    """save text detection result one by one
    Args:
        image: path to the image to be processed or numpy array or PIL image
        file_name (str): export image file name
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4]
            for QUAD output
    Return:
        None
    """
    # read/convert image
    image = read_image(image)

    # result directory
    res_file = os.path.join(output_dir, file_name + "_text_detection.txt")
    res_img_file = os.path.join(output_dir, file_name + "_text_detection.png")
    text_heatmap_file = os.path.join(output_dir, file_name + "_text_score_heatmap.png")
    link_heatmap_file = os.path.join(output_dir, file_name + "_link_score_heatmap.png")

    # create output dir
    create_dir(output_dir)

    # export heatmaps
    cv2.imwrite(text_heatmap_file, heatmaps["text_score_heatmap"])
    cv2.imwrite(link_heatmap_file, heatmaps["link_score_heatmap"])

    with open(res_file, "w") as f:
        for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))
            strResult = ",".join([str(r) for r in region]) + "\r\n"
            f.write(strResult)

            region = region.reshape(-1, 2)
            cv2.polylines(
                image,
                [region.reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    (region[0][0] + 1, region[0][1] + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                )
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    tuple(region[0]),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness=1,
                )

    # Save result image
    cv2.imwrite(res_img_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
