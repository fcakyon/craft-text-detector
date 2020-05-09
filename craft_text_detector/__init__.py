from __future__ import absolute_import

__version__ = "0.2.1"

from craft_text_detector.imgproc import read_image

from craft_text_detector.file_utils import export_detected_regions, export_extra_results

from craft_text_detector.predict import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)

# load craft model
craft_net = load_craftnet_model()


# detect texts
def detect_text(
    image_path,
    output_dir=None,
    rectify=True,
    export_extra=True,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=False,
    long_size=1280,
    show_time=False,
    refiner=True,
    crop_type="poly",
):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        rectify: rectify detected polygon by affine transform
        export_extra: export heatmap, detection points, box visualization
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        long_size: desired longest image size for inference
        show_time: show processing time
        refiner: enable link refiner
        crop_type: crop regions by detected boxes or polys ("poly" or "box")
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualization of the detected characters/links,
         "text_crop_paths": list of paths of the exported text boxes/polys,
         "times": elapsed times of the sub modules, in seconds}
    """
    global craft_net

    # load image
    image = read_image(image_path)

    # load refiner if required
    if refiner:
        refine_net = load_refinenet_model(cuda)
    else:
        refine_net = None

    # load craftnet again if cuda is turned on
    if cuda:
        craft_net = load_craftnet_model(True)

    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        cuda=cuda,
        long_size=long_size,
        show_time=show_time,
    )

    # arange regions
    if crop_type == "box":
        regions = prediction_result["boxes"]
    elif crop_type == "poly":
        regions = prediction_result["polys"]
    else:
        raise TypeError("crop_type can be only 'polys' or 'boxes'")

    # export if output_dir is given
    prediction_result["text_crop_paths"] = []
    if output_dir is not None:
        # export detected text regions
        exported_file_paths = export_detected_regions(
            image_path=image_path,
            image=image,
            regions=regions,
            output_dir=output_dir,
            rectify=rectify,
        )
        prediction_result["text_crop_paths"] = exported_file_paths

        # export heatmap, detection points, box visualization
        if export_extra:
            export_extra_results(
                image_path=image_path,
                image=image,
                regions=regions,
                heatmaps=prediction_result["heatmaps"],
                output_dir=output_dir,
            )

    # return prediction results
    return prediction_result
