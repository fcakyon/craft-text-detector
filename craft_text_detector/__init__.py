from __future__ import absolute_import

import os

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.file_utils as file_utils
import craft_text_detector.image_utils as image_utils
import craft_text_detector.predict as predict
import craft_text_detector.torch_utils as torch_utils

__version__ = "0.4.1"


__all__ = [
    "read_image",
    "load_craftnet_model",
    "load_refinenet_model",
    "get_prediction",
    "export_detected_regions",
    "export_extra_results",
    "empty_cuda_cache",
    "Craft",
]

read_image = image_utils.read_image
load_craftnet_model = craft_utils.load_craftnet_model
load_refinenet_model = craft_utils.load_refinenet_model
get_prediction = predict.get_prediction
export_detected_regions = file_utils.export_detected_regions
export_extra_results = file_utils.export_extra_results
empty_cuda_cache = torch_utils.empty_cuda_cache


class Craft:
    def __init__(
        self,
        output_dir=None,
        rectify=True,
        export_extra=True,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=1280,
        refiner=True,
        crop_type="poly",
    ):
        """
        Arguments:
            output_dir: path to the results to be exported
            rectify: rectify detected polygon by affine transform
            export_extra: export heatmap, detection points, box visualization
            text_threshold: text confidence threshold
            link_threshold: link confidence threshold
            low_text: text low-bound score
            cuda: Use cuda for inference
            long_size: desired longest image size for inference
            refiner: enable link refiner
            crop_type: crop regions by detected boxes or polys ("poly" or "box")
        """
        self.craft_net = None
        self.refine_net = None
        self.output_dir = output_dir
        self.rectify = rectify
        self.export_extra = export_extra
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.long_size = long_size
        self.refiner = refiner
        self.crop_type = crop_type

        # load craftnet
        self.load_craftnet_model()
        # load refinernet if required
        if refiner:
            self.load_refinenet_model()

    def load_craftnet_model(self):
        """
        Loads craftnet model
        """
        self.craft_net = load_craftnet_model(self.cuda)

    def load_refinenet_model(self):
        """
        Loads refinenet model
        """
        self.refine_net = load_refinenet_model(self.cuda)

    def unload_craftnet_model(self):
        """
        Unloads craftnet model
        """
        self.craft_net = None
        empty_cuda_cache()

    def unload_refinenet_model(self):
        """
        Unloads refinenet model
        """
        self.refine_net = None
        empty_cuda_cache()

    def detect_text(self, image, image_path=None):
        """
        Arguments:
            image: path to the image to be processed or numpy array or PIL image

        Output:
            {
                "masks": lists of predicted masks 2d as bool array,
                "boxes": list of coords of points of predicted boxes,
                "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
                "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
                "heatmaps": visualization of the detected characters/links,
                "text_crop_paths": list of paths of the exported text boxes/polys,
                "times": elapsed times of the sub modules, in seconds
            }
        """

        if image_path is not None:
            print("Argument 'image_path' is deprecated, use 'image' instead.")
            image = image_path

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.cuda,
            long_size=self.long_size,
        )

        # arange regions
        if self.crop_type == "box":
            regions = prediction_result["boxes"]
        elif self.crop_type == "poly":
            regions = prediction_result["polys"]
        else:
            raise TypeError("crop_type can be only 'polys' or 'boxes'")

        # export if output_dir is given
        prediction_result["text_crop_paths"] = []
        if self.output_dir is not None:
            # export detected text regions
            if type(image) == str:
                file_name, file_ext = os.path.splitext(os.path.basename(image))
            else:
                file_name = "image"
            exported_file_paths = export_detected_regions(
                image=image,
                regions=regions,
                file_name=file_name,
                output_dir=self.output_dir,
                rectify=self.rectify,
            )
            prediction_result["text_crop_paths"] = exported_file_paths

            # export heatmap, detection points, box visualization
            if self.export_extra:
                export_extra_results(
                    image=image,
                    regions=regions,
                    heatmaps=prediction_result["heatmaps"],
                    file_name=file_name,
                    output_dir=self.output_dir,
                )

        # return prediction results
        return prediction_result
