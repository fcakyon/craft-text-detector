import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from craft_text_detector import (
    export_detected_regions,
    export_extra_results,
    get_prediction,
    load_craftnet_model,
    load_refinenet_model,
    read_image,
)


class TestCraftTextDetectorHelpers(unittest.TestCase):
    image_path = "figures/idcard.png"

    def test_load_craftnet_model(self):
        craft_net = load_craftnet_model(cuda=False)
        self.assertTrue(craft_net)

        with TemporaryDirectory() as dir_name:
            weight_path = Path(dir_name, "weights.pth")
            self.assertFalse(weight_path.is_file())
            load_craftnet_model(cuda=False, weight_path=weight_path)
            self.assertTrue(weight_path.is_file())

    def test_load_refinenet_model(self):
        refine_net = load_refinenet_model(cuda=False)
        self.assertTrue(refine_net)

        with TemporaryDirectory() as dir_name:
            weight_path = Path(dir_name, "weights.pth")
            self.assertFalse(weight_path.is_file())
            load_refinenet_model(cuda=False, weight_path=weight_path)
            self.assertTrue(weight_path.is_file())

    def test_read_image(self):
        image = read_image(self.image_path)
        self.assertTrue(image.shape, (500, 786, 3))

    def test_get_prediction(self):
        # load image
        image = read_image(self.image_path)

        # load models
        craft_net = load_craftnet_model()
        refine_net = None

        # perform prediction
        text_threshold = 0.9
        link_threshold = 0.2
        low_text = 0.2
        cuda = False
        prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            cuda=cuda,
            long_size=720,
        )

        self.assertEqual(len(prediction_result["boxes"]), 35)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 111)
        self.assertEqual(len(prediction_result["polys"]), 35)
        self.assertEqual(
            prediction_result["heatmaps"]["text_score_heatmap"].shape, (240, 368, 3)
        )

    def test_get_prediction_without_read_image(self):
        # set image filepath
        image = self.image_path

        # load models
        craft_net = load_craftnet_model()
        refine_net = None

        # perform prediction
        text_threshold = 0.9
        link_threshold = 0.2
        low_text = 0.2
        cuda = False
        prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            cuda=cuda,
            long_size=720,
        )

        self.assertEqual(len(prediction_result["boxes"]), 35)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 111)
        self.assertEqual(len(prediction_result["polys"]), 35)
        self.assertEqual(
            prediction_result["heatmaps"]["text_score_heatmap"].shape, (240, 368, 3)
        )


if __name__ == "__main__":
    unittest.main()
