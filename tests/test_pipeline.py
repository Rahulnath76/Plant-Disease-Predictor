"""
Unit Tests for Plant Disease Prediction Package
"""

import sys
import os
import unittest
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("src"))

from plant_disease.config import Config
from plant_disease.models import Model, get_resnet18_model, build_model
from plant_disease.utils import set_seed, get_disease_remedy
from plant_disease.predict import predict


class TestPlantDiseasePipeline(unittest.TestCase):

    def setUp(self):
        set_seed(42)

    def test_config_defaults(self):
        self.assertEqual(Config.NUM_CLASSES, 10)
        self.assertEqual(len(Config.CLASSES), 10)

    def test_custom_cnn_output_shape(self):
        model = Model(num_classes=10)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 10))

    def test_resnet18_output_shape(self):
        model = get_resnet18_model(num_classes=10)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 10))

    def test_predict_dummy_image(self):
        img = Image.new('RGB', (224, 224), color='green')
        res = predict(img, model_path="dummy.pth")
        self.assertIn('class_name', res)
        self.assertIn('confidence', res)


if __name__ == "__main__":
    unittest.main()
