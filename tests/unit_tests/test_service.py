import os
import sys
sys.path.append('../..')

import unittest
import mock
import mxnet as mx
from mxnet_vision_service import MXNetVisionService as mx_vision_service
from utils.mxnet_utils import Image

class TestServingFrontend(unittest.TestCase):

    def test_vision_init():
        model_path = '../../models/resnet-18.zip'
        service = mx_vision_service(path=model_path)
        assert hasattr(service, 'labels'), "Fail to load synset file from model archive."
        assert len(service.labels) > 0, "Labels attribute is empty."

    def test_vision_inference():
        model_path = '../../models/resnet-18.zip'
        batch_size = 1
        output_length = 1000
        raw_image = 'input.jpg'
        data = mx.nd.random_uniform(0, 255, shape=(3, 224, 224))
        Image.write(raw_image, data)
        img_buf = open(raw_image, 'rb').read()

        service = mx_vision_service(path=model_path)
        output = service.inference([img_buf])
        assert output[0].shape == (batch_size, output_length)
        os.remove(raw_image)

    def runTest(self):
        test_vision_init()
        test_vision_inference()

    