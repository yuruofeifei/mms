import os
import sys
sys.path.append('../..')

import mxnet as mx
from mxnet_vision_service import MXNetVisionService as mx_vision_service

def test_vision_init():
    model_path = '../../models/resnet-18.zip'
    service = mx_vision_service(path=model_path)
    assert hasattr(service, 'labels'), "Fail to load synset file from model archive."
    assert len(service.labels) > 0, "Labels attribute is empty."

def test_vision_inference():
    model_path = '../../models/resnet-18.zip'
    batch_size = 1
    output_length = 1000
    data = mx.nd.random_uniform(shape=(batch_size, 3, 224, 224))

    service = mx_vision_service(path=model_path)
    output = service._inference([data])
    assert output[0].shape == (batch_size, output_length)


if __name__ == '__main__':
    test_vision_init()
    test_vision_inference()