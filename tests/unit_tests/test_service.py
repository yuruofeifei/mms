import os
import sys
sys.path.append('../..')

import unittest
import mock
import mxnet as mx
from mxnet_vision_service import MXNetVisionService as mx_vision_service
from utils.mxnet_utils import Image

class TestServingFrontend(unittest.TestCase):
    def _train_and_export(self):
        num_class = 10
        data1 = mx.sym.Variable('data1')
        data2 = mx.sym.Variable('data2')
        conv1 = mx.sym.Convolution(data=data1, kernel=(2, 2), num_filter=2, stride=(2, 2))
        conv2 = mx.sym.Convolution(data=data2, kernel=(3, 3), num_filter=3, stride=(1, 1))
        pooling1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(1, 1), pool_type="avg")
        pooling2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), stride=(1, 1), pool_type="max")
        flatten1 = mx.sym.flatten(data=pooling1)
        flatten2 = mx.sym.flatten(data=pooling2)
        sum = mx.sym.sum(data=flatten1, axis=1) + mx.sym.sum(data=flatten2, axis=1)
        fc = mx.sym.FullyConnected(data=sum, num_hidden=num_class)
        sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')

        dshape1 = (10, 3, 64, 64)
        dshape2 = (10, 3, 32, 32)
        lshape = (10,)

        mod = mx.mod.Module(symbol=sym, data_names=('data1', 'data2'),
                            label_names=('softmax_label',))
        mod.bind(data_shapes=[('data1', dshape1), ('data2', dshape2)],
                 label_shapes=[('softmax_label', lshape)])
        mod.init_params()
        mod.init_optimizer(optimizer_params={'learning_rate': 0.01})

        data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                           mx.nd.random.uniform(5, 15, dshape2)],
                                     label=[mx.nd.ones(lshape)])
        mod.forward(data_batch)
        mod.backward()
        mod.update()
        signature = {'input_type': 'image/jpeg', 'output_type': 'application/json'}
        with open('synset.txt', 'w') as synset:
            for i in range(10):
                synset.write('test label %d\n' % (i))
        mod.export_serving('test', 0, signature, use_synset=True)

    def test_vision_init(self):
        self._train_and_export()
        model_path = 'test.zip'
        service = mx_vision_service(path=model_path)
        assert hasattr(service, 'labels'), "Fail to load synset file from model archive."
        assert len(service.labels) > 0, "Labels attribute is empty."

    def test_vision_inference(self):
        self._train_and_export()
        model_path = 'test.zip'
        service = mx_vision_service(path=model_path)

        raw_image1 = 'input1.jpg'
        raw_image2 = 'input2.jpg'

        # Test same size image inputs
        data1 = mx.nd.random_uniform(0, 255, shape=(3, 64, 64))
        data2 = mx.nd.random_uniform(0, 255, shape=(3, 32, 32))
        Image.write(raw_image1, data1)
        Image.write(raw_image2, data2)
        img_buf1 = open(raw_image1, 'rb').read()
        img_buf2 = open(raw_image1, 'rb').read()

        output = service.inference([img_buf1, img_buf2])
        assert len(output[0]) == 5

        # test different size image inputs
        data1 = mx.nd.random_uniform(0, 255, shape=(3, 96, 96))
        data2 = mx.nd.random_uniform(0, 255, shape=(3, 24, 24))
        Image.write(raw_image1, data1)
        Image.write(raw_image2, data2)
        img_buf1 = open(raw_image1, 'rb').read()
        img_buf2 = open(raw_image1, 'rb').read()

        output = service.inference([img_buf1, img_buf2])
        assert len(output[0]) == 5

    def runTest(self):
        self.test_vision_init()
        self.test_vision_inference()
