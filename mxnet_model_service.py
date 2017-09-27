import mxnet as mx
import numpy as np
import zipfile
import json
import os

from mxnet.gluon.utils import download
from mxnet.io import DataBatch
from model_service import SingleNodeService, MultiNodesService, URL_PREFIX

SIGNATURE_FILE = 'signature.json'


def _check_input_shape(inputs, signature):
    """Check input data shape consistency with signature.

    Parameters
    ----------
    inputs : List of NDArray
        Input data in NDArray format.
    signature : dict
        Dictionary containing model signature.
    """
    assert isinstance(inputs, list), "Input data must be a list."
    assert len(inputs) == len(signature['input']), "Input number mismatches with " \
                                           "signature. %d expected but got %d." \
                                           % (len(signature['input']), len(inputs))
    for input, sig_input in zip(inputs, signature['input']):
        assert isinstance(input, mx.nd.NDArray), "Each input must be NDArray."
        assert len(input.shape) == \
               len(sig_input['data_shape']), "Shape dimension of input %s mismatches with " \
                                "signature. %d expected but got %d." \
                                % (sig_input['data_name'], len(sig_input['data_shape']),
                                   len(input_shape))
        for idx in range(len(input.shape)):
            if idx != 0 and sig_input['data_shape'][idx] != 0:
                assert sig_input['data_shape'][idx] == \
                       input.shape[idx], "Input %s has different shape with " \
                                         "signature. %s expected but got %s." \
                                         % (sig_input['data_name'], sig_input['data_shape'],
                                            input.shape)


class MXNetBaseService(SingleNodeService):
    """MXNetBaseService defines the fundamental loading model and inference
       operations when serving MXNet model. This is a base class and needs to be
       inherited.
    """
    def __init__(self, path, synset=None, ctx=mx.cpu()):
        super(MXNetBaseService, self).__init__(path, ctx)
        curr_dir = os.getcwd()
        model_file = download(url=path, path=curr_dir) \
                     if path.lower().startswith(URL_PREFIX) else path
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        model_dir = '%s/%s' % (os.path.dirname(model_file), model_name)
        with zipfile.ZipFile(model_file) as zip:
            zip.extractall(path=os.path.dirname(model_file))

        signature_file_path = '%s/%s' % (model_dir, SIGNATURE_FILE)
        if not os.path.isfile(signature_file_path):
            raise RuntimeError('Signature file is not found. Please put signature.json '
                               'into the model file directory.')
        with open(signature_file_path) as signature_file:
            self.signature = json.load(signature_file)
        data_names = []
        data_shapes = []
        for input in self.signature['input']:
            data_names.append(input['data_name'])
            # Replace 0 entry in data shape with 1 for binding executor.
            data_shape = input['data_shape']
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1
            data_shapes.append((input['data_name'], tuple(data_shape)))

        # Load MXNet module
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, model_name), 0)
        self.mx_model = mx.mod.Module(symbol=sym, context=mx.cpu(),
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

        # Read synset file
        # If synset is not specified, check whether model archive contains synset file.
        archive_synset = '%s/synset.txt' % (model_dir)
        if synset is None and os.path.isfile(archive_synset):
            synset = archive_synset
        if synset:
            self.labels = [line.strip() for line in open(synset).readlines()]



    def _inference(self, data):
        """Internal inference methods for MXNet. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        """
        # Check input shape
        _check_input_shape(data, self.signature)
        self.mx_model.forward(DataBatch(data))
        return self.mx_model.get_outputs()

