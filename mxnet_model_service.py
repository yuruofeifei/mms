import mxnet as mx
import numpy as np
import tarfile
import json
import os

from mxnet.gluon.utils import download
from mxnet.io import DataBatch
from model_service import SingleNodeModel, MultiNodesModel

URL_PREFIX = ('http://', 'https://', 's3://')
MODEL_DIR = 'models/'
CONFIG_FILE = 'config.json'

def check_input_shape(inputs, metadata):
	"""
	Check input data shape consistency with metadata.

	Parameters
    ----------
    inputs : List of NDArray
        Input data in NDArray format.
    metadata : dict
        Dictionary containing model metadata.
	"""
	data_names = metadata['input'].keys()
	data_shapes = metadata['input'].values()
	assert isinstance(inputs, list), "Input data must be a list."
	assert len(inputs) == len(data_names), "Input number mismatches with " \
											  "metadata. %d expected but got %d." \
											  % (len(data_names), len(inputs))
	for input, data_shape, data_name in zip(inputs, data_shapes, data_names):
		assert isinstance(input, mx.nd.NDArray), "Each input must be NDArray."
		assert len(input.shape) == \
			   len(data_shape), "Shape dimension of input %s mismatches with " \
							    "metadata. %d expected but got %d." \
							    % (data_name, data_shape,
								   len(input.shape))
		for idx in range(len(input.shape)):
			if idx != 0 and data_shape[idx] != 0:
				assert data_shape[idx] \
					   != input.shape[idx], "Input %s has different shape with " \
											"metadata. %s expected but got %s." \
											% (data_name, data_shape, input.shape)

class MXNetBaseService(SingleNodeModel):
	"""MXNetBaseService defines the fundamental loading model and inference
	   operations when serving MXNet model. This is a base class and needs to be
	   inherited.
	"""
	def __init__(self, path):
		super(MXNetBaseService, self).__init__()
		if path.lower.startswith(URL_PREFIX):
			model_file = download(url=path, path=MODEL_DIR)
			with tarfile.open(model_file) as tar:
				tar.extractall(path=MODEL_DIR)
			file_path = os.path.basename(model_file)
		else:
			file_path = path

		config_file_path = '%s/%s' % (file_path, CONFIG_FILE)
		if not os.path.isfile(config_file_path):
			raise RuntimeError('Config file is not found. Please put config.json '
							   'into the model file directory.')
		with open(config_file_path) as config_file:
			self.metadata = json.load(config_file)
		data_names = self.metadata['input'].keys
		data_shapes = self.metadata['input'].items()

		sym, arg_params, aux_params = mx.model.load_checkpoint(file_path, 0)
		self.mx_model = mx.mod.Module(symbol=sym, context=mx.cpu(),
									  data_names=data_names, label_names=None)
		self.mx_model.bind(for_training=False, data_shapes=data_shapes,
						   label_shapes=self.mx_model.label_shapes)
		self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

	def _inference(self, data):
		"""
        Internal inference methods for MXNet. Run forward computation and
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
		check_input_shape(data, self.metadata)
		return self.mx_model.forward(DataBatch(data))

class MXNetModel(SingleNodeModel):
	def __init__(self, path):
		if path.startswith("file://models/resnet-18"):
			filepath = path[7:]
			sym, arg_params, aux_params = mx.model.load_checkpoint(filepath, 0)
			self.mx_model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
			self.mx_model.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
							   label_shapes=self.mx_model._label_shapes)
			self.mx_model.set_params(arg_params, aux_params, allow_missing=True)
			with open('models/synset.txt', 'r') as f:
				self.labels = [l.rstrip() for l in f]
		else:
			raise Exception("Currently only loading resnet-18 local model is supported!")
	

	def _predict(self, data):
		img = self.get_image(data, show=True)
		# compute the predict probabilities
		self.mx_model.forward(DataBatch([mx.nd.array(img)]))
		prob = self.mx_model.get_outputs()[0].asnumpy()
		# print the top-5
		prob = np.squeeze(prob)
		a = np.argsort(prob)[::-1]
		response = {}
		for i in a[0:5]:
			response[prob[i]] = self.labels[i]
		return response


	def get_image(self, url, show=False):
		# download and show the image
		fname = mx.test_utils.download(url)
		img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
		if img is None:
			return None
		# convert into format (batch, RGB, width, height)
		img = cv2.resize(img, (224, 224))
		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 1, 2)
		img = img[np.newaxis, :]
		return img

