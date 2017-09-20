from model import SingleNodeModel, MultiNodesModel

import mxnet as mx
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


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
		self.mx_model.forward(Batch([mx.nd.array(img)]))
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

