import mms

from storage import KVStorage
from model import Model
from mxnet_model import MXNetModel


class ModelService(object):
	def __init__(self, models):
		self.model_storage = KVStorage('model')
		self.func_storage = KVStorage('func')
		for model_name, model_path in models.iteritems():
			self.add_model(model_name, model_path)
		self.register_func('predict', self.predict)

	def register_func(self, func_name, func):
		self.func_storage[func_name] = func

	def get_func(self, func_name):
		return self.func_storage[func_name]

	def add_model(self, model_name, model_path):
		self.model_storage[model_name] = Model.pull_model('MXNetModel', model_path)

	def predict(self, model_name, data, version=None):
		return self.model_storage[model_name].predict(data)
