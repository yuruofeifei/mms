import mms

from abc import ABCMeta, abstractmethod


class ModelRegistry(type):
	def __init__(cls, *args, **kwargs):
		setattr(mms, cls.__name__, cls)


class Model(object):
	__metaclass__ = ABCMeta
	__metaclass__ = ModelRegistry

	def __init__(self):
		self.load_processor()

	@abstractmethod
	def predict(self, data):
		pass

	@staticmethod
	def pull_model(name, path):
		return getattr(mms, name)(path)

	def preprocess(self, data, process_type):
		return data

	def postprocess(self, data, process_type):
		return data

	@abstractmethod
	def load_processor(self):
		pass

	def validate_processor(self, process_type):
		return hasattr(self.__class__, process_type)


class SingleNodeModel(Model):
	def load_processor(self):
		config = {'resize': lambda x: x}
		for k, v in config.iteritems():
			setattr(SingleNodeModel, k, v)


	def predict(self, data, preprocess_type=None, postprocess_type=None):
		if preprocess_type:
			data = self.preprocess(data, preprocess_type)

		data = self._predict(data)

		if postprocess_type:
			data = self.postprocess(data, postprocess_type)

		return data

	
	@abstractmethod
	def _predict(self, data):
		return data


class MultiNodesModel(Model):
	pass



