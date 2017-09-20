from abc import ABCMeta, abstractmethod


class Model(object):
	__metaclass__ = ABCMeta

	def __init__(self, path):
		self.context = {}
		self.load_processor()

	@abstractmethod
	def predict(self, data, preprocess_type=None, postprocess_type=None):
		pass

	def _preprocess(self, data, process_type):
		return data

	def _postprocess(self, data, process_type):
		return data

	@abstractmethod
	def _load_processor(self):
		pass

	def _validate_processor(self, process_type):
		return hasattr(self.__class__, process_type)


class SingleNodeModel(Model):
	def _load_processor(self):
		config = {
			'resize': lambda x: x
		}
		for k, v in config.iteritems():
			setattr(SingleNodeModel, k, v)


	def predict(self, data, preprocess_type=None, postprocess_type=None):
		if preprocess_type:
			data = self._preprocess(data, preprocess_type)

		data = self._predict(data)

		if postprocess_type:
			data = self._postprocess(data, postprocess_type)

		return data

	
	@abstractmethod
	def _predict(self, data):
		return data


class MultiNodesModel(Model):
	pass



