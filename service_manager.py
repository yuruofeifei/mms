import ast
import inspect

from storage import KVStorage
from model_service import ModelService, SingleNodeService, MultiNodesService


class ServiceManager(object):
	def __init__(self):
		# registry for model defination and user defined functions
		self.model_registry = KVStorage('model')
		self.func_registry = KVStorage('func')

		# loaded models
		self.loaded_models = KVStorage('loaded_model')

	def register_user_defined_func(self, func_name, func):
		self.func_registry[func_name] = func

	def get_model_registry(self, model_names=None):
		if model_names is None:
			return self.model_registry
		return {model_name: self.model_registry[model_name] for model_name in model_names}

	def add_model_to_registry(self, model_name, ModelClassDef):
		self.model_registry[model_name] = ModelClassDef

	def get_loaded_models(self, model_names=None):
		if model_names is None:
			return self.loaded_models
		return {model_name: self.loaded_models[model_name] for model_name in model_names}

	def get_user_defined_func(self, func_name):
		return self.func_registry[func_name]

	def load_model(self, model_name, model_path, ModelClassDef):
		self.loaded_models[model_name] = ModelClassDef(model_path)

	def predict(self, model_name, data):
		return self.loaded_models[model_name].predict(data)

	def parse_models_from_module(self, user_defined_module_name):
		module = __import__(user_defined_module_name)
		classes = [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
		return filter(lambda cls: cls is not ModelService and issubclass(cls, ModelService), classes)
