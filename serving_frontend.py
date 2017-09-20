from service_manager import ServiceManager
from flask_handler import FlaskRequestHandler


class ServingFrontend(object):
	def __init__(self):
		self.service_manager = ServiceManager()
		self.handler = FlaskRequestHandler()
		self.endpoint_mapping = {}

	def start_model_serving(self):
		self.handler.start_handler()

	def load_models(self, models, ModelClassDef):
		for model_name, model_path in models.iteritems():
			self.service_manager.load_model(model_name, model_path, ModelClassDef)

	def register_module(self, user_defined_module_name):
		model_class_definations = self.service_manager.parse_models_from_module(user_defined_module_name)
		for ModelClassDef in model_class_definations:
			self.service_manager.add_model_to_registry(ModelClassDef.__name__, ModelClassDef)

	def get_registered_models(self, model_names=None):
		if not isinstance(model_names, list) and model_names is not None:
			model_names = [model_names]
		return self.service_manager.get_model_registry(model_names)

	def get_loaded_models(self, model_names=None):
		if not isinstance(model_names, list) and model_names is not None:
			model_names = [model_names]
		return self.service_manager.get_loaded_models(model_names)

	def get_query_string(self, field):
		return self.handler.get_query_string(field)

	def predict_across_all_models(self, data, func):
		return self.service_manager.predict_across_all_models(data, func)

	def predict(self, model_name, data):
		return self.service_manager.predict(model_name, data)

	def get_endpoint_mapping(self):
		return self.endpoint_mapping

	def describe_api(self):
		return self.get_endpoint_mapping()

	def add_endpoints(self, endpoint_mapping):
		for api_name, kwargs in endpoint_mapping.iteritems():
			self.handler.add_endpoint(api_name, **kwargs)
		self.endpoint_mapping.update(endpoint_mapping)


