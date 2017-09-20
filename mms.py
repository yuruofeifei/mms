import argparse
import sys

from serving_frontend import ServingFrontend
from sdk_generator import SDKGenerator


def parse_args():
	parser = argparse.ArgumentParser(description='MXNet Model Serving')
	
	parser.add_argument('--models', 
						required=True,
						action=StoreDictKeyPair,
						metavar='KEY1=VAL1,KEY2=VAL2...', 
						nargs="+", 
						help='Delopy models')

	parser.add_argument('--gen-api', help='Generate API')
	return parser.parse_args()

class StoreDictKeyPair(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, 'models', {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in values})

class MMS(object):
	def __init__(self):
		self.args = parse_args()
		self.serving_frontend = ServingFrontend()
		self.serving_frontend.register_module('mxnet_vision_model')
		model_class_definations = self.serving_frontend.get_registered_models()
		self.serving_frontend.load_models(self.args.models, 
										  model_class_definations['MXNetVisionModel'])
		self.serving_frontend.add_endpoints({
			'api_description': {
				'endpoint': 'describe_api',
				'method': 'GET',
				'callback': self.describe_api
			},
			'predict': {
				'endpoint': 'predict/<model_name>',
				'method': 'GET',
				'callback': self.predict_endpoint
			},
			'predict_all': {
				'endpoint': 'predict/all',
				'method': 'GET',
				'callback': self.predict_all_endpoint
			}
		})

	def start_model_serving(self):
		self.serving_frontend.start_model_serving()

	# user defined function
	def max_prob(self, class_prob_kv):
		return dict([max(class_prob_kv.items())])

	def prediction_to_html(self, prediction):
		response = ''
		for kv_pair in prediction.items():
			response += 'probability=%f, class=%s <br><br>' % (kv_pair[0], kv_pair[1])
		return response

	# user defined endpoint
	def predict_endpoint(self, model_name):
		url = self.serving_frontend.get_query_string('url')
		return self.prediction_to_html(self.serving_frontend.predict(model_name, url))

	def predict_all_endpoint(self):
		url = self.serving_frontend.get_query_string('url')
		return self.prediction_to_html(self.serving_frontend.predict_across_all_models(url, self.max_prob))

	
	def describe_api(self):
		return self.get_endpoint_mapping()

if __name__ == '__main__':
	mms = MMS()
	mms.start_model_serving()
