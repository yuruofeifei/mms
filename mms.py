import argparse
import sys
import mms

from flask import Flask, request
from model_service import ModelService
from sdk_generator import SDKGenerator

def max_prob(class_prob_kv):
	return max(class_prob_kv, key=class_prob_kv.get)

class MMS(object):
	def __init__(self):
		self.args = self.parse_args()
		self.service = ModelService(models=self.args.models)
		self.register_user_defined_func('max_prob_across_all_models', max_prob)

		self.app = Flask('mms')
		self.add_endpoint('predict/<model_name>', 'predict', self.predict)


	def add_endpoint(self, endpoint, api_name, func):
		self.app.add_url_rule('/' + endpoint, api_name, func)


	def predict(self, model_name):
		return self.service.predict(model_name, request.args['url'])


	def register_user_defined_func(self, func_name, func):
		self.service.register_func(func_name, func)


	def run(self):
		self.app.run()


	def parse_args(self):
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


if __name__ == '__main__':
	mms = MMS()
	mms.run()
