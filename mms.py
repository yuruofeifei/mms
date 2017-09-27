import sys

from arg_parser import ArgParser
from serving_frontend import ServingFrontend
from client_sdk_generator import ClientSDKGenerator


class MMS(object):
    def __init__(self):
        self.args = ArgParser.parse_args()
        self.serving_frontend = ServingFrontend()
        
        # Port 
        self.port = self.args.port or 8080
        self.host = '127.0.0.1'

        self.serving_frontend.register_module('mxnet_model_service')
        mode_class_name = 'MXNetBaseService'
        # Register user defined model service
        if self.args.process is not None:
            class_defs = self.serving_frontend.register_module(self.args.process)
            if len(class_defs) < 2:
            	raise Exception('User defined modulde must derive base ModelService.')
            # First class is the base ModelService class
            mode_class_name = class_defs[1].__name__

        # Load models using registered model definitions
        registered_models = self.serving_frontend.get_registered_models()
        ModelClassDef = registered_models[mode_class_name]
        self.serving_frontend.load_models(self.args.models, ModelClassDef)

        # Setup endpoint
        openapi_endpoints = self.serving_frontend.setup_openapi_endpoints(self.host, self.port)

        # Generate client SDK
        if self.args.gen_api is not None:
            ClientSDKGenerator.generate(openapi_endpoints, self.args.gen_api)

    def start_model_serving(self):
        self.serving_frontend.start_model_serving(self.host, self.port)

if __name__ == '__main__':
    mms = MMS()
    mms.start_model_serving()
