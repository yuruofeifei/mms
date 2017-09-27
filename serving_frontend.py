import json
from functools import partial

from service_manager import ServiceManager
from flask_handler import FlaskRequestHandler


class ServingFrontend(object):
    def __init__(self):
        self.service_manager = ServiceManager()
        self.handler = FlaskRequestHandler()

    def start_model_serving(self, host, port):
        self.handler.start_handler(host, port)

    def load_models(self, models, ModelClassDef):
        for model_name, model_path in models.iteritems():
            self.service_manager.load_model(model_name, model_path, ModelClassDef)

    def register_module(self, user_defined_module_name):
        model_class_definations = self.service_manager.parse_models_from_module(user_defined_module_name)
        for ModelClassDef in model_class_definations:
            self.service_manager.add_model_to_registry(ModelClassDef.__name__, ModelClassDef)
        return model_class_definations

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

    def add_endpoint(self, api_definition, callback, **kwargs):
        endpoint = api_definition.keys()[0]
        method = api_definition[endpoint].keys()[0]
        api_name = api_definition[endpoint][method]['operationId']
        self.handler.add_endpoint(api_name, endpoint, partial(callback, **kwargs), [method.upper()])

    def setup_openapi_endpoints(self, host, port):
        models = self.service_manager.get_loaded_models()
        # TODO: not hardcode host:port
        self.openapi_endpoints = {
            'swagger': '2.0',
            'info': {
                'version': '1.0.0',
                  'title': 'Model Serving Apis'
              },
              'host': host + ':' + str(port),
              'schemes': ['http'],
              'paths': {},
          }


        # 1. Predict endpoints
        for model_name, model in models.iteritems():
            input_type = model.signature['input_type']
            inputs = model.signature['inputs']
            output_type = model.signature['output_type']
            
            # Contruct predict openapi specs
            endpoint = '/' + model_name + '/predict'
            predict_api = {
                endpoint: {
                    'post': {
                        'operationId': model_name + '_predict', 
                        'consumes': ['multipart/form-data'],
                        'produces': [output_type],
                        'parameters': [],
                        'responses': {
                            '200': {}
                        }
                    }
                }
            }
            input_names = ['input' + str(idx) for idx in range(len(inputs))]
            # Setup endpoint for each model
            for idx in range(len(inputs)):
                # Check input content type to set up proper openapi consumes field
                if input_type == 'application/json':
                    parameter = {
                        'in': 'formData',
                        'name': input_names[idx],
                        'description': '%s should tensor with shape: %s' % 
                            (input_names[idx], inputs[idx]['data_shape'][1:]),
                        'required': 'true',
                        'schema': {
                            'type': 'string'
                        }
                    }
                elif input_type == 'image/jpeg':
                    parameter = {
                        'in': 'formData',
                        'name': input_names[idx],
                        'description': '%s should be image with shape: %s' % 
                            (input_names[idx], inputs[idx]['data_shape'][1:]),
                        'required': 'true',
                        'type': 'file'
                    }
                else:
                    raise Exception('%s is not supported for input content-type' % input_type)
                predict_api[endpoint]['post']['parameters'].append(parameter)

            # Contruct openapi response schema
            if output_type == 'application/json':
                responses = {
                    'description': 'OK',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'prediction': {
                                'type': 'string'
                            }
                        }
                    }
                }
            elif output_type == 'image/jpeg':
                responses = {
                    'description': 'OK',
                    'schema': {
                        'type': 'file'
                    }
                }
            else:
                raise Exception('%s is not supported for output content-type' % output_type)
            predict_api[endpoint]['post']['responses']['200'].update(responses) 

            self.openapi_endpoints['paths'].update(predict_api)

            # Setup Flask endpoint for predict api
            self.add_endpoint(predict_api, 
                              self.predict_callback, 
                              model=model,
                              input_names=input_names)


        # 2. Ping endpoints
        ping_api = {
            '/ping': {
                'get': {
                    'operationId': 'ping', 
                    'produces': ['application/json'],
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'health': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
                
            }
        }
        self.openapi_endpoints['paths'].update(ping_api)
        self.add_endpoint(ping_api, self.ping_callback)


        # 3. Describe apis endpoints
        api_description_api = {
            '/api-description': {
                'get': {
                    'produces': ['application/json'],
                    'operationId': 'apiDescription', 
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'description': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
                
            }
        }
        self.openapi_endpoints['paths'].update(api_description_api)
        self.add_endpoint(api_description_api, self.describe_api)

        return self.openapi_endpoints
    
    def ping_callback(self, **kwargs):
        try:
            for model in self.service_manager.loaded_models.values():
                model.ping()
        except:
            return self.handler.jsonify({'health': 'unhealthy!'})

        return self.handler.jsonify({'health': 'healthy!'})

    def describe_api(self, **kwargs):
        return self.handler.jsonify({'description': self.openapi_endpoints})

    def predict_callback(self, **kwargs):
        model = kwargs['model']
        input_type = model.signature['input_type']
        output_type = model.signature['output_type']
        input_names = kwargs['input_names']

        if input_type == 'application/json':
            data = map(self.handler.get_form_data, input_names)
        elif input_type == 'image/jpeg':
            data = [self.handler.get_file_data(name).read() for name in input_names]
        else:
            raise Exception('%s is not supported for input content-type' % input_type)

        if output_type == 'application/json':
            return self.handler.jsonify({'prediction': model.inference(data)})
        elif output_type == 'image/jpeg':
            return self.handler.send_file(model.inference(data), output_type)
        else:
            raise Exception('%s is not supported for output content-type' % output_type)

