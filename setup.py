#!/usr/bin/env python
from setuptools import setup, find_packages
setup(
    py_modules=['mms', 'arg_parser', 'client_sdk_generator', 'flask_handler', 'log', 'model_service', 'mxnet_model_service',
    'mxnet_vision_service', 'request_handler', 'service_manager', 'serving_frontend', 'storage', 'export_model'],
    name='mms',
    version='0.1-dev2',
    description='MXNet Model Serving',
    url='https://github.com/yuruofeifei/mms',
    keywords='MXNet Serving',
    packages=['utils'],
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests'],
    entry_points={
        'console_scripts':['mms=mms:mms', 'mms_export=export_model:export']
    },
    include_package_data=True
)