#!/usr/bin/env python
from setuptools import setup, find_packages
print find_packages(exclude=["tests", "build", "app", "models", "dist", "mms.egg-info"])
setup(
    py_modules=['mms', 'arg_parser', 'client_sdk_generator', 'flask_handler', 'log', 'model_service', 'mxnet_model_service',
    'mxnet_vision_service', 'request_handler', 'service_manager', 'serving_frontend', 'storage', 'export_model'],
    name='mms',
    version='0.1-rc1',
    description='MXNet Model Serving',
    url='https://github.com/yuruofeifei/mms',
    keywords='MXNet Serving',
    packages=['utils'],
    install_requires=["Flask", 'Pillow', 'requests'],
    entry_points={
        'console_scripts':['mms=mms:mms', 'mms_export=export_model:export']
    },
    include_package_data=True
)