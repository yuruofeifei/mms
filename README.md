Dependencies:
pip install Flask


Usage:
python mms.py --models model_name1:model_path1 model_name2:model_path2

mms.py -> Flask entry point(Add endpoint, parse arguments, register user defined functions) 

model_service.py -> Service that will talk to model

model.py -> Model interface(single node cluster, multi node cluster)

mxnet_model.py -> MXNet Model