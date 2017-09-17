## Dependencies:
pip install Flask


## Usage:
```python
python mms.py --models model_name1:model_path1 model_name2:model_path2
```

(Currently only loading resnet-18 from local file is supported)

## Example: 

### (1) Single model:
```python
python mms.py --model resnet-18=file://models/resnet-18
```
This will setup a local server serving resnet-18 for prediction.

Try this request http://127.0.0.1:5000/predict/resnet-18?url=https%3A%2F%2Fwww.what-dog.net%2FImages%2Ffaces2%2Fscroll001.jpg , you will see prediction.

### (2) Multi model:
```python
python mms.py --model resnet-18=file://models/resnet-18 resnet-18-2=file://models/resnet-18
```
This will setup a local server serving two resnet-18 models for prediction, I'm re-using resnet-18 model to save space.

I have also registered a user defined function which will find class with probability:
```python
def max_prob(class_prob_kv):
	return dict([max(class_prob_kv.items())])
```

Try this request http://127.0.0.1:5000/predict/all?url=https%3A%2F%2Fwww.what-dog.net%2FImages%2Ffaces2%2Fscroll001.jpg ,
you can see the prediction pair with max probability.


## Description:

mms.py -> This file is flask entry point which can add endpoint, parse url, register user defined functions, setup flask localhost

model_service.py -> Service that will send request to model

model.py -> Model interface which contains single-node/multi-node cluster

mxnet_model.py -> MXNet Model

