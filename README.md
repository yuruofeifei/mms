## Usage:
```python
python mms.py --models resnet-18=models/resnet-18.zip [--process mxnet_vision_service] [--gen-api python] [--port 8080]
```
### Arguments:
1. models: required, model_name=model_path pairs, multiple models are supported.
2. process: optional, our system will load input module and will initialize mxnet models with the service defined in the module. The module should contain a valid class extends our base model service with customized preprocess and postprocess.
3. gen-api: optional, this will generate an open-api formated client sdk in build folder.
4. port: optional, default 8080

## Endpoints:
After local server is up, there will be three built-in endpoints:
1. [POST] &nbsp; host:port/\<model-name>/predict       
2. [GET] &nbsp; &nbsp; host:port/ping                        
3. [GET] &nbsp; &nbsp; host:port/api-description             


## Prediction endpoint example:

### 1.Use curl:
```
curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "input0=@white-sleeping-kitten.jpg"
```
```
{
  "prediction": [
    [
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.3166358768939972
      },
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.3160117268562317
      },
      {
        "class": "n04074963 remote control, remote",
        "probability": 0.047916918992996216
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.036371976137161255
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.03163142874836922
      }
    ]
  ]
}
```
### 2.Use generated client code:
  ```python
  import swagger_client
  print swagger_client.DefaultApi().resnet18_predict('white-sleeping-kitten.jpg')
  ```
  ```
  {
    'prediction': 
      "[[{u'class': u'n02123045 tabby, tabby cat', u'probability': 0.3166358768939972}, {u'class': u'n02124075 Egyptian cat', u'probability': 0.3160117268562317}, {u'class': u'n04074963 remote control, remote', u'probability': 0.047916918992996216}, {u'class': u'n02123159 tiger cat', u'probability': 0.036371976137161255}, {u'class': u'n02127052 lynx, catamount', u'probability': 0.03163142874836922}]]"
  }
  ```
  
### Ping endpoint example:
Since ping is a GET endpoint, we can see it in a browser by visiting:

http://127.0.0.1:8080/ping

```
{
  "health": "healthy!"
}
```

### API description example:
This endpoint will list all the apis in OpenAPI compatible format:

http://127.0.0.1:8080/api-description

```
{
  "description": {
    "host": "127.0.0.1:8080", 
    "info": {
      "title": "Model Serving Apis", 
      "version": "1.0.0"
    }, 
    "paths": {
      "/api-description": {
        "get": {
          "operationId": "apiDescription", 
          "produces": [
            "application/json"
          ], 
          "responses": {
            "200": {
              "description": "OK", 
              "schema": {
                "properties": {
                  "description": {
                    "type": "string"
                  }
                }, 
                "type": "object"
              }
            }
          }
        }
      }, 
      "/ping": {
        "get": {
          "operationId": "ping", 
          "produces": [
            "application/json"
          ], 
          "responses": {
            "200": {
              "description": "OK", 
              "schema": {
                "properties": {
                  "health": {
                    "type": "string"
                  }
                }, 
                "type": "object"
              }
            }
          }
        }
      }, 
      "/resnet-18/predict": {
        "post": {
          "consumes": [
            "multipart/form-data"
          ], 
          "operationId": "resnet-18_predict", 
          "parameters": [
            {
              "description": "input0 should be image with shape: [3, 224, 224]", 
              "in": "formData", 
              "name": "input0", 
              "required": "true", 
              "type": "file"
            }
          ], 
          "produces": [
            "application/json"
          ], 
          "responses": {
            "200": {
              "description": "OK", 
              "schema": {
                "properties": {
                  "prediction": {
                    "type": "string"
                  }
                }, 
                "type": "object"
              }
            }
          }
        }
      }
    }, 
    "schemes": [
      "http"
    ], 
    "swagger": "2.0"
  }
}
```

## Multi model setup:
```python
python mms.py --model resnet-18=file://models/resnet-18 vgg16=file://models/vgg16
```
This will setup a local host serving resnet-18 model and vgg16 model on the same port.



## Dependencies:

Flask, MXNet, numpy, JAVA(7+, required by swagger codegen)


## Design:
To be updated

## Testing:
python -m unittest tests/unit_tests/test_serving_frontend
