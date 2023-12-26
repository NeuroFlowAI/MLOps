# NeuroFlow MLOps

## Overview
- Model Serving Platform for seamless demonstration of models from multiple ML projects.

## Technologies Used
- Nvidia Triton Inference Server - GPU Inference
  - For more details, visit the [Nvidia Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).
- FastAPI - API serving
- MongoDB - Advanced Data/Log Management

## Expected Effects
- A model serving platform that can be utilized in any project.
- Enables ML engineers without MLOps knowledge to deploy ML models immediately.
- Based on Docker, enabling easy deployment of models from TensorFlow, PyTorch, ONNX, etc.

## How to Run
### Set Up Triton Config (.pbtxt)
```protobuf
name: "wine_prediction_model"
platform: "pytorch_libtorch"
input [
  {
    name: "wine_data__0"
    data_type: TYPE_FP32
    dims: [ -1, 77 ]
  },
  {
    name: "climate_data__1"
    data_type: TYPE_FP32
    dims: [ -1, 7, 22 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]  
  }
]
```


### Pull Docker and Run
```bash
docker pull nvcr.io/nvidia/tritonserver:21.10-py3

docker run --gpus all --rm \
-p8000:8000 -p8001:8001 -p8002:8002 \
-v /home/ubuntu/playground/neuroflow_models:/models\ 
nvcr.io/nvidia/tritonserver:20.08-py3 \
tritonserver --model-repository=/models
```

## Development Notes
- Export Model using `torch.jit.trace`
- Adhere to the naming convention - `NAME__INDEX` format
- Beware of Input/Ouput Dimentions writing `.pbtxt` configs.
