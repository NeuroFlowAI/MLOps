#!/bin/bash

# Pull the Docker image
docker pull nvcr.io/nvidia/tritonserver:21.10-py3

# Run the Docker container
docker run --gpus all --rm \
-p8000:8000 -p8001:8001 -p8002:8002 \
-v /home/ubuntu/playground/models:/models \
nvcr.io/nvidia/tritonserver:21.10-py3 \
tritonserver --model-repository=/models
