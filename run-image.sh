#!/bin/sh

docker run \
  --name experiment2 \
  --mount type=bind,src="$(pwd)"/models,dst=/models/ \
  --mount type=bind,src="$(pwd)"/data,dst=/data/ \
  --mount type=bind,src="$(pwd)"/reports,dst=/reports/ \
  -d trainer:latest 

