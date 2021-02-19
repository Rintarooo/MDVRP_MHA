#!/bin/bash

# default:test:env
IMAGE_NAME=${2:-test:env}

if [ "$1" = "build" ]; then
	docker build -t $IMAGE_NAME .
	echo -e "\n\n\ndocker images | head"
	docker images | head
elif [ "$1" = "run" ]; then
	docker run -it --rm  \
		--gpus=all \
		-v ${PWD}:/app \
		$IMAGE_NAME
		#test:env
else
	echo "command should be:
	 ${0} build
	 ${0} run"
fi