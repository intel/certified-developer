SHELL := /bin/bash # Use bash syntax

install-python:
	sudo apt install software-properties-common
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt install python3.9
	sudo apt install python3-pip

install-python-packages:
	pip install pandas numpy

install-dockertools:
	sudo apt-get update
	sudo apt install docker.io
	sudo apt  install docker-compose
	sudo gpasswd -a $(USER) docker
	newgrp docker

install-tools: install-python install-python-packages install-dockertools

setup-store:
	# create model stores
	mkdir -p ../store/models/robot_maintenance
	# create data stores
	mkdir -p ../store/datasets/{monitoring,robot_maintenance}
	# create output stores
	mkdir -p ../store/outputs/robot_maintenance

create-data:
	python3 ../robot_maintenance/src/utils/generate_data.py --save_path '../store/datasets/robot_maintenance/'

launch:
	docker-compose -f docker-compose.yaml -p app up

launch-nodata:
	docker-compose -f docker-compose.yaml -p app up

shutdown:
	docker-compose -f docker-compose.yaml -p app down