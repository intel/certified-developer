SHELL := /bin/bash # Use bash syntax

setup-requirements:
	pip install -r ../robot_maintenance/src/requirements.txt

setup-store:
	# create model stores
	mkdir -p ../store/models/robot_maintenance
	# create data stores
	mkdir -p ../store/datasets/{monitoring,robot_maintenance}
	# create output stores
	mkdir -p ../store/outputs/robot_maintenance

create-data:
	python3 ../robot_maintenance/src/utils/generate_data.py --save_path '../store/datasets/robot_maintenance/'
