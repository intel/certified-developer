FROM public.ecr.aws/docker/library/python:3.8@sha256:d411270700143fa2683cc8264d9fa5d3279fd3b6afff62ae81ea2f9d070e390c

# copy assets over to image
COPY . /pickerbot

# set the working directory
WORKDIR /pickerbot

# install dependancies
RUN apt-get update && apt-get install -y \
python3-dev \
build-essential

# set PATH
ENV PATH=.local/bin:$PATH

# install python dependancies    
RUN pip3 install --user --no-cache-dir -r requirements.txt

# exposing endpoint port
EXPOSE 5000

ENTRYPOINT ["python", "serve.py"]
