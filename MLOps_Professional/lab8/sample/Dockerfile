FROM public.ecr.aws/docker/library/python:3.8

# copy assets over to image
COPY /sample /supportbot_chatbot

# set the working directory
WORKDIR /supportbot_chatbot

# install dependancies
RUN apt-get update && apt-get install -y \
python3-dev \
build-essential

# install python dependancies    
RUN pip3 install --user --no-cache-dir -r requirements.txt

# set PATH
ENV PATH=.local/bin:$PATH

# exposing endpoint port
EXPOSE 5000

ENTRYPOINT ["python", "serve.py"]
