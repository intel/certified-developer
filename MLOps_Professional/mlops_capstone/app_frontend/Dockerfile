FROM public.ecr.aws/docker/library/python:3.8@sha256:d411270700143fa2683cc8264d9fa5d3279fd3b6afff62ae81ea2f9d070e390c

# copy assets over to image
COPY . /app_frontend

# set the working directory
WORKDIR /app_frontend

# install dependancies
RUN pip3 install --user --no-cache-dir -r requirements.txt

# set PATH
ENV PATH="/root/.local/bin:${PATH}"

# exposing endpoint port
EXPOSE 5005

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port", "5005", "--server.address", "0.0.0.0"]