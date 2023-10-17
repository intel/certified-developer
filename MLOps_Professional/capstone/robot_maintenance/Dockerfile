FROM public.ecr.aws/docker/library/python:3.8

# copy assets over to image
COPY /src /robot_maintenance

# set the working directory
WORKDIR /robot_maintenance

# install dependancies
RUN pip3 install --user --no-cache-dir -r requirements.txt

# set PATH
ENV PATH=.local/bin:$PATH

# exposing endpoint port
EXPOSE 5000

ENTRYPOINT ["python", "serve.py"]