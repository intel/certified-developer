# Use an official runtime as a parent image
FROM tiangolo/uvicorn-gunicorn-fastapi:latest

# Copy the app directory contents into the container at /app
COPY ./.env .env
COPY ./requirements.txt requirements.txt
COPY ./pyproject.toml pyproject.toml
COPY ./app /app/app

# Install any needed packages specified in requirements.txt
RUN pip install -r ./requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80