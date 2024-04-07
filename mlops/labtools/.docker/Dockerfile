# Use an official Python runtime as a parent image.
FROM python:3.12-slim

# Set environment variables to avoid Python buffering and enable unattended installation.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Add ~/.local/bin to PATH for pipx installed packages
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies required for pyenv and other tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev 
    libsqlite3-dev \
    libpq-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev 
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    ca-certificates && \
    # Clean up apt cache to reduce image size.
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pipx, add it to PATH, and install Poetry using pipx
RUN python -m pip install --upgrade pip && \
    pip install pipx && \
    pipx ensurepath && \
    pipx install poetry

# Clone the project repository.
RUN git clone https://github.com/witzbeck/intel_mlops /intel_mlops && \
    # Set the working directory to the cloned repository.
    cd /intel_mlops && \
    # Check out the dev branch.
    git checkout dev

# Set the working directory to the cloned repository.
WORKDIR /intel_mlops/mlops/labtools

# Install the project dependencies.
RUN poetry config virtualenvs.in-project true && \
    poetry install

# Expose a port
EXPOSE 8000
