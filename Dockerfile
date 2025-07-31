# Start from the official NVIDIA DeepStream Triton image
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# Set the working directory inside the container
USER root
WORKDIR /app

# First, copy only the requirements file. This is a Docker caching optimization.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install the Python packages using pip.
# --no-cache-dir is a good practice to keep the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt


# Now, copy all of your application code into the container's /app directory.
# This comes AFTER installing requirements, so code changes don't trigger
# a full package re-installation, making builds faster.
COPY . .

RUN chmod +x setup_deepstream.sh

COPY requirements_deepstream.txt .
RUN pip install --no-cache-dir -r requirements_deepstream.txt

# Expose FastAPI port
EXPOSE 7862

# Entrypoint: run the FastAPI + DeepStream server
# ENTRYPOINT ["bash", "-lc", "source ~/.bashrc && python3 deepstream_server.py"]
