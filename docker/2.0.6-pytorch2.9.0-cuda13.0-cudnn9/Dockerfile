FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=TRUE
ARG DEBIAN_FRONTEND=noninteractive

# Install large image and its tile sources.
RUN pip install large-image[all] --find-links https://girder.github.io/large_image_wheels

# Install dsa-helpers.
RUN pip install --no-cache-dir dsa-helpers==2.0.6

# Create a cache directory.
RUN mkdir /.cache
RUN chmod 777 /.cache