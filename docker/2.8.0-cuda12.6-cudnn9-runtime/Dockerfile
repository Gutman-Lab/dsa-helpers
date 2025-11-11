FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir dsa-helpers && \
    pip install --no-cache-dir large-image[all] --find-links https://girder.github.io/large_image_wheels && \
    pip install --no-cache-dir histomicstk

# # Dependencies for OpenCV (not needed anymore, but keeping for reference)
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir /app && chmod 777 /app && mkdir /data && chmod 777 /data

WORKDIR /app