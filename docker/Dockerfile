FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Dependencies for OpenCV.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --no-cache-dir dsa-helpers
RUN pip install large-image[all] --find-links https://girder.github.io/large_image_wheels