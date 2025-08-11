FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN pip install --no-cache-dir dsa-helpers
RUN pip install --no-cache-dir large-image[all] --find-links https://girder.github.io/large_image_wheels
RUN python -m pip install --no-cache-dir histomicstk --find-links https://girder.github.io/large_image_wheels

# Dependencies for OpenCV.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir /app
RUN chmod 777 /app

RUN mkdir /data
RUN chmod 777 /data