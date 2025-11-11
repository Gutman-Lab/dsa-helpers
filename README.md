# Digital Slide Archive (DSA) Helpers
Digital Slide Archive Python utility library.

This library is available for installation through [Python Package Index (PyPI)](https://pypi.org/).

This library was tested using Python version 3.11.8 and uses the dependencies described in requirements.txt.

This Python PyPI package is found [here](https://pypi.org/project/dsa-helpers/).

## Installation instructions
DSA Helpers depends on large image and large image eager iterator and OpenCV, which you must install separately. Also, you must install the eager iterator from wheel and it should be installed at the end. Python 3.12 or below is supported, Python 3.14 is not currently supported!

1. Install large with all tile sources:
```
$ pip install large-image[all] --find-links https://girder.github.io/large_image_wheels
```
2. Install OpenCV, either the headless version (better for running in Docker) or full version
3. Install dsa-helpers from pip
4. Pip install the large image with eager iterator wheel, use the --force flag and you can safely igonore the warnings that you get about some large image library compatibilities

A readthedocs for this library can be found [here](https://david-andrew-gutman-dsa-helpers.readthedocs-hosted.com/en/latest/index.html).