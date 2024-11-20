# Digital Slide Archive (DSA) Helpers
Digital Slide Archive Python utility library.

This library is available for installation through [Python Package Index (PyPI)](https://pypi.org/).

This library was tested using Python version 3.11.8 and uses the dependencies described in requirements.txt.

This Python PyPI package is found [here](https://pypi.org/project/dsa-helpers/).

## Extra dependencies
If you are getting a large image error, then install large image using the instructions provided in their [GitHub page](https://github.com/girder/large_image#:~:text=pip%20install%20large%2Dimage%5Ball%5D%20%2D%2Dfind%2Dlinks%20https%3A//girder.github.io/large_image_wheels).

## Instructions for Development
1. Install requirements for building and distributing package:
```
$ python3 -m pip install --upgrade build  # for building package
$ python3 -m pip install --upgrade twine  # for uploading to PyPI
```
2. When ready to build:
    - Modify "pyproject.toml" file, change the "version" key to a different value than a version already pushed.
    - Run ```$ python3 -m build```. This will create a dist directory and put your new wheel and tar distribution files there.
3. To install locally for development: ```$ pip3 install dist/dsa_helpers-*.whl```, choosing the wheel version you want to test.
4. Upload to PyPI using twine: ```python3 -m twine upload --repository pypi dist/*```
    * You can specify specific files to upload, the line above pushes everything in dist directory.