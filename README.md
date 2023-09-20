# Computer Vision Utilities
Computer vision tools for computer vision engineers.

## Features

### TODO
* Inprove documentation on installing on linux, apple silicon and windows
* Refactor bounding boxes to shapely geometry



## Acknowledgments & Credits
First and foremost, we would like to express our deepest gratitude to Odd Industries for their incredible work on the first version of this library. Their dedication to coding and openly releasing the initial version has paved the way for the development of this project, which has since evolved and grown.

While the API for this release may not be backward compatible, the invaluable contributions from Odd Industries have helped shape the general concept of the OpenCV pipeline that we have embraced in this library. Their pioneering work has served as a solid foundation, enabling us to build upon and improve the project further.

As Entel Ocean we are happy to take this library to the furure

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.

## Install

## Install in dev mode (2023)

1. Create a conda enviroment like:

```shell
conda create -n "cv310" python=3.10
```

2. Install this module
```bash
pip install -e .
```

## Build (2024)

```bash
python setup.py sdist bdist_wheel
```

## Wheel install (2024)
```bash
pip install --upgrade cvutils-x.y.z-py3-none-any.whl
```

# Usage
```bash
cvutils test
```


## Release/liberation policy
Odd Industries's cvutils had a many functionality with wide range of software engineering quality (documentation, abstraction, testing, updated requirements, etc). That same issue made a lot of work to release it open responsibly. At Entel Ocean, having access to previous code and engineer, we will be releasing only features that we are committed to use and support. If this repo opening takes traction, we'll open an issue where the community also could bring old features to present.
