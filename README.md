# ASPset-510

![ASPset logo](docs/images/aspset_logo.svg)

ASPset-510 (**A**ustralian **S**ports **P**ose Data**set**) is a large-scale video dataset for
the training and evaluation of 3D human pose estimation models. It contains 17 different amateur
subjects performing 30 sports-related actions each, for a total of 510 action clips.

This repository contains Python code for working with ASPset-510.


## Requirements

### Mandatory

```bash
$ conda env create -f environment.yml
```

* python >= 3.6
* numpy
* [ezc3d](https://github.com/pyomeca/ezc3d)
* [posekit](https://github.com/anibali/posekit)

### Optional

```bash
$ conda env update environment-extra.yml
```

For GUI parts of the code:

* [PyOpenGL](http://pyopengl.sourceforge.net/)
* [glfw](https://github.com/FlorianRhiem/pyGLFW)
* matplotlib


## Acknowledgments

ASPset-510 is brought to you by [La Trobe University](https://www.latrobe.edu.au/) and the
[Australian Institute of Sport](https://www.ais.gov.au/).
