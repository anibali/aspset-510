**UPDATE 2024-02-04: Test set annotations have now been released. [Click to download](https://archive.org/download/aspset510/aspset510_v1_test-joints_3d.tar.gz).**

# ASPset-510

![ASPset logo](docs/images/aspset_logo.svg)

ASPset-510 (**A**ustralian **S**ports **P**ose Data**set**) is a large-scale video dataset for
the training and evaluation of 3D human pose estimation models. It contains 17 different amateur
subjects performing 30 sports-related actions each, for a total of 510 action clips.

This repository contains Python code for working with ASPset-510.

If you don't want to use these scripts and would prefer to directly download the data yourself,
ASPset-510 is available on the Internet Archive at
[https://archive.org/details/aspset510](https://archive.org/details/aspset510).


## Requirements

### Core

```bash
$ conda env create -f environment.yml
```

* python >= 3.6
* numpy
* [ezc3d](https://github.com/pyomeca/ezc3d)
* [posekit](https://github.com/anibali/posekit)

### GUI (Optional)

```bash
$ conda env update -f environment-gui.yml
```

* [PyOpenGL](http://pyopengl.sourceforge.net/)
* [glfw](https://github.com/FlorianRhiem/pyGLFW)
* matplotlib

### PyTorch (Optional)

```bash
$ conda env update -f environment-torch.yml
```


## Scripts

### Downloading the dataset

`download_data.py` downloads and extracts ASPset-510 data.

Example usage:

```bash
$ python src/aspset510/bin/download_data.py --data-dir=./data
```

Note that by default the original archive files will be downloaded and kept in the `archives`
subdirectory of whichever path you set using `--data-dir`. To set a different path for the
archives, use the `--archive-dir` option. To download the archives without extracting them,
use the `--skip-extraction` option.

### Browsing clips from the dataset

`browse_clips.py` provides a graphical user interface for browsing clips from ASPset-510.

Example usage:

```bash
$ python src/aspset510/bin/browse_clips.py --data-dir=./data
```

![Screenshot of the clip browser GUI](docs/images/browse_clips_gui.jpg)


## Acknowledgments and license

ASPset-510 is brought to you by [La Trobe University](https://www.latrobe.edu.au/) and the
[Australian Institute of Sport](https://www.ais.gov.au/). It is dedicated to the public
domain under the [CC0 1.0 license](https://creativecommons.org/publicdomain/zero/1.0/).

If you find this dataset useful for your own work, please cite the following paper:

```
@article{nibali2021aspset,
  title={{ASPset}: An Outdoor Sports Pose Video Dataset With {3D} Keypoint Annotations},
  author={Nibali, Aiden and Millward, Joshua and He, Zhen and Morgan, Stuart},
  journal={Image and Vision Computing},
  pages={104196},
  year={2021},
  issn={0262-8856},
  doi={https://doi.org/10.1016/j.imavis.2021.104196},
  url={https://www.sciencedirect.com/science/article/pii/S0262885621001013},
  publisher={Elsevier}
}
```
