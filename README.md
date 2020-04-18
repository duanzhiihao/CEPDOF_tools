# CEPDOF_tools
**[Updated Apr 17, 2020]** This repository is the Python software toolkit for bounding-box visualization and algorithm evaluation on the [CEPDOF dataset](http://vip.bu.edu/cepdof/).

## Annotation Format
CEPDOF's annotation format follows the [COCO dataset](http://cocodataset.org/#home) convention. `/CEPDOF_sample` is a toy sample of the CEPDOF dataset. Please refer to our dataset page shown above for more details.

## CEPDOF API Usage
Put the `cepdof_api.py` in your working directory and make use of the functions in it. Some examples are described below.

**Requirements:**
- Python3 (tested on Python 3.6 and 3.7)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [pycocotools](https://github.com/cocodataset/cocoapi) (for Windows users, please refer to [this GitHub repo](https://github.com/maycuatroi/pycocotools-window))

## Visualization
An exmaple for parsing and visualizing the annotations is provided in [visualize_demo.ipynb](https://github.com/duanzhiihao/CEPDOF_tools/blob/master/visualize_demo.ipynb).

## Evaluation on CEPDOF
Our evaluation code is built upon `pycocotools` so the usage is similar to it, except that we require `[cx, cy, w, h, degree (clockwise)]` for each bounding box. An exapmle for evaluation on CEPDOF is provided in [eval_demo.ipynb](https://github.com/duanzhiihao/CEPDOF_tools/blob/master/eval_demo.ipynb). The detection results should be in the JSON format like `video_0_results.json`.

## Evaluation on HABBOF
**!!TO DO!!** For backward compatibility, we will release code for evaluation on the [HABBOF](https://vip.bu.edu/habbof/) dataset.

## Citation
If you publish any work reporting results on the CEPDOF or HABBOF dataset, please cite the corresponding paper.
