# CEPDOF_tools
This repository is the Python software toolkit for bounding-box visualization and algorithm evaluation on the [CEPDOF dataset](http://vip.bu.edu/cepdof/).

## Updates
- [May 26, 2020]: Update docstrings and comments. Functionality is unchanged.
- [Apr 17, 2020]: Initial commit

## Annotation Format
CEPDOF's annotation format follows the [COCO dataset](http://cocodataset.org/#home) convention, except that we use **[cx,cy,w,h,degree (clockwise)]** for each bounding box instead of [x1,y1,w,h] in COCO. `/CEPDOF_sample` is a toy sample of the CEPDOF dataset. For more details, please refer to our dataset page shown above .

## CEPDOF API Usage
Put the `cepdof_api.py` in your working directory and make use of the functions in it. Some examples are described below.

**Requirements:**
- Python3 (tested on Python 3.6 and 3.7)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [pycocotools](https://github.com/cocodataset/cocoapi) (for Windows users, please refer to [this GitHub repo](https://github.com/maycuatroi/pycocotools-window))

## Visualization
Example code for parsing and visualizing the annotations is provided in [visualize_demo.ipynb](https://github.com/duanzhiihao/CEPDOF_tools/blob/master/visualize_demo.ipynb).

## Evaluation on CEPDOF
Our evaluation code is built upon [pycocotools](https://github.com/cocodataset/cocoapi) so the usage is similar to it, except that we use **[cx,cy,w,h,degree (clockwise)]** instead of [x1,y1,w,h] for each bounding box. The detection results should be in the JSON format as in `video_0_results.json`. Example code for evaluation on CEPDOF is provided in [eval_demo.ipynb](https://github.com/duanzhiihao/CEPDOF_tools/blob/master/eval_demo.ipynb). 

## Evaluation on HABBOF
**!!TO DO!!** For backward compatibility, we will release code for evaluation on the [HABBOF](https://vip.bu.edu/habbof/) dataset.

## Citation
If you publish any work reporting results on the CEPDOF or the HABBOF dataset, please cite the corresponding paper.
