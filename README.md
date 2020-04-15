# CEPDOF_tools
Python code for visualizing and evaluating on the CEPDOF dataset `link to the dataset page`.

# Requirements
- opencv-python
- [pycocotools](https://github.com/cocodataset/cocoapi)

# Annotation Format
CEPDOF's annotation format follows the [COCO dataset](http://cocodataset.org/#home) convention. `/CEPDOF_sample` is a toy sample of the CEPDOF dataset. Please refer to the dataset page `link to the dataset page` for more details.

# CEPDOF API Usage
Put the `cepdof_api.py` in your working directory and make use of the functions in it. Some examples are described below.

# Visualization
An exmaple for parsing and visualizing the annotations is provided in `visualize_demo.ipynb`.

# Evaluation
Our evaluation code is built upon `pycocotools` so the usage is similar to it, except that we require `[cx, cy, w, h, degree (clockwise)]` for each bounding box. An exapmle for evaluation on CEPDOF is provided in `eval_demo.ipynb`. The detection results should be in the JSON format like `video_0_results.json`.

# Citation
`TODO`
