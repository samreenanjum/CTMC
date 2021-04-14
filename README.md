# CTMC: Cell Tracking with Mitosis Detection Dataset Challenge
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Anjum_CTMC_Cell_Tracking_With_Mitosis_Detection_Dataset_Challenge_CVPRW_2020_paper.pdf)] [[Website](https://ivc.ischool.utexas.edu/ctmc/)]

[Samreen Anjum](https://www.ischool.utexas.edu/~samreen/) and [Danna Gurari](https://www.ischool.utexas.edu/~dannag/AboutMe.html)

Presented at the [Computer Vision for Microscopy Images (CVMI)](https://cvmi2020.github.io/accepted.html) workshop, [CVPR 2020](http://cvpr2020.thecvf.com/)

This repository contains a Python implementation of the evaluation metric, TRA, inspired from the [Cell Tracking Challenge](https://github.com/CellTrackingChallenge/measures) and modified to support evaluation of tracking multiple objects with bounding boxes in videos.

## Challenge

We are hosting a challenge at the Computer Vision for Microscopy Images (CVMI) workshop at CVPR 2021. This challenge is hosted on the MOT Challenge platform. 

Results Format:
Following the MOTChallenge guidelines, the file format should be the same as the ground truth file, which is a CSV text-file containing one object instance per line. Each line must contain 10 values:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
```

Evaluation:
The official code for evaluating is provided by the MOTChallenge team [here](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official). 
The script to run the evaluation code on the CTMC dataset can be found [here](https://github.com/samreenanjum/CTMC/blob/master/HOTA-metrics/scripts/run_ctmc.py).


## Contact

If you have any questions or suggestions, please contact: [Samreen Anjum](https://www.ischool.utexas.edu/~samreen/) - samreen@utexas.edu


