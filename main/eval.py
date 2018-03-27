from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')

import os
import numpy as np
import cv2
import cPickle

from utils.eval_utils import voc_eval
from utils.draw_utils import draw_pr


if __name__ == "__main__":
    # 1. result file to pkl file
    result_path = '/home/jwwangchn/Documents/Projects/2018-Research-UAV-PP/results/PR/2018-03-25'
    SSD = ('SSD', os.path.join(result_path, 'result_ssd.txt'))
    FasterRCNN = ('Faster R-CNN', os.path.join(result_path, 'result_faster_rcnn.txt'))
    YOLOv2 = ('YOLOv2', os.path.join(result_path, 'result_yolov2.txt'))
    results = [FasterRCNN, SSD, YOLOv2]
    color = {'RRPN': '#f03b20', 'Faster R-CNN': '#2b8cbe', 'SSD': '#fec44f', 'YOLOv2': '#a1d99b'}

    for result in results:
        if os.path.exists('./annots.pkl'):
            os.remove('./annots.pkl')
        detpath = '/home/jwwangchn/data/VOCdevkit/VOC220/Annotations/{}.xml'
        imagesetfile = '/home/jwwangchn/data/VOCdevkit/VOC220/ImageSets/Main/test.txt'
        # TODO: image_path = None -> imshow
        rec, prec, ap = voc_eval.voc_eval(result[1], detpath, imagesetfile, 'person', '.', 0.5, use_07_metric = True, mode = 0, image_path = '../visual', visual_algorithm = result[0])
        with open(os.path.join('../output', result[0] + '_person' + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    # 2. draw pr curve
    PR_path = '../output'
    SSD = ('SSD', os.path.join(PR_path, 'SSD' + '_person_pr.pkl'))
    FasterRCNN = ('Faster R-CNN', os.path.join(PR_path, 'Faster R-CNN' + '_person_pr.pkl'))
    YOLOv2 = ('YOLOv2', os.path.join(PR_path, 'YOLOv2' + '_person_pr.pkl'))

    PR = [FasterRCNN, SSD, YOLOv2]
    color = {'RRPN': '#f03b20', 'Faster R-CNN': '#2b8cbe', 'SSD': '#fec44f', 'YOLOv2': '#a1d99b'}
    draw_pr.draw_pr(PR, '../figures/pr_bbox.pdf', color)

