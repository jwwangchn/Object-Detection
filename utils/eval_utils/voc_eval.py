# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import cv2


from utils.box_utils import iou_rotate
from utils.box_utils import coordinate_convert
from utils.eval_utils import copy_test_image
from configs import cfgs
import shutil

def parse_rec(filename, mode=0):
    """ Parse a PASCAL VOC xml file 
    mode: bbox -> 0, rbbox -> 1
    """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)

        if mode == 0:
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        else:
            bbox = obj.find('robndbox')
            cx = int(float(bbox.find('cx').text))
            cy = int(float(bbox.find('cy').text))
            w = int(float(bbox.find('w').text))
            h = int(float(bbox.find('h').text))
            angle = float(bbox.find('angle').text)
            angle = int(-angle * 180.0 / np.pi)
            obj_struct['bbox'] = [cx, cy, h, w, angle]
        objects.append(obj_struct)

    return objects

def draw_box(im, bbox, color, save_path = None, mode=0, label = None):
    """
    :param im: read by opencv
    :param bbox: mode=0 -> bbox (xmin, ymin, xmax, ymax) or (cx, cy, w, h)?, mode=1 -> rbbox (cx, cy, w, h, angle)
    :param color: (b, g, r)
    :param save_path: The path of saving image, None -> imshow (default None)
    :param mode: mode=0 -> bbox, mode=1 -> rbbox
    """
    if mode == 0:
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        if label != None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, label, (bbox[0], bbox[1] + 10), font, 1, color, 2, cv2.LINE_AA)
    else:
        rect = coordinate_convert.forward_convert(bbox[: ,:5], with_label=False)
        cv2.drawContours(im, rect, color, 3)
        if label != None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, label, (bbox[0], bbox[1] + 10), font, 1, color, 2, cv2.LINE_AA)
    cv2.imwrite(save_path, im)


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             mode = 0,
             image_path = None,
             visual_algorithm = None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric],
                                mode = 0,
                                image_path)

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    mode: bounding boxes -> 0 or rotation bounding boxes -> 1 (default 0)
    image_path: The path to save iamge (default is None), None -> imshow
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # copy visual image
    draw_image_list = []
    if image_path != None:
        dst_path_ = os.path.join(image_path, visual_algorithm)
        # if fold is not exist, create fold
        if os.path.exists(dst_path_) == False:
            print('visual fold is not exist!!!')
            os.mkdir(dst_path_)
        elif cfgs.DELETE_OLD_VISUAL_FOLD:
            # delete old visual fold
            shutil.rmtree(dst_path_)
            os.mkdir(dst_path_)


        print('Save visual images to: ', os.path.abspath(dst_path_))
        
        if os.listdir(dst_path_):
            print('visual fold is not empty, skip copy processing!!!')
            image_path = dst_path_
            draw_image_list = [_.split('.')[0] for _ in os.listdir(image_path)]
            print(draw_image_list)
        else:
            print('visual fold is empty, start to copy images!!!')
            src_path = os.path.join(imagesetfile.strip('test.txt'), '../..', 'JPEGImages')
            dst_path = dst_path_
            image_path = dst_path_
            draw_image_list = copy_test_image.copy_test_image(imagesetfile, 
                                                                src_path, 
                                                                dst_path, 
                                                                shuffle = cfgs.SHUFFLE_VISUAL_IMAGE, 
                                                                visual_num = cfgs.VISUAL_IMAGE_NUM)
            print(draw_image_list)
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename), mode)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        
        if image_path != None and image_ids[d] in draw_image_list:        
            im = cv2.imread(os.path.join(image_path, image_ids[d] + '.jpg'))
            save_path = os.path.join(image_path, image_ids[d] + '.jpg')
            
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            if mode == 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
            else:
                overlaps = iou_rotate.iou_rotate_calculate(BBGT[:, :5], np.array(bb[:, :5]), use_gpu=False)

            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            if image_path != None and image_ids[d] in draw_image_list:
                # plot bbox
                
                for bbox in BBGT:
                    # Ground Truth -> blue (255, 0, 0)
                    draw_box(im, np.int0(bbox), (255, 0, 0), save_path, mode)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                    if image_path != None and image_ids[d] in draw_image_list:
                        # plot bbox
                        # Right (IoU > 0.5) -> green (0, 255, 0)
                        draw_box(im, np.int0(bb), (0, 255, 0), save_path, mode)
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
            if image_path != None and image_ids[d] in draw_image_list:
                # plot bbox
                # Error (IoU < 0.5) -> red (0, 0, 255)
                draw_box(im, np.int0(bb), (0, 0, 255), save_path, mode)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
