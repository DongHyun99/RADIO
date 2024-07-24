import os
import json
import numpy as np
import cv2

from tqdm import tqdm
from glob import glob

from pycocotools.coco import COCO
from pycocotools import mask as mask_util

from examples.eval_sam_model import evaluate


if __name__=='__main__':
    
    eval_dir = 'result/SAM/COCO/json/'
    target_dir = 'data/COCO/annotations/instances_val2017.json'
    
    results = []
    
    for i in tqdm(glob(os.path.join(eval_dir, '*.json'))):
        with open(i, 'r') as f:
            j = json.load(f)
        results.append(j['result'])
        
    evaluate(results, 'box', 'coco', target_dir)
       
    
    
    '''
    eval_dir = 'result/SAM/COCO/json/'

    with open('data/COCO/coco_vitdet.json', 'r') as f:
        coco_json = json.load(f)

    bbox = [d['bbox'] for d in coco_json]
    num_class = max([d['category_id'] for d in coco_json])

    se_seg = [i for i in coco_json if i['image_id']==139]
    
    new_img = np.zeros((426, 640))
    
    for idx, im in enumerate(se_seg):
        
        inst_seg = mask_util.decode(im['segmentation']) * idx+1
        new_img += inst_seg
        
    cv2.imwrite('dvc.jpg', new_im
    '''