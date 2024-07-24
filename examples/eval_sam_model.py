# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
import sys
sys.path.append('/home/dhhan/RADIO/')

import argparse
import json
import os
import cv2
import torch.distributed as dist

import numpy as np
import torch
from lvis import LVIS
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import List
from segment_anything import SamPredictor

from examples.amg import sam_model_registry, RADIOVenc
from examples.sam_eval_utils import Clicker, evaluate_predictions_on_coco, evaluate_predictions_on_lvis, get_iou_metric, iou


def bbox_xywh_to_xyxy(bbox: List[int]) -> List[int]:
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def ann_to_mask(ann, h, w):
    if type(ann["segmentation"]) == list:
        rles = mask_util.frPyObjects(ann["segmentation"], h, w)
        rle = mask_util.merge(rles)
    elif type(ann["segmentation"]["counts"]) == list:
        rle = mask_util.frPyObjects(ann["segmentation"], h, w)
    else:
        raise NotImplementedError()

    mask = mask_util.decode(rle) > 0

    return mask


def sync_output(world_size, output, local_rank):
    outs = [None for _ in range(world_size)]
    dist.all_gather_object(outs, output)
    merged_outs = []
    for sublist in outs:
        merged_outs += sublist

    return merged_outs


def predict_mask_from_box(predictor: SamPredictor, bbox: np.ndarray) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


def predict_mask_from_point(
    predictor: SamPredictor, point_coords: np.ndarray, point_labels: np.ndarray
) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


class eval_dataset(Dataset):
    def __init__(self, dataset, image_root, prompt_type, annotation_json_file, source_json_file=None):
        self.dataset = dataset
        self.image_root = image_root
        self.prompt_type = prompt_type
        self.annotation_json_file = annotation_json_file

        if self.dataset == "coco":
            self.images = os.listdir(self.image_root)
            self.images = [os.path.join(self.image_root, image) for image in self.images]
            self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]
        elif self.dataset == "lvis":
            self.images = json.load(open(self.annotation_json_file, "r"))["images"]
            self.images = [
                os.path.join(self.image_root, image["coco_url"].split("/")[-2], image["coco_url"].split("/")[-1])
                for image in self.images
            ]
            self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]
        else:
            raise NotImplementedError()

        if self.prompt_type == "point" or self.prompt_type == "box":
            self.annotations = json.load(open(self.annotation_json_file, "r"))["annotations"]
        elif self.prompt_type == "box_from_detector":
            self.source_json_file = json.load(open(source_json_file))
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if self.prompt_type == "point" or self.prompt_type == "box":
            anns = [ann for ann in self.annotations if ann["image_id"] == self.ids[idx]]
            return {"image_path": image_path, "anns": anns}
        elif self.prompt_type == "box_from_detector":
            detections = [det for det in self.source_json_file if det["image_id"] == self.ids[idx]]
            return {"image_path": image_path, "detections": detections}
        else:
            raise NotImplementedError()


def collate_fn(batch):
    return batch


def run_box(efficientvit_sam, dataloader, save_dir, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
    predictor = SamPredictor(efficientvit_sam)

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
        data = data[0]
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))
        predictor.set_image(sam_image)
        anns = data["anns"]
        for k, ann in enumerate(anns):
            if ann["area"] < 1:
                continue

            sam_mask = ann_to_mask(ann, sam_image.shape[0], sam_image.shape[1])

            bbox = np.array(bbox_xywh_to_xyxy(ann["bbox"]))
            pre_mask = predict_mask_from_box(predictor, bbox)

            miou = iou(pre_mask, sam_mask)

            result = {
                "area": ann["area"],
                "iou": miou,
            }

            cv2.imwrite(f'{save_dir}/img/{data["image_path"].split("/")[-1][:-4]}_{k}.jpg', pre_mask.astype(np.uint8) * 255)
            rle = mask_util.encode(np.array(pre_mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            rle['result'] = result
            with open(f'{save_dir}/json/{data["image_path"].split("/")[-1][:-4]}_{k}.json', 'w') as f:
                json.dump(rle, f, indent=4)
            
            
            output.append(result)

    world_size = int(os.environ["WORLD_SIZE"])
    merged_outs = sync_output(world_size, output, local_rank)

    return merged_outs


def run_point(efficientvit_sam, dataloader, num_click, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
    predictor = SamPredictor(efficientvit_sam)

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
        data = data[0]
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))
        predictor.set_image(sam_image)
        anns = data["anns"]
        for ann in anns:
            if ann["area"] < 1:
                continue

            sam_mask = ann_to_mask(ann, sam_image.shape[0], sam_image.shape[1])

            point_coords_list = []
            point_labels_list = []

            clicker = Clicker(gt_mask=sam_mask)
            pre_mask = np.zeros_like(sam_mask)

            for i in range(num_click):
                clicker.make_next_click(pre_mask)
                point_coords_list.append(clicker.clicks_list[-1].coords[::-1])
                point_labels_list.append(int(clicker.clicks_list[-1].is_positive))
                point_coords = np.stack(point_coords_list, axis=0)
                point_labels = np.array(point_labels_list)

                pre_mask = predict_mask_from_point(predictor, point_coords, point_labels)

            miou = iou(pre_mask, sam_mask)

            result = {
                "area": ann["area"],
                "iou": miou,
            }

            output.append(result)

    world_size = int(os.environ["WORLD_SIZE"])
    merged_outs = sync_output(world_size, output, local_rank)

    return merged_outs


def run_box_from_detector(efficientvit_sam, dataloader, local_rank, save_dir):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
    predictor = SamPredictor(efficientvit_sam)

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
        data = data[0]
        sam_image = Image.open(data["image_path"]).convert("RGB")
        predictor.set_image(np.array(sam_image))
        detections = data["detections"]
        for k, det in enumerate(detections):
            bbox = np.array(bbox_xywh_to_xyxy(det["bbox"]))
            sam_mask = predict_mask_from_box(predictor, bbox)
            cv2.imwrite(f'{save_dir}/img/{data["image_path"].split("/")[-1][:-4]}_{k}.jpg', sam_mask.astype(np.uint8) * 255)
            rle = mask_util.encode(np.array(sam_mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            det["segmentation"] = rle
            with open(f'{save_dir}/json/{data["image_path"].split("/")[-1][:-4]}_{k}.json', 'w') as f:
                json.dump(det, f, indent=4)
            
        output += detections

    world_size = int(os.environ["WORLD_SIZE"])
    merged_outs = sync_output(world_size, output, local_rank)

    return merged_outs


def evaluate(results, prompt_type, dataset, annotation_json_file=None):
    if prompt_type == "point" or prompt_type == "box":
        print(", ".join([f"{key}={val:.3f}" for key, val in get_iou_metric(results).items()]))
    elif prompt_type == "box_from_detector":
        iou_type = "segm"
        if dataset == "coco":
            print(", ".join([f"{key}={val:.3f}" for key, val in get_iou_metric(results).items()]))
            coco_api = COCO(annotation_json_file)
            evaluate_predictions_on_coco(coco_gt=coco_api, coco_results=results, iou_type=iou_type)
        elif dataset == "lvis":
            lvis_api = LVIS(annotation_json_file)
            evaluate_predictions_on_lvis(lvis_gt=lvis_api, lvis_results=results, iou_type=iou_type)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="box", choices=["point", "box", "box_from_detector"])
    parser.add_argument("--num_click", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "lvis"])
    parser.add_argument("--image_root", type=str, default='data/COCO/val2017')
    parser.add_argument("--annotation_json_file", type=str, default='data/COCO/annotations/instances_val2017.json')
    parser.add_argument("--source_json_file", type=str, default='data/COCO/annotations/instances_val2017.json')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default='result/SAM/COCO/')
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    sam = sam_model_registry['vit_h']('models/sam_vit_h_4b8939.pth')

    radio_model = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.1', adaptor_names='sam', vitdet_window_size=16)
    sam.image_encoder = RADIOVenc(radio_model, sam.image_encoder)

    preproc = radio_model.make_preprocessor_external()
    sam.pixel_mean = preproc.norm_mean * 255
    sam.pixel_std = preproc.norm_std * 255
    
    dataset = eval_dataset(
        args.dataset, args.image_root, args.prompt_type, args.annotation_json_file, args.source_json_file
    )
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn
    )

    if args.prompt_type == "point":
        results = run_point(sam, dataloader, args.num_click, local_rank)
    elif args.prompt_type == "box":
        os.makedirs(os.path.join(args.save_dir,'img'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir,'json'), exist_ok=True)
        results = run_box(sam, dataloader, args.save_dir, local_rank)
    elif args.prompt_type == "box_from_detector":
        os.makedirs(os.path.join(args.save_dir,'img'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir,'json'), exist_ok=True)
        results = run_box_from_detector(sam, dataloader, local_rank, args.save_dir)
    else:
        raise NotImplementedError()

    if local_rank == 0:
        evaluate(results, args.prompt_type, args.dataset, args.annotation_json_file)