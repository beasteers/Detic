# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import glob
import json
import pickle
import fire
import numpy as np
import pandas as pd
import cv2
import tqdm

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from detectron2.utils.visualizer import Visualizer

from detic.modeling.text.text_encoder import build_text_encoder

from shapely.geometry import Polygon, MultiPolygon


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["custom_load_json", "custom_register_instances"]


# vocab

object_classes = [
    # 'background',
    'left hand',
    'right hand',
    'object held in left hand',
    'object held in right hand',
    'object held in both hands',
    'object acted upon by left hand',
    'object acted upon by right hand',
    'object acted upon by both hands',
]

# Main dataset annotation file: ./path/to/dataset/egohos_train.json


def create_dataset(dataset_dicts):
    
    return {
        "info": {
            "description": "EGOHOS: Ego-centric Hand-Object Segmentation"
        },
        "licenses": [],
        "images": [
            {"id": d['image_id'], "width": d['width'], "height": d['height'], "file_name": d['file_name']}
            for d in dataset_dicts
        ],
        "annotations": [
            # a: category_id, segmentation, bbox, bbox_mode, area
            {"image_id": d["image_id"], "is_crowd": 0, **a}
            for d in dataset_dicts
            for a in d['annotations']
        ],
        "categories": [
            {"id": i+1, "name": c}
            for i, c in enumerate(object_classes)
        ]
    }




def create_dataset_dict(image_root):
    # get file list
    img_fs = sorted(glob.glob(os.path.join(image_root, 'image/*.jpg')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in img_fs]
    mask_fs = [os.path.join(image_root, f'label/{n}.png') for n in names]
    
    # load annotations for each file
    dataset_dicts = []
    for i, (fname_im, fname_mask) in tqdm.tqdm(enumerate(zip(img_fs, mask_fs)), desc=f'loading {image_root}...', total=len(img_fs)):
        mask = cv2.imread(fname_mask, cv2.IMREAD_GRAYSCALE)
        dataset_dicts.append({
            "file_name": os.path.relpath(fname_im, image_root),
            "height": mask.shape[0],
            "width": mask.shape[1],
            "image_id": i,
            # "pos_category_ids": [],
            # "neg_category_ids": [],
            # "not_exhaustive_category_ids": [],
            # "captions": [],
            # "caption_features": [],
            "annotations": _get_annotations(mask),
        })

    i = 0
    for d in dataset_dicts:
        for a in d['annotations']:
            a['id'] = i
            i += 1
    return dataset_dicts



def _get_annotations(mask):
    # convert masks to contours (index 0 is background)
    anns = [
        _contour_to_ann(_bool_mask_to_contour(mask == i+1), category_id=i) 
        for i in range(len(object_classes))
    ]
    return [d for d in anns if d]


def _bool_mask_to_contour(mask):
    # given a boolean mask, get the contours
    if not np.any(mask): 
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= 2]


def _contour_to_ann(contours, category_id=0, **meta):
    # given contours get the proper annotation structure
    if not contours:
        return {}
    polys = [
        Polygon(c[:, 0]).simplify(1.0, preserve_topology=False)
        for c in contours
    ]
    multi_poly = MultiPolygon([
        p for px in polys 
        for p in (px.geoms if isinstance(px, MultiPolygon) else [px])
    ])
    x1, y1, x2, y2 = multi_poly.bounds
    return {
        "category_id": category_id+1,
        "segmentation": [
            np.array(poly.exterior.coords).ravel().tolist()
            for poly in multi_poly.geoms
        ],
        "iscrowd": 0,
        "bbox": [x1, y1, x2-x1, y2-y1], 
        # "bbox_mode": BoxMode.XYXY_ABS,
        "bbox_mode": BoxMode.XYWH_ABS,
        "area": multi_poly.area,
        **meta
    }


# Class Counts: datasets/metadata/{name}_cat_count.json


def get_class_counts(dataset_dicts):
    # get class counts
    counts = {'category_id': {}}
    for dd in dataset_dicts:
        for d in dd['annotations']:
            for c in counts:
                xs = d.get(c, -1)
                for x in (xs if isinstance(xs, (list, tuple)) else [xs]):
                    if x != -1:
                        counts[c][x] = counts[c].get(x, 0) + 1
    class_image_count = [{'id': k, 'image_count': c} for k, c in counts['category_id'].items()]
    return class_image_count




# Pre-computed text features: datasets/metadata/egohos.npy


def get_zs_weight(classes):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    return text_encoder(classes).detach().cpu().numpy()



# ln -s /scratch/work/ptg/EGOHOS datasets/egohos

# Register different splits

DATA_ROOT = 'datasets/egohos'

def main(data_root=DATA_ROOT, out_path=DATA_ROOT, name='egohos'):
    # create text-embedding weights file
    np.save("datasets/metadata/egohos.npy", get_zs_weight(object_classes))

    for split in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, split)): 
            continue
        split_name = f'{name}_{split}'

        # create COCO json
        image_root = os.path.join(data_root, split)
        dataset_dicts = create_dataset_dict(image_root)
        dataset = create_dataset(dataset_dicts)
        with open(os.path.join(out_path, f'{split_name}.json'), 'w') as f:
            json.dump(dataset, f)

        # create categories file
        class_image_count = get_class_counts(dataset_dicts)
        with open(f"datasets/metadata/{split_name}_cat_count.json", 'w') as f:
            json.dump(class_image_count, f)

        desc_dataset(dataset)

def desc_dataset(dataset):
    if isinstance(dataset, str):
        dataset = json.load(open(dataset))
    from IPython import embed
    import pandas as pd
    df = pd.DataFrame([
        {k:d[k] for k in ['image_id', 'category_id']}
        for d in dataset['annotations']
    ])
    ids = {d['id'] for d in dataset['images']}
    cats = {c['id']: c['name'] for c in dataset['categories']}
    df['category_id'] = df.category_id.apply(lambda i: cats[i])
    counts = df.groupby('image_id').category_id.value_counts().unstack().fillna(0).value_counts()
    # for k, c in counts.iterrows():
    #     print(k, c)
    print(counts)
    print(ids-set(df.image_id.unique()))
    print(set(df.image_id.unique()) - ids)
    # embed()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
