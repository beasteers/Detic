# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import glob
import numpy as np
import cv2

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta

from shapely.geometry import Polygon, MultiPolygon

logger = logging.getLogger(__name__)

__all__ = ["custom_load_egohos_json", "custom_register_egohos_instances"]

# vocab

object_classes = [
    'left hand',
    'right hand',
    '1st order interacting object by left hand',
    '1st order interacting object by right hand',
    '1st order interacting object by both hands',
    '2nd order interacting object by left hand',
    '2nd order interacting object by right hand',
    '2nd order interacting object by both hands',
]
L, R, O1L, O1R, O1B, O2L, O2R, O2B = range(len(object_classes))

property_classes = [
    'interacting',
]

relation_classes = [
    'interacting',
]

# 

def custom_register_egohos_instances(name, metadata, image_root):
    """
    """
    DatasetCatalog.register(name, lambda: custom_load_egohos_json(
        image_root, name))
    MetadataCatalog.get(name).set(
        image_root=image_root, 
        evaluator_type="lvis", 
        **metadata
    )


def custom_load_egohos_json(image_root, dataset_name=None):
    '''
    Modifications:
      use `file_name`
      convert neg_category_ids
      add pos_category_ids
    '''
    # get file list
    img_fs = sorted(glob.glob(os.path.join(image_root, 'image/*.jpg')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in img_fs]
    mask_fs = [
        os.path.join(image_root, 'label', f'{n}.png') 
        for n in names]

    # load annotations for each file
    dataset_dicts = []
    for i, (fname_im, fname_mask) in enumerate(zip(img_fs, mask_fs)):
        mask = cv2.imread(fname_mask)
        dataset_dicts.append({
            "file_name": fname_im,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "image_id": i,
            # "pos_category_ids": [],
            # "neg_category_ids": [],
            # "not_exhaustive_category_ids": [],
            # "captions": [],
            # "caption_features": [],
            "annotations": get_annotations(mask),
        })
    return dataset_dicts

def get_annotations(mask):
    # convert masks to contours
    masks = [mask == i+1 for i in range(len(object_classes))]
    contours = [_bool_mask_to_contour(m) for m in masks]
    (
        left, right, 
        obj1_left, obj1_right, obj1_both,
        obj2_left, obj2_right, obj2_both,
    ) = contours
    
    # convert to hand, object, and relation annotations
    annotations = [
        _obj_interaction_ann(left, obj1_left, L, O1L),
        _obj_interaction_ann(left, obj2_left, L, O2L),
        _obj_interaction_ann(right, obj1_right, R, O1R),
        _obj_interaction_ann(right, obj2_right, R, O2R),
        _obj_interaction_ann(left, obj1_both, L, O1B),
        _obj_interaction_ann(left, obj2_both, L, O2B),
        _obj_interaction_ann(right, obj1_both, R, O1B),
        _obj_interaction_ann(right, obj2_both, R, O2B),
    ]

    # filter empty annotations
    return [d for d in annotations if d.get('segmentation')]

def _obj_interaction_ann(hand, obj, hand_id, obj_id):
    # given a hand object pair of contours, construct the annotations for separate+combo
    is_interacting = hand and obj
    property_id = 0 if is_interacting else -1
    return [
        _contour_to_ann(hand, category_id=hand_id, property_id=property_id),
        _contour_to_ann(obj, category_id=obj_id, property_id=property_id),
        _contour_to_ann(hand + obj, relation_id=0) if is_interacting else {},
    ]



def _bool_mask_to_contour(mask):
    # given a boolean mask, get the contours
    if not np.any(mask): 
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= 2]

def _contour_to_ann(contours, **meta):
    # given contours get the proper annotation structure
    if not contours:
        return {}
    polys = [
        Polygon(c).simplify(1.0, preserve_topology=False)
        for c in contours
    ]
    multi_poly = MultiPolygon(polys)
    return {
        "segmentation": [
            np.array(poly.exterior.coords).ravel().tolist()
            for poly in polys
        ],
        "bbox": multi_poly.bounds, 
        "bbox_mode": BoxMode.XYXY_ABS,
        "area": multi_poly.area,
        **meta
    }



# Register different splits


_CUSTOM_SPLITS_EGOHOS = {
    "egohos_train": "train",
    "egohos_val": "val",
    "egohos_test_indomain": "test_indomain",
    "egohos_test_outdomain": "test_outdomain",
}

metadata = {
    "thing_classes": object_classes,
    "property_classes": property_classes,
    "relation_classes": relation_classes,
}
for key, image_root in _CUSTOM_SPLITS_EGOHOS.items():
    custom_register_egohos_instances(
        key,
        metadata,
        os.path.join("/EGOHOS", image_root),
    )