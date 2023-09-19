# Copyright (c) Facebook, Inc. and its affiliates.
import os
import json
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

D = NAME = 'egohos'
SPLITS = {
    f'{NAME}_{s}': (s, f'{NAME}_{s}.json')
    for s in os.listdir(f'datasets/{D}')
    if os.path.isdir(f'datasets/{D}/{s}')
}

for key, (image_root, json_file) in SPLITS.items():
    register_coco_instances(
        key,
        {
            "class_image_count": json.load(open(f'datasets/metadata/{key}_cat_count.json'))
        },
        os.path.join("datasets", D, json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", D, image_root),
    )
