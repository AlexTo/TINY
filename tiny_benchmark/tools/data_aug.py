import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
import pylab
import math
import os

data_dir = "../dataset/tiny_set/erase_with_uncertain_dataset"
train_file = f'{data_dir}/annotations/corner/task/tiny_set_train_sw640_sh512_all.json'
train_file_aug = f'{data_dir}/annotations/corner/task/tiny_set_train_sw640_sh512_all_augmented.json'
train_tiny3_and_others_file = f'{data_dir}/annotations/corner/task/tiny_set_train_sw640_sh512_tiny3_and_others.json'

np.random.seed(12345)


def tinify(img, original_size):
    scale = original_size / 7.45
    resized_img = img.resize((math.ceil(img.width / scale), math.ceil(img.height / scale)), Image.ANTIALIAS)
    return resized_img


def gen_ann_file():
    with open(train_file) as f:
        train = json.load(f)

    annotations = train['annotations']
    images = train['images']

    for a in annotations:
        if a['size'] <= 12:
            a['ignore'] = True

    removed_images = []
    remaining_images = []

    for img in images:
        found_bbox = False
        for a in annotations:
            if a['image_id'] == img['id'] and not a['ignore']:
                found_bbox = True
                break
        if not found_bbox:
            removed_images.append(img)
        else:
            remaining_images.append(img)

    train['images'] = remaining_images

    with open(train_tiny3_and_others_file, 'w') as f:
        json.dump(train, f)


def new_ann(x, y, w, h, img_id):
    return {
        'segmentation': [[x, y, x, y + h, x + w, y + h, x + w, y]],
        'bbox': [x, y, w, h],
        'area': 0,
        'iscrowd': 0,
        'category_id': 1,
        'image_id': img_id,
        'ignore': False,
        'uncertain': False,
        'logo': False,
        'in_dense_image': False,
        'size': np.sqrt(w * h),
    }


# Find the area to place "person" based on the current corner values and existing annotations of the image
def find_area(corner, tiny_persons, new_person):
    x = np.random.randint(0, corner[2] - corner[0] - new_person.width)
    y = np.random.randint(0, corner[3] - corner[1] - new_person.height)
    return x, y


def place_random(orig_img, file_name, corner, tiny_persons, existing_anns, max_objs=20):
    if os.path.exists(f"{data_dir}/train_augmented/{file_name}"):
        orig_img = Image.open(f"{data_dir}/train_augmented/{file_name}").convert('RGB')

    num_to_put = max(0, max_objs - len(existing_anns))

    if num_to_put == 0:
        return
    img_id = existing_anns[0]['image_id']
    anns = []

    patches = np.random.choice(list(range(len(tiny_persons))), num_to_put, replace=True)
    for p in patches:
        person = tiny_persons[p]
        x, y = find_area(corner, tiny_persons, person)
        ann = new_ann(x, y, person.width, person.height, img_id)
        anns.append(ann)
        orig_img.paste(person, (x + corner[0], y + corner[1]))

    orig_img.save(f"{data_dir}/train_augmented/{file_name}")

    return anns


def augment():
    with open(train_tiny3_and_others_file) as f:
        train_tiny3_and_others = json.load(f)
    with open(train_file) as f:
        train = json.load(f)

    new_anns = []
    images = train_tiny3_and_others['images']
    coco = COCO(train_tiny3_and_others_file)
    for i in images:
        ann_ids = coco.getAnnIds(imgIds=i['id'])

        original_img = Image.open(f"{data_dir}/train/{i['file_name']}").convert('RGB')
        cropped_img = original_img.crop(i['corner'])
        anns = coco.loadAnns(ann_ids)

        tinified_persons = []

        for a in anns:
            if a['ignore']:
                continue
            bbox = a['bbox']
            p = cropped_img.crop([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            tiny_p = tinify(p, a['size'])
            tinified_persons.append(tiny_p)

        if len(tinified_persons) > 0:
            results = place_random(original_img, i['file_name'], i['corner'], tinified_persons, anns)
            if results is not None:
                new_anns.extend(results)

    annotations = train['annotations']
    idx = len(annotations)
    for a in new_anns:
        a['id'] = idx
        idx = idx + 1

    annotations.extend(new_anns)
    with open(train_file_aug, 'w') as f:
        json.dump(train, f)


if __name__ == "__main__":
    augment()
