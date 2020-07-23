# assuming current working directory is root/tiny_benchmark

import json
import os
import pickle
import random

import PIL
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

tiny_img_dir = '../dataset/tiny_set/erase_with_uncertain_dataset/train'

tiny_sea_person_file = \
    '../dataset/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_sea.json'

tiny_earth_person_file = \
    '../dataset/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_earth.json'

target_dir = '../dataset/unsplash'


def get_size_band(ann):
    size = ann['size']
    if size <= 8:
        return 1
    if size <= 12:
        return 2
    if size <= 20:
        return 3
    return 4


def collect_patches(ann_file, img_dir):
    with open(ann_file) as f:
        data = json.load(f)
    coco = COCO(ann_file)
    imgs = data['images']
    patches = {}
    for img in imgs:
        original_img = Image.open(f"{img_dir}/{img['file_name']}").convert('RGB')
        cropped_img = original_img.crop(img['corner'])

        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)

        for a in anns:
            if a['ignore']:
                continue
            bbox = a['bbox']
            p = cropped_img.crop([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            size_band = get_size_band(a)
            if f"tiny{size_band}" not in patches:
                patches[f"tiny{size_band}"] = []
            patches[f"tiny{size_band}"].append(p)
    return patches


def new_ann(x, y, w, h, img_id, ann_id):
    return {
        'id': ann_id,
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


def find_empty_spot(original_img, sample, cat):
    if cat == 'earth':
        # just a heuristic to paste person on the ground with the assumption that the ground is usually
        # at the bottom half of the picture
        x = np.random.randint(0, original_img.width - sample.width)
        y = np.random.randint(original_img.height / 2, original_img.height - sample.height)
    else:
        x = np.random.randint(0, original_img.width - sample.width)
        y = np.random.randint(0, original_img.height - sample.height)
    return x, y


def patch_dir(src_img_dir, target_img_dir, persons, no_per_size_band=20, cat='earth', img_start_id=0, ann_start_id=0):
    img_id = img_start_id
    ann_id = ann_start_id
    anns = []
    images = []
    for f in os.listdir(src_img_dir):

        all_samples = []
        for band in persons:
            samples = np.random.choice(list(range(len(persons[band]))), no_per_size_band, replace=True)
            all_samples = all_samples + [persons[band][i] for i in samples]

        original_img = Image.open(f'{src_img_dir}/{f}').convert('RGB')
        for sample in all_samples:
            # if random.uniform(0, 1) > 0.5:
            #    sample = sample.rotate(random.uniform(0, 45), PIL.Image.NEAREST)
            if random.uniform(0, 1) > 0.5:
                sample = sample.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            x, y = find_empty_spot(original_img, sample, cat)
            ann = new_ann(x, y, sample.width, sample.height, img_id, ann_id)
            anns.append(ann)
            original_img.paste(sample, (x, y))
            ann_id = ann_id + 1
        original_img.save(f"{target_img_dir}/{f}")
        images.append({
            "file_name": f,
            "height": original_img.height,
            "width": original_img.width,
            "id": img_id
        })
        img_id = img_id + 1
    return anns, images


def main():
    if not os.path.exists(f'{target_dir}/persons.pickle'):
        sea_persons = collect_patches(tiny_sea_person_file, tiny_img_dir)
        earth_persons = collect_patches(tiny_earth_person_file, tiny_img_dir)
        persons = {
            'sea_persons': sea_persons,
            'earth_persons': earth_persons
        }
        with open(f'{target_dir}/persons.pickle', 'wb') as f:
            pickle.dump(persons, f)
    else:
        with open(f'{target_dir}/persons.pickle', 'rb') as f:
            persons = pickle.load(f)

    earth_anns, earth_images = patch_dir(f'{target_dir}/orig_images/earth', f'{target_dir}/images',
                                         persons['earth_persons'],
                                         cat="earth")

    sea_anns, sea_images = patch_dir(f'{target_dir}/orig_images/sea', f'{target_dir}/images', persons['sea_persons'],
                                     cat="sea", ann_start_id=len(earth_anns), img_start_id=len(earth_images))

    result = {
        "type": "instance",
        "annotations": earth_anns + sea_anns,
        "images": earth_images + sea_images,
        "categories": [{
            "id": 1,
            "name": "person"}]
    }
    with open(f"{target_dir}/annotations/unsplash_train.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
