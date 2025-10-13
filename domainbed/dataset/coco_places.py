'''
The datasets generated process of coco_palces and colored_coco follows the paper " [Systematic generalization with group invariant predictions](https://github.com/Faruk-Ahmed/predictive_group_invariance?tab=readme-ov-file)".

For "coco" you can create the datasets using the coco_places.py and colored_coco.py, which will require installing the [cocoapi](https://github.com/cocodataset/cocoapi) and download the [Places](http://places2.csail.mit.edu/) dataset.
'''

import os, sys, time, io, requests
import numpy as np
import random
from PIL import Image
from pycocotools.coco import COCO
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import imageio
from PIL import ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ['dog','zebra', 'horse', 'bird', 'cow', 'boat', 'airplane', 'truck', 'train', 'bus']
Animal = ['dog','zebra', 'horse', 'bird', 'cow']
Vehicle = ['boat', 'airplane', 'truck', 'train', 'bus']
NUM_CLASSES = len(CLASSES)

BATCH_SIZE=200

output_dir = '../data/COCOPlaces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

place_data_dir = '../data/COCOPlaces'
places_dir = os.path.join(place_data_dir, 'data_256')

biased_places = ['d/desert/sand', 'f/forest/broadleaf']

env_confounder_strength = [0.8, 0.9, 0.1]

biased_place_fnames = {}
for i, target_place in enumerate(biased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]    
    random.shuffle(L)
    biased_place_fnames[i] = L
    
    
tr1_i = 400*NUM_CLASSES
tr2_i = 400*NUM_CLASSES
te_i = 200*NUM_CLASSES

coco = COCO('coco/annotations/instances_train2017.json')


tr1_s, tr2_s, te_s = 0, 0, 0
for c, class_name in enumerate(CLASSES):
    if c == 5:
        tr1_s = 0
    catIds = coco.getCatIds(catNms=[CLASSES[c]])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    print('Class {} (train/test) : #images = {}'.format(c, len(images)))

    # ============ env_train1 ============
    env_name = 'env_train1'
    label = 'Animal' if class_name in Animal else 'Vehicle'
    _path = os.path.join(output_dir, env_name, label)
    if not os.path.exists(_path):
        os.makedirs(_path)

    tr1_target = tr1_i // NUM_CLASSES
    tr1_written = 0
    i = -1
    while tr1_written < tr1_target and i + 1 < len(images):
        # 收集一批任务：挑出满足面积阈值的样本（不做IO）
        tasks = []
        while len(tasks) < BATCH_SIZE and i + 1 < len(images) and (tr1_written + len(tasks)) < tr1_target:
            i += 1
            im = images[i]
            annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(anns) == 0:
                continue
            pos = max(range(len(anns)), key=lambda idx: anns[idx]['area'])
            if anns[pos]['area'] < 10000:
                continue
            mask = coco.annToMask(anns[pos]).astype('uint8')  # 0/1
            mask3 = np.tile((mask * 255)[:, :, None], [1, 1, 3])  # 0/255 三通道
            tasks.append({
                'coco_url': im['coco_url'],
                'mask3': mask3
            })

        # 并发下载 COCO 图片
        coco_imgs = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=16) as ex:
            future_to_idx = {ex.submit(requests.get, t['coco_url'], timeout=10): idx for idx, t in enumerate(tasks)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    resp = fut.result()
                    arr = np.asarray(Image.open(io.BytesIO(resp.content)))
                    if arr.ndim == 2:
                        arr = np.tile(arr[:, :, None], [1, 1, 3])
                    coco_imgs[idx] = arr
                except Exception:
                    coco_imgs[idx] = None

        # 逐个样本：决定 places 路径 → 读本地图 → resize & 合成 → 保存
        for k in range(len(tasks)):
            if coco_imgs[k] is None:
                continue
            # 随机选择偏置目录
            if class_name in Animal:
                place1 = biased_place_fnames[1]
                place2 = biased_place_fnames[0]
            else:
                place1 = biased_place_fnames[0]
                place2 = biased_place_fnames[1]
            # 与原逻辑一致：np.random.random() > conf_strength 决策
            use_place1 = (np.random.random() > env_confounder_strength[0])
            # 使用当前已写入数作为索引（避免下载失败导致错位）
            place_idx = tr1_written
            place_rel = place1[place_idx] if use_place1 else place2[place_idx]
            place_abs = os.path.join(places_dir, place_rel)
            try:
                place_img = np.asarray(Image.open(place_abs).convert('RGB'))
            except Exception:
                continue

            # resize（skimage.resize 输出 float64 in [0,1]）
            resized_mask = resize(tasks[k]['mask3'], (64, 64), preserve_range=True)  # 保留0~255
            resized_image = resize(coco_imgs[k], (64, 64))
            resized_place = resize(place_img, (64, 64))

            m = np.clip(resized_mask / 255.0, 0.0, 1.0)
            new_im = resized_place * (1 - m) + resized_image * m
            new_im = (np.clip(new_im, 0, 1) * 255).astype(np.uint8)

            image_path = os.path.join(_path, '{}.png'.format(tr1_s))
            try:
                imageio.imwrite(image_path, new_im)
                tr1_s += 1
                tr1_written += 1
            except Exception:
                pass

    print('  env_train1 wrote:', tr1_written)

    # ============ env_train2 ============
    if c == 5:
        tr2_s = 0
    env_name = 'env_train2'
    label = 'Animal' if class_name in Animal else 'Vehicle'
    _path = os.path.join(output_dir, env_name, label)
    if not os.path.exists(_path):
        os.makedirs(_path)

    tr2_target = tr2_i // NUM_CLASSES
    tr2_written = 0
    # train2 的 places 索引偏移：int(conf0 * tr1_si)
    place_offset_tr2 = int(env_confounder_strength[0] * tr1_written)
    # 从上面消耗到的 i 继续往后取
    while tr2_written < tr2_target and i + 1 < len(images):
        tasks = []
        while len(tasks) < BATCH_SIZE and i + 1 < len(images) and (tr2_written + len(tasks)) < tr2_target:
            i += 1
            im = images[i]
            annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(anns) == 0:
                continue
            pos = max(range(len(anns)), key=lambda idx: anns[idx]['area'])
            if anns[pos]['area'] < 10000:
                continue
            mask = coco.annToMask(anns[pos]).astype('uint8')
            mask3 = np.tile((mask * 255)[:, :, None], [1, 1, 3])
            tasks.append({
                'coco_url': im['coco_url'],
                'mask3': mask3
            })

        coco_imgs = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=16) as ex:
            future_to_idx = {ex.submit(requests.get, t['coco_url'], timeout=10): idx for idx, t in enumerate(tasks)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    resp = fut.result()
                    arr = np.asarray(Image.open(io.BytesIO(resp.content)))
                    if arr.ndim == 2:
                        arr = np.tile(arr[:, :, None], [1, 1, 3])
                    coco_imgs[idx] = arr
                except Exception:
                    coco_imgs[idx] = None

        for k in range(len(tasks)):
            if coco_imgs[k] is None:
                continue
            if class_name in Animal:
                place1 = biased_place_fnames[1]
                place2 = biased_place_fnames[0]
            else:
                place1 = biased_place_fnames[0]
                place2 = biased_place_fnames[1]
            use_place1 = (np.random.random() > env_confounder_strength[1])
            place_idx = place_offset_tr2 + tr2_written
            place_rel = place1[place_idx] if use_place1 else place2[place_idx]
            place_abs = os.path.join(places_dir, place_rel)
            try:
                place_img = np.asarray(Image.open(place_abs).convert('RGB'))
            except Exception:
                continue

            resized_mask = resize(tasks[k]['mask3'], (64, 64), preserve_range=True)
            resized_image = resize(coco_imgs[k], (64, 64))
            resized_place = resize(place_img, (64, 64))

            m = np.clip(resized_mask / 255.0, 0.0, 1.0)
            new_im = resized_place * (1 - m) + resized_image * m
            new_im = (np.clip(new_im, 0, 1) * 255).astype(np.uint8)

            image_path = os.path.join(_path, '{}.png'.format(tr2_s))
            try:
                imageio.imwrite(image_path, new_im)
                tr2_s += 1
                tr2_written += 1
            except Exception:
                pass

    print('  env_train2 wrote:', tr2_written)

    # ============ env_test ============
    if c == 5:
        te_s = 0
    env_name = 'env_test'
    label = 'Animal' if class_name in Animal else 'Vehicle'
    _path = os.path.join(output_dir, env_name, label)
    if not os.path.exists(_path):
        os.makedirs(_path)

    te_target = te_i // NUM_CLASSES
    te_written = 0
    # test 的 places 偏移：int(conf0*tr1_si) + int(conf1*tr2_si)
    place_offset_te = int(env_confounder_strength[0] * tr1_written) + int(env_confounder_strength[1] * tr2_written)
    while te_written < te_target and i + 1 < len(images):
        tasks = []
        while len(tasks) < BATCH_SIZE and i + 1 < len(images) and (te_written + len(tasks)) < te_target:
            i += 1
            im = images[i]
            annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(anns) == 0:
                continue
            pos = max(range(len(anns)), key=lambda idx: anns[idx]['area'])
            if anns[pos]['area'] < 10000:
                continue
            mask = coco.annToMask(anns[pos]).astype('uint8')
            mask3 = np.tile((mask * 255)[:, :, None], [1, 1, 3])
            tasks.append({
                'coco_url': im['coco_url'],
                'mask3': mask3
            })

        coco_imgs = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=16) as ex:
            future_to_idx = {ex.submit(requests.get, t['coco_url'], timeout=10): idx for idx, t in enumerate(tasks)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    resp = fut.result()
                    arr = np.asarray(Image.open(io.BytesIO(resp.content)))
                    if arr.ndim == 2:
                        arr = np.tile(arr[:, :, None], [1, 1, 3])
                    coco_imgs[idx] = arr
                except Exception:
                    coco_imgs[idx] = None

        for k in range(len(tasks)):
            if coco_imgs[k] is None:
                continue
            if class_name in Animal:
                place1 = biased_place_fnames[1]
                place2 = biased_place_fnames[0]
            else:
                place1 = biased_place_fnames[0]
                place2 = biased_place_fnames[1]
            use_place1 = (np.random.random() > env_confounder_strength[2])
            place_idx = place_offset_te + te_written
            place_rel = place1[place_idx] if use_place1 else place2[place_idx]
            place_abs = os.path.join(places_dir, place_rel)
            try:
                place_img = np.asarray(Image.open(place_abs).convert('RGB'))
            except Exception:
                continue

            resized_mask = resize(tasks[k]['mask3'], (64, 64), preserve_range=True)
            resized_image = resize(coco_imgs[k], (64, 64))
            resized_place = resize(place_img, (64, 64))

            m = np.clip(resized_mask / 255.0, 0.0, 1.0)
            new_im = resized_place * (1 - m) + resized_image * m
            new_im = (np.clip(new_im, 0, 1) * 255).astype(np.uint8)

            image_path = os.path.join(_path, '{}.png'.format(te_s))
            try:
                imageio.imwrite(image_path, new_im)
                te_s += 1
                te_written += 1
            except Exception:
                pass

    print('  env_test wrote:', te_written)