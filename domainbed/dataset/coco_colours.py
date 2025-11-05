import os, io, time, random
import numpy as np
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from pycocotools.coco import COCO
from skimage.transform import resize
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ====== 类别定义（与你第一段 ColoredCOCO 脚本一致的次序）======
CLASSES = ['boat', 'airplane', 'truck', 'dog','zebra', 'horse', 'bird','train','bus','motorcycle']
NUM_CLASSES = len(CLASSES)

# ====== 输出与批量参数 ======
OUTPUT_DIR = '../data/ColoredCOCO'
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 200
MAX_WORKERS = 16
AREA_MIN = 10000  # 实例面积阈值（与原逻辑一致）
SIZE = (64, 64)

# ====== 颜色策略（与最初 ColoredCOCO 一致）======
env_confounder_strength = [0.8, 0.9, 0.1]  # train1、train2、test

biased_colours = np.array([
    [0,100,0],[188,143,143],[255,0,0],[255,215,0],[0,255,0],
    [65,105,225],[0,225,225],[0,0,255],[255,20,147],[160,160,160]
], dtype=np.int32)

_D = 2500
rng = np.random.default_rng(2025)

def random_different_enough_colour(existing):
    while True:
        x = rng.integers(0, 255, size=3)
        if np.min(np.sum((x - existing)**2, axis=1)) > _D:
            return x

unbiased_colours = np.array([random_different_enough_colour(biased_colours) for _ in range(10)], dtype=np.int32)
test_unbiased_colours = np.array([random_different_enough_colour(np.vstack([biased_colours, unbiased_colours])) for _ in range(10)], dtype=np.int32)

def sample_train_colour(c_idx: int, conf_strength: float) -> np.ndarray:
    """训练环境：以 conf_strength 作为“保持偏置”的概率阈（原逻辑：random()>strength 则用去偏色）"""
    if np.random.random() > conf_strength:
        col = unbiased_colours[rng.integers(0, unbiased_colours.shape[0])]
    else:
        col = biased_colours[c_idx]
    return col

def sample_test_colour() -> np.ndarray:
    """测试环境：使用与训练不同的去偏色（沿用你原脚本“强制去偏”的做法）"""
    return test_unbiased_colours[rng.integers(0, test_unbiased_colours.shape[0])]

# ====== 数量设置：每类样本数 ======
TR1_PER_CLASS = 400
TR2_PER_CLASS = 400
TE_PER_CLASS  = 200

# ====== COCO ======
COCO_ANN = 'coco/annotations/instances_train2017.json'
coco = COCO(COCO_ANN)

def ensure_dir(*parts):
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p

def to_uint8_img(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)

def compose_and_save(img_arr, mask3, bg_color, out_path):
    """resize → 融合（bg * (1-m) + img * m）→ 保存"""
    # 背景颜色块（0.75 强度，保持与你第一段代码一致）
    place_img = 0.75 * np.ones((SIZE[0], SIZE[1], 3), dtype='float32') * (bg_color[None, None, :] / 255.0)

    # skimage.resize 输出 float64 in [0,1]
    resized_mask = resize(mask3, SIZE, preserve_range=True)  # 0..255
    m = np.clip(resized_mask / 255.0, 0.0, 1.0)

    resized_image = resize(img_arr, SIZE)        # 0..1
    resized_place = resize(place_img, SIZE)      # 0..1

    new_im = resized_place * (1 - m) + resized_image * m
    new_im = to_uint8_img(new_im)
    imageio.imwrite(out_path, new_im)

def gather_batch_tasks(images, catIds, need, start_i):
    """从 images 中收集一批满足面积阈值的任务（不做IO），返回 tasks 与新的索引 i"""
    tasks = []
    i = start_i
    while len(tasks) < BATCH_SIZE and (i + 1) < len(images) and (len(tasks) < need):
        i += 1
        im = images[i]
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if not anns:
            continue
        pos = max(range(len(anns)), key=lambda idx: anns[idx]['area'])
        if anns[pos]['area'] < AREA_MIN:
            continue
        mask = coco.annToMask(anns[pos]).astype('uint8')  # 0/1
        mask3 = np.tile((mask * 255)[:, :, None], [1, 1, 3])  # 0/255 -> 3通道
        tasks.append({'url': im['coco_url'], 'mask3': mask3})
    return tasks, i

def download_images(tasks):
    """并发下载 COCO 图片；灰度图转 3 通道；失败记 None"""
    coco_imgs = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_idx = {ex.submit(requests.get, t['url'], timeout=10): idx for idx, t in enumerate(tasks)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                resp = fut.result()
                arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert('RGB'))
                coco_imgs[idx] = arr
            except Exception:
                coco_imgs[idx] = None
    return coco_imgs

def synthesize_env(env_name: str, class_idx: int, class_name: str,
                   target_per_class: int, conf_strength: float,
                   use_test_rule: bool, global_counter: dict):
    """合成单个环境的单个类别"""
    catIds = coco.getCatIds(catNms=[class_name])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    out_dir = ensure_dir(OUTPUT_DIR, env_name, class_name)
    i = -1
    written = 0

    print(f'[{env_name}] Class {class_idx} ({class_name}) : #images = {len(images)}  → target {target_per_class}')
    while written < target_per_class and (i + 1) < len(images):
        need = target_per_class - written
        tasks, i = gather_batch_tasks(images, catIds, need, i)
        if not tasks:
            break
        coco_imgs = download_images(tasks)

        for k in range(len(tasks)):
            img_arr = coco_imgs[k]
            if img_arr is None:
                continue

            # 选择背景颜色
            if use_test_rule:
                bg = sample_test_colour()
            else:
                bg = sample_train_colour(class_idx, conf_strength)

            # 输出文件名：全局递增（跨类可避免覆盖；也可换成本类递增）
            fname = f'{global_counter[env_name]:06d}.png'
            out_path = os.path.join(out_dir, fname)

            try:
                compose_and_save(img_arr, tasks[k]['mask3'], bg, out_path)
                written += 1
                global_counter[env_name] += 1
            except Exception:
                # 某些极端图片/掩码异常，跳过
                pass

        # 进度提示
        if written % 100 == 0:
            print('>', end='', flush=True); time.sleep(0.2)
    print(f'\n  [{env_name}] wrote: {written}/{target_per_class}')

def main():
    # 三个环境的全局计数器（仅用于命名，不影响数据量）
    global_counter = {'env_train1': 0, 'env_train2': 0, 'env_test': 0}

    for c, class_name in enumerate(CLASSES):
        # env_train1：偏置强度 0.8（random()>0.8 → 去偏；否则用该类偏置色）
        synthesize_env('env_train1', c, class_name, TR1_PER_CLASS, env_confounder_strength[0], False, global_counter)

        # env_train2：偏置强度 0.9
        synthesize_env('env_train2', c, class_name, TR2_PER_CLASS, env_confounder_strength[1], False, global_counter)

        # env_test：沿用你原来的“强制去偏新颜色”策略
        synthesize_env('env_test',  c, class_name, TE_PER_CLASS,  env_confounder_strength[2], True,  global_counter)

    print('Done. Output root:', OUTPUT_DIR)

if __name__ == '__main__':
    main()