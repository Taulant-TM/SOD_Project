import os
import random
from typing import List, Tuple
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (128, 128)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path.decode() if isinstance(path, bytes) else path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path.decode() if isinstance(path, bytes) else path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return m.astype(np.float32)


def preprocess_pair(image: np.ndarray, mask: np.ndarray, image_size: Tuple[int, int] = IMAGE_SIZE):
    img = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

    img = img / 255.0
    m = (m >= 128).astype(np.float32)
    m = np.expand_dims(m, axis=-1)

    return img.astype(np.float32), m.astype(np.float32)


def augment_numpy(image: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if random.random() < 0.5:
        factor = 0.15 * (random.random() * 2 - 1)  
        image = np.clip(image + factor, 0.0, 1.0)

    if random.random() < 0.3:
        h, w = image.shape[:2]
        crop_scale = random.uniform(0.85, 1.0)
        ch = int(h * crop_scale)
        cw = int(w * crop_scale)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        image = image[y0:y0+ch, x0:x0+cw]
        mask = mask[y0:y0+ch, x0:x0+cw]
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    return image, mask


def load_paths_from_dir(base_dir: str):
    tr_img_dir = os.path.join(base_dir, "DUTS-TR", "DUTS-TR-Image")
    tr_mask_dir = os.path.join(base_dir, "DUTS-TR", "DUTS-TR-Mask")
    te_img_dir = os.path.join(base_dir, "DUTS-TE", "DUTS-TE-Image")
    te_mask_dir = os.path.join(base_dir, "DUTS-TE", "DUTS-TE-Mask")

    def list_sorted(d):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")
        return sorted([os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")])

    tr_imgs = list_sorted(tr_img_dir)
    tr_masks = list_sorted(tr_mask_dir)
    te_imgs = list_sorted(te_img_dir)
    te_masks = list_sorted(te_mask_dir)

    if len(tr_imgs) != len(tr_masks):
        print("[WARNING] Number of TR images and masks differ. Proceeding with zip (shortest).")
    if len(te_imgs) != len(te_masks):
        print("[WARNING] Number of TE images and masks differ. Proceeding with zip (shortest).")

    return tr_imgs, tr_masks, te_imgs, te_masks


def build_and_save_npz(all_image_paths: List[str], all_mask_paths: List[str], out_path: str = "data/dataset.npz",
                    img_size: Tuple[int, int] = IMAGE_SIZE, verbose: bool = True):
    images = []
    masks = []
    L = len(all_image_paths)
    for i, (ip, mp) in enumerate(zip(all_image_paths, all_mask_paths)):
        img = load_image(ip)
        m = load_mask(mp)
        img, m = preprocess_pair(img, m, image_size=img_size)
        images.append(img)
        masks.append(m)
        if verbose and (i % 500 == 0 or i == L-1):
            print(f"[build_npz] {i+1}/{L}")

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, X=images, Y=masks)
    if verbose:
        print(f"[build_npz] saved {out_path} -> X:{images.shape} Y:{masks.shape}")


def make_tf_dataset_from_paths(image_paths: List[str], mask_paths: List[str],
                            img_size: Tuple[int, int] = IMAGE_SIZE, batch_size: int = 8,
                            augment: bool = False, shuffle: bool = False):
    def _load_pair(img_p, mask_p):
        img = tf.numpy_function(lambda p: load_image(p), [img_p], tf.float32)
        mask = tf.numpy_function(lambda p: load_mask(p), [mask_p], tf.float32)
        img.set_shape([None, None, 3])
        mask.set_shape([None, None])
        img, mask = tf.numpy_function(lambda a, b: preprocess_pair(a, b, img_size),
                                    [img, mask], [tf.float32, tf.float32])
        img.set_shape([img_size[1], img_size[0], 3])
        mask.set_shape([img_size[1], img_size[0], 1])
        return img, mask

    img_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    mask_ds = tf.data.Dataset.from_tensor_slices(mask_paths)
    ds = tf.data.Dataset.zip((img_ds, mask_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=SEED)
    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def _tf_augment(img, mask):
            img = tf.image.random_flip_left_right(img)
            mask = tf.image.random_flip_left_right(mask)
            img = tf.image.random_brightness(img, max_delta=0.12)
            return img, mask
        ds = ds.map(_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    base = "data"
    tr_i, tr_m, te_i, te_m = load_paths_from_dir(base)
    print("TR:", len(tr_i), "TE:", len(te_i))
