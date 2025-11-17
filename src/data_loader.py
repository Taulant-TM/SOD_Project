import os
import random 
import numpy as np
import cv2
import tensorflow as tf 
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (128,128)
SEED = 42
random.seed(SEED)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask


def preprocess(image, mask):
    image = cv2.resize(image, IMAGE_SIZE)
    mask = cv2.resize(mask, IMAGE_SIZE)

    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    mask = np.expand_dims(mask, axis = -1)

    return image, mask


def augment(image, mask):
    if random.random() < 0.5:
        image = np.flip(image, axis = 1)
        mask = np.flip(mask, axis=1)

    if random.random() < 0.5:
        factor = 0.2 * (random.random() - 0.5)
        image = np.clip(image + factor, 0, 1)

    return image, mask


def load_paths(base_dir):
    train_img_dir = os.path.join(base_dir, "DUTS-TR/DUTS-TR-Image")
    train_mask_dir = os.path.join(base_dir, "DUTS-TR/DUTS-TR-Mask")

    test_img_dir = os.path.join(base_dir, "DUTS-TE/DUTS-TE-Image")
    test_mask_dir = os.path.join(base_dir, "DUTS-TE/DUTS-TE-Mask")

    train_images = sorted([os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir)])
    train_masks = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir)])

    test_images = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)])
    test_masks = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir)])

    print(f"[INFO] Training samples: {len(train_images)}")
    print(f"[INFO] Testing samples: {len(test_images)}")

    return train_images, train_masks, test_images, test_masks 


def create_dataset(image_paths, mask_paths, augment_data = False):
    images = []
    masks = []

    for img_path,mask_path in zip(image_paths, mask_paths):
        img = load_image(img_path)
        mask = load_mask(mask_path)

        img, mask = preprocess(img, mask)

        if augment_data:
            img, mask = augment(img, mask)

        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)


def prepare_final_datasets(base_dir="data"):
    train_imgs, train_masks, test_imgs, test_masks = load_paths(base_dir)

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        train_imgs,train_masks, test_size=0.15, random_state=SEED
    )

    X_train, y_train = create_dataset(train_img_paths, train_mask_paths, augment_data=True)
    X_val, y_val = create_dataset(val_img_paths, val_mask_paths, augment_data= False)
    X_test, y_test = create_dataset(test_imgs, test_masks, augment_data= False)

    print("[INFO] Final dataset shapes:")
    print("Train: ", X_train.shape, y_train.shape)
    print("Val: ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    prepare_final_datasets()