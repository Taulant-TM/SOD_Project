import os
import cv2
import numpy as np
from data_loader import load_paths_from_dir, preprocess_pair, build_and_save_npz

DATA_ROOT = "data"
OUT_NPZ = os.path.join(DATA_ROOT, "dataset.npz")
SEED = 42
np.random.seed(SEED)


def main():
    tr_imgs, tr_masks, te_imgs, te_masks = load_paths_from_dir(DATA_ROOT)

    all_images = tr_imgs + te_imgs
    all_masks = tr_masks + te_masks

    idx = np.arange(len(all_images))
    np.random.shuffle(idx)
    all_images = [all_images[i] for i in idx]
    all_masks = [all_masks[i] for i in idx]

    N = len(all_images)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val

    train_imgs = all_images[:n_train]
    train_masks = all_masks[:n_train]
    val_imgs = all_images[n_train:n_train + n_val]
    val_masks = all_masks[n_train:n_train + n_val]
    test_imgs = all_images[n_train + n_val:]
    test_masks = all_masks[n_train + n_val:]

    print(f"Total: {N}, train:{len(train_imgs)}, val:{len(val_imgs)}, test:{len(test_imgs)}")

    def build_split(img_list, mask_list):
        images = []
        masks = []
        for ip, mp in zip(img_list, mask_list):
            img = preprocess_pair(cv2.imread(ip), cv2.imread(mp, 0))
            images.append(img[0])
            masks.append(img[1])
        return np.stack(images, axis=0), np.stack(masks, axis=0)

    X_train, y_train = build_split(train_imgs, train_masks)
    X_val, y_val = build_split(val_imgs, val_masks)
    X_test, y_test = build_split(test_imgs, test_masks)

    np.savez_compressed(OUT_NPZ,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    print(f"[prepare_dataset] saved {OUT_NPZ}")
    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()