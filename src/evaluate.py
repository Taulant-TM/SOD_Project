import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_paths_from_dir, make_tf_dataset_from_paths
from sod_model import iou_metric, precision_recall_f1

PROJECT_ROOT = "."
DATA_ROOT = "data"
BEST_MODEL_DIR = os.path.join("outputs", "best_model.keras")
OUT_DIR = os.path.join("outputs", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 8

def load_model():
    if os.path.exists(BEST_MODEL_DIR) and (os.path.isdir(BEST_MODEL_DIR) or BEST_MODEL_DIR.endswith(".keras")):
        print("[evaluate] Loading best model from disk...")
        return tf.keras.models.load_model(BEST_MODEL_DIR, compile=False)
    else:
        raise FileNotFoundError("No saved best model found. Train first and ensure outputs/best_model.keras exists.")

def visualize_save(img, gt, pred, save_path):
    img_vis = (img * 255).astype(np.uint8) if img.max() <= 1.1 else img.astype(np.uint8)
    pred_arr = pred.squeeze()
    gt_arr = gt.squeeze()
    overlay = img_vis.astype(np.float32) / 255.0
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_arr * 0.6, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img_vis); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(gt_arr, cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
    axes[2].imshow(pred_arr, cmap="gray"); axes[2].set_title("Pred"); axes[2].axis("off")
    axes[3].imshow(overlay); axes[3].set_title("Overlay"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    tr_i, tr_m, te_i, te_m = load_paths_from_dir(DATA_ROOT)
    all_imgs = tr_i + te_i
    all_masks = tr_m + te_m
    N = len(all_imgs)
    np.random.seed(42)
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    test_imgs = [all_imgs[i] for i in idx[n_train + n_val:]]
    test_masks = [all_masks[i] for i in idx[n_train + n_val:]]

    test_ds = make_tf_dataset_from_paths(test_imgs, test_masks, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    model = load_model()

    results = []
    total_time = 0.0
    total_images = 0
    idx_out = 0

    print("[evaluate] Running inference on test set...")
    for batch_images, batch_masks in test_ds:
        t0 = time.time()
        preds = model.predict(batch_images)
        t1 = time.time()
        total_time += (t1 - t0)
        total_images += batch_images.shape[0]

        batch_iou = iou_metric(batch_masks, preds).numpy()
        p, r, f1 = precision_recall_f1(batch_masks, preds, threshold=0.5)
        p, r, f1 = float(p.numpy()), float(r.numpy()), float(f1.numpy())
        mae = float(np.mean(np.abs(batch_masks.numpy() - preds)))

        for i in range(batch_images.shape[0]):
            img_np = batch_images[i].numpy()
            gt_np = batch_masks[i].numpy()
            pred_np = preds[i]
            fname = f"pred_{idx_out:05d}.png"
            save_path = os.path.join(OUT_DIR, fname)
            visualize_save(img_np, gt_np, pred_np, save_path)

            bin_pred = (pred_np >= 0.5).astype(np.float32)
            inter = float((gt_np * bin_pred).sum())
            union = float(gt_np.sum()) + float(bin_pred.sum()) - inter
            sample_iou = (inter + 1e-7) / (union + 1e-7)
            tp = float((gt_np * bin_pred).sum())
            fp = float(((1 - gt_np) * bin_pred).sum())
            fn = float((gt_np * (1 - bin_pred)).sum())
            sample_prec = (tp + 1e-7) / (tp + fp + 1e-7)
            sample_rec = (tp + 1e-7) / (tp + fn + 1e-7)
            sample_f1 = 2 * sample_prec * sample_rec / (sample_prec + sample_rec + 1e-7)
            sample_mae = float(np.mean(np.abs(gt_np - pred_np)))

            results.append({"image": fname,
                            "iou": float(sample_iou),
                            "precision": float(sample_prec),
                            "recall": float(sample_rec),
                            "f1": float(sample_f1),
                            "mae": float(sample_mae)})
            idx_out += 1

    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    keys = ["image", "iou", "precision", "recall", "f1", "mae"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    avg_iou = np.mean([r["iou"] for r in results]) if results else 0.0
    avg_prec = np.mean([r["precision"] for r in results]) if results else 0.0
    avg_rec = np.mean([r["recall"] for r in results]) if results else 0.0
    avg_f1 = np.mean([r["f1"] for r in results]) if results else 0.0
    avg_mae = np.mean([r["mae"] for r in results]) if results else 0.0
    avg_time_ms = (total_time / total_images) * 1000.0 if total_images > 0 else 0.0

    print("=== Test set results ===")
    print(f"Images: {total_images}")
    print(f"Mean IoU:  {avg_iou:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print(f"F1:        {avg_f1:.4f}")
    print(f"MAE:       {avg_mae:.4f}")
    print(f"Avg inference time per image: {avg_time_ms:.1f} ms")
    print(f"Saved visualizations + CSV to: {OUT_DIR}")


if __name__ == "__main__":
    main()
