import os
import time
import numpy as np
import tensorflow as tf

from data_loader import (load_paths_from_dir, make_tf_dataset_from_paths)
from sod_model import (build_sod_model, combined_bce_iou_loss, soft_iou_batch,
                    precision_recall_f1, iou_metric)

DATA_ROOT = "data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 35
PATIENCE = 6
LR = 1e-3
BASE_FILTERS = 32
DEPTH = 3

CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")
BEST_MODEL_DIR = os.path.join("outputs", "best_model.keras")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(BEST_MODEL_DIR), exist_ok=True)

print("[train] Loading file lists...")
tr_imgs, tr_masks, te_imgs, te_masks = load_paths_from_dir(DATA_ROOT)
all_imgs = tr_imgs + te_imgs
all_masks = tr_masks + te_masks
N = len(all_imgs)
idx = np.arange(N)
np.random.seed(42)
np.random.shuffle(idx)
all_imgs = [all_imgs[i] for i in idx]
all_masks = [all_masks[i] for i in idx]

n_train = int(0.70 * N)
n_val = int(0.15 * N)
train_imgs = all_imgs[:n_train]; train_masks = all_masks[:n_train]
val_imgs = all_imgs[n_train:n_train + n_val]; val_masks = all_masks[n_train:n_train + n_val]
test_imgs = all_imgs[n_train + n_val:]; test_masks = all_masks[n_train + n_val:]

train_ds = make_tf_dataset_from_paths(train_imgs, train_masks, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=True, shuffle=True)
val_ds = make_tf_dataset_from_paths(val_imgs, val_masks, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=False, shuffle=False)
test_ds = make_tf_dataset_from_paths(test_imgs, test_masks, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=False, shuffle=False)

print(f"[train] Dataset: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

model = build_sod_model(input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3),
                        base_filters=BASE_FILTERS, depth=DEPTH,
                        batch_norm=True, dropout_rate=0.1)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                        optimizer=optimizer,
                        model=model)
manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

start_epoch = 0
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    start_epoch = int(ckpt.step.numpy())
    print(f"[train] Restored checkpoint: {manager.latest_checkpoint} (resuming at epoch {start_epoch})")
else:
    print("[train] No checkpoint found, training from scratch.")

def compute_metrics_np(y_true, y_pred):
    pred = (y_pred >= 0.5).astype(np.float32)
    inter = (y_true * pred).sum(axis=(1,2,3))
    union = y_true.sum(axis=(1,2,3)) + pred.sum(axis=(1,2,3)) - inter
    iou = ((inter + 1e-7) / (union + 1e-7)).mean()
    tp = inter
    fp = ((1 - y_true) * pred).sum(axis=(1,2,3))
    fn = (y_true * (1 - pred)).sum(axis=(1,2,3))
    precision = ((tp + 1e-7) / (tp + fp + 1e-7)).mean()
    recall = ((tp + 1e-7) / (tp + fn + 1e-7)).mean()
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)
    return iou, precision, recall, f1

@tf.function
def train_step(images, masks):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = combined_bce_iou_loss(masks, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds

@tf.function
def val_step(images, masks):
    preds = model(images, training=False)
    loss = combined_bce_iou_loss(masks, preds)
    return loss, preds

best_val_loss = float("inf")
epochs_without_improve = 0

print("[train] Starting training loop...")
for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    print(f"\n>>> Epoch {epoch+1}/{EPOCHS}")

    train_losses = []
    for step, (imgs, masks) in enumerate(train_ds):
        loss, preds = train_step(imgs, masks)
        train_losses.append(float(loss.numpy()))
        if step % 100 == 0:
            print(f"  [train] step {step} loss={np.mean(train_losses):.4f}")

    val_losses = []
    val_ious = []
    val_precs = []
    val_recs = []
    val_f1s = []

    for step, (v_imgs, v_masks) in enumerate(val_ds):
        v_loss, v_preds = val_step(v_imgs, v_masks)
        val_losses.append(float(v_loss.numpy()))
        try:
            iou, p, r, f1 = compute_metrics_np(v_masks.numpy(), v_preds.numpy())
            val_ious.append(iou); val_precs.append(p); val_recs.append(r); val_f1s.append(f1)
        except Exception:
            pass

    avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
    mean_iou = float(np.mean(val_ious)) if val_ious else float("nan")
    mean_f1 = float(np.mean(val_f1s)) if val_f1s else float("nan")

    epoch_time = time.time() - t0
    print(f"Epoch {epoch+1} time {epoch_time:.1f}s - train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} IoU={mean_iou:.4f} F1={mean_f1:.4f}")

    ckpt.step.assign(epoch + 1)
    save_path = manager.save()
    print(f"[train] checkpoint saved: {save_path}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improve = 0
        model.save(BEST_MODEL_DIR, overwrite=True)
        print(f"[train] New best model saved to: {BEST_MODEL_DIR}")
    else:
        epochs_without_improve += 1
        print(f"[train] no improvement for {epochs_without_improve} epoch(s)")

    if epochs_without_improve >= PATIENCE:
        print("[train] Early stopping triggered.")
        break

print("[train] Training finished.")
print(f"[train] Best validation loss: {best_val_loss:.4f}")
