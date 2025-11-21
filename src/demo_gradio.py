import os
import time
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr

from data_loader import preprocess_pair

BEST_MODEL_DIR = os.path.join("outputs", "best_model.keras")
IMG_SIZE = (128, 128)

if not os.path.exists(BEST_MODEL_DIR):
    raise FileNotFoundError("Best model not found. Run training first.")

model = tf.keras.models.load_model(BEST_MODEL_DIR, compile=False)

def run_inference(img):
    if img is None:
        return None, None, "No image"
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    inp_batch = np.expand_dims(inp, 0)

    t0 = time.time()
    pred = model.predict(inp_batch)[0, ..., 0]
    t1 = time.time()
    inf_ms = (t1 - t0) * 1000.0

    pred_resized = cv2.resize(pred, (orig_w, orig_h))
    mask_vis = (pred_resized * 255).astype(np.uint8)
    mask_vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

    overlay = (img_rgb.astype(np.float32) / 255.0).copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_resized * 0.6, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return mask_vis_rgb, overlay, f"{inf_ms:.1f} ms"

iface = gr.Interface(fn=run_inference,
                    inputs=gr.Image(type="numpy"),
                    outputs=[gr.Image(type="numpy", label="Predicted mask"),
                            gr.Image(type="numpy", label="Overlay"),
                            gr.Textbox(label="Inference time")],
                    title="SOD Demo",
                    description="Upload an image and see predicted saliency mask + overlay")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", share=False)