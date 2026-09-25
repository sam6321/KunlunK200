"""ResNet50 image classification on Kunlun XPU via Paddle Inference."""
import os
import sys
import time

import numpy as np
from PIL import Image
from paddle.inference import Config, create_predictor

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 224.0 / min(w, h)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - MEAN) / STD
    return arr[np.newaxis, :]


def main():
    image = sys.argv[1]
    model = sys.argv[2]
    params = sys.argv[3]
    labels_path = sys.argv[4] if len(sys.argv) > 4 else ""

    use_cpu = os.environ.get("PADDLE_CPU") == "1"
    config = Config(model, params)
    config.switch_ir_optim(True)
    if not use_cpu:
        config.enable_xpu()
    print("device", "cpu" if use_cpu else "xpu")
    predictor = create_predictor(config)

    blob = preprocess(image)
    name = predictor.get_input_names()[0]
    handle = predictor.get_input_handle(name)
    handle.reshape(blob.shape)
    handle.copy_from_cpu(blob)

    predictor.run()  # warmup
    t0 = time.perf_counter()
    loops = 10
    for _ in range(loops):
        predictor.run()
    ms = (time.perf_counter() - t0) * 1000.0 / loops

    out_name = predictor.get_output_names()[0]
    raw = predictor.get_output_handle(out_name).copy_to_cpu()
    print(
        f"output {out_name} {raw.shape} min {float(raw.min()):.6g} "
        f"max {float(raw.max()):.6g} sum {float(raw.sum()):.6g}"
    )
    scores = raw.reshape(-1)
    order = np.argsort(scores)[::-1][:5]

    labels = []
    if labels_path:
        with open(labels_path, encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    print(f"input {blob.shape} {blob.dtype}")
    print(f"latency {ms:.2f} ms/iter ({loops} loops, device warmup excluded)")
    for rank, idx in enumerate(order, 1):
        name = labels[idx] if idx < len(labels) else ""
        print(f"  {rank}. class {int(idx):4d}  {scores[idx]:8.4f}  {name}")


if __name__ == "__main__":
    main()
