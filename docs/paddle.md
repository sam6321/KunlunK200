# Paddle

Paddle is the stack that runs a model on this card. Use the **`paddlepaddle_xpu` 2.6.1** wheel for **CPython 3.10**. The CPU-only `paddlepaddle` wheels and the 2.3 XPU wheels are a different generation.

```bash
python3 -m venv ~/paddle-xpu
~/paddle-xpu/bin/pip install 'numpy<2' paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl
```

NumPy 2 builds a flat score vector out of this wheel. Pin NumPy 1.x (1.26 worked).

Put XRE 4.33 ahead of the runtime the wheel bundles (that copy is 4.31):

```bash
export LD_LIBRARY_PATH=/usr/local/xpu-4.33.0/lib64:$LD_LIBRARY_PATH
```

The model is the 2020 inference ResNet50 (`model` + `params`, about 98 MB of float weights):

`https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ResNet50.tar.gz`

```bash
~/paddle-xpu/bin/python examples/paddle_resnet50.py \
  dog.jpg ResNet50/model ResNet50/params imagenet_classes.txt
```

`examples/paddle_resnet50.py` calls `enable_xpu()` with the XPU fusion passes on. On a photo of a Samoyed the top class was **Samoyed at 0.923**. The timed section is `predictor.run()` only, after the tensor is already on the device, repeated on that same input: **0.68 ms**, about 1470 forwards per second on **die 0**. A stream of new images also pays for resize and the copies. Run one process per die to use both. The arithmetic is in [performance](performance.md).

`enable_xpu(10 * 1024 * 1024)` is the older form. 2.6.1 warns that the size argument is deprecated. Calling `enable_xpu()` with no argument is what the script does.

This graph is float. The quantize pass reported zero quantize ops. It is not an INT8 or INT4 model.
