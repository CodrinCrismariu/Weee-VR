"""Google's current MediaPipe hand landmark model (MediaPipe Tasks `hand_landmarker.task`, the 2023
"full" model) -> Core ML FP16, as a drop-in for HandLandmarks.mlpackage.

usage: ~/hands-venv/bin/python convert-landmarker.py <out-dir> [test-image]
(venv: torch (cpu), tensorflow-cpu, tf2onnx, onnx, onnx2torch, onnxruntime, coremltools, pillow)

Path: .tflite -> ONNX (tf2onnx) -> PyTorch (onnx2torch) -> traced -> Core ML ML Program, FP16 so it
runs on the Neural Engine. A wrapper keeps the app's interface: input "image" 1x3xSxS planar RGB in
0...1 (S = 224 here; the app reads S from the model), outputs "landmarks" 21x(x, y, z) normalised to
the crop (z in crop widths, like x), "scores" hand presence 0...1, "lr" handedness 0...1 and "world"
21x3 metric landmarks (m, around the hand's centre). The traced model is checked against the
original .tflite before it is written.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
import torch
from onnx2torch import convert as onnx_to_torch

import coremltools as ct

TASK = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
TEST_IMAGE = "https://storage.googleapis.com/mediapipe-assets/right_hands.jpg"


def load_tflite(cache: Path) -> bytes:
    task = cache / "hand_landmarker.task"
    if not task.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TASK, task)
    with zipfile.ZipFile(task) as z:
        return z.read("hand_landmarks_detector.tflite")


def tflite_outputs(model: bytes):
    """Run the original model; returns (interpreter, run(image NHWC) -> {name: array})."""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()

    def run(image: np.ndarray) -> dict:
        interpreter.set_tensor(inp["index"], image.astype(np.float32))
        interpreter.invoke()
        return {o["name"]: interpreter.get_tensor(o["index"]).copy() for o in outs}

    return inp, outs, run


def classify_outputs(outs) -> dict:
    """Map the tflite outputs by shape: 63 values -> image or world landmarks (told apart by name
    order: MediaPipe's are Identity (image), Identity_1 (presence), Identity_2 (handedness),
    Identity_3 (world))."""
    by_name = {o["name"]: o for o in outs}
    names = sorted(by_name)
    print("tflite outputs:", [(n, list(by_name[n]["shape"])) for n in names])
    return {"landmarks": names[0], "presence": names[1], "handedness": names[2], "world": names[3]}


class Wrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, size: int, order: list):
        super().__init__()
        self.net = net
        self.size = float(size)
        self.order = order          # index of landmarks, presence, handedness, world in net outputs

    def forward(self, image):
        outputs = self.net(image.permute(0, 2, 3, 1))      # model takes NHWC
        landmarks = outputs[self.order[0]].reshape(1, 21, 3) / self.size
        presence = outputs[self.order[1]].reshape(1, 1)
        handedness = outputs[self.order[2]].reshape(1, 1)
        world = outputs[self.order[3]].reshape(1, 21, 3)
        return landmarks, presence, handedness, world


def main(out: Path, test_image: str | None) -> None:
    cache = Path.home() / "handmodels-new"
    model = load_tflite(cache)
    inp, outs, run_tflite = tflite_outputs(model)
    size = int(inp["shape"][1])
    print("tflite input:", inp["name"], list(inp["shape"]), inp["dtype"])
    names = classify_outputs(outs)

    # tflite -> ONNX
    tflite_path = cache / "hand_landmarks_detector.tflite"
    tflite_path.write_bytes(model)
    onnx_model, _ = tf2onnx.convert.from_tflite(str(tflite_path), opset=13)
    onnx_outputs = [o.name for o in onnx_model.graph.output]
    print("onnx outputs:", onnx_outputs)
    # tf2onnx keeps the tflite tensor names
    order = [onnx_outputs.index(names[k]) for k in ("landmarks", "presence", "handedness", "world")]
    for node in onnx_model.graph.node:
        if any(o in (names["presence"], names["handedness"]) for o in node.output):
            print("presence/handedness produced by:", node.op_type)

    # ONNX -> torch, wrapped to the app's interface
    net = onnx_to_torch(onnx_model).eval()
    wrapper = Wrapper(net, size, order).eval()

    # Test input: a real hand photo, else noise.
    if test_image:
        from PIL import Image
        data = Path(test_image).read_bytes() if Path(test_image).exists() else urllib.request.urlopen(test_image).read()
        picture = Image.open(io.BytesIO(data)).convert("RGB")
        side = min(picture.size)
        picture = picture.crop(((picture.width - side) // 2, (picture.height - side) // 2,
                                (picture.width + side) // 2, (picture.height + side) // 2)).resize((size, size))
        nhwc = np.asarray(picture, dtype=np.float32)[None] / 255
    else:
        nhwc = np.random.default_rng(0).random((1, size, size, 3), dtype=np.float32)
    nchw = torch.from_numpy(nhwc.transpose(0, 3, 1, 2).copy())

    reference = run_tflite(nhwc)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, nchw)
        mine = [t.numpy() for t in traced(nchw)]
    ref_landmarks = reference[names["landmarks"]].reshape(1, 21, 3) / size
    print(f"presence tflite {reference[names['presence']].ravel()[0]:.4f} torch {mine[1].ravel()[0]:.4f}")
    print(f"handedness tflite {reference[names['handedness']].ravel()[0]:.4f} torch {mine[2].ravel()[0]:.4f}")
    error = np.abs(ref_landmarks - mine[0]).max()
    world_error = np.abs(reference[names["world"]].reshape(1, 21, 3) - mine[3]).max()
    print(f"max landmark error {error:.2e} (crop widths), world {world_error:.2e} m")
    print("landmarks x/y range", mine[0][0, :, :2].min(), mine[0][0, :, :2].max())
    assert error < 1e-3 and world_error < 1e-3, "torch conversion doesn't match the tflite model"

    # torch -> Core ML (FP16 for the Neural Engine)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, size, size), dtype=np.float32)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in ("landmarks", "scores", "lr", "world")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    mlmodel.short_description = "MediaPipe Tasks hand landmarks (2023 full model), Apache-2.0"
    destination = out / "HandLandmarks.mlpackage"
    import shutil
    shutil.rmtree(destination, ignore_errors=True)
    mlmodel.save(str(destination))
    total = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {destination} ({total:.1f} MB), input {size}x{size}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else TEST_IMAGE)
