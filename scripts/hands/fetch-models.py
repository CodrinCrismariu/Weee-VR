"""Fetch MediaPipe Hand (palm detector + 21-landmark model) as Core ML and convert to FP16.

usage: python fetch-models.py <out-dir>       (run with ~/calib/venv/bin/python in WSL)

Source: huggingface.co/hugging-mac/mediapipe-hand-coreml (Qualcomm AI Hub's ONNX export of Google's
MediaPipe Hand, converted to Core ML; Apache-2.0 upstream). Those packages compute in FP32, which
Core ML can only run on the CPU or GPU; recompiling the same program with FP16 compute precision
lets it run on the Neural Engine, keeping the GPU free for rendering. Inputs/outputs stay FP32.
Writes <out-dir>/HandDetector.mlpackage and <out-dir>/HandLandmarks.mlpackage.
"""
import shutil
import sys
import urllib.request
from pathlib import Path

import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load as load_milproto

REPO = "https://huggingface.co/hugging-mac/mediapipe-hand-coreml/resolve/main/coreml"
MODELS = {"hand_detector": "HandDetector", "hand_landmark_detector": "HandLandmarks"}
FILES = ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"]


def main(out: Path) -> None:
    cache = out / "_fp32"
    for source, name in MODELS.items():
        package = cache / f"{source}.mlpackage"
        for file in FILES:
            target = package / file
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{REPO}/{source}.mlpackage/{file}", target)

        model = ct.models.MLModel(str(package), skip_model_load=True)
        spec = model.get_spec()
        program = load_milproto(spec, specification_version=spec.specificationVersion,
                                file_weights_dir=model.weights_dir)
        fp16 = ct.convert(program, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                          minimum_deployment_target=ct.target.iOS17, skip_model_load=True)
        destination = out / f"{name}.mlpackage"
        shutil.rmtree(destination, ignore_errors=True)
        fp16.save(str(destination))
        inputs = [(i.name, list(i.type.multiArrayType.shape)) for i in fp16.get_spec().description.input]
        outputs = [(o.name, list(o.type.multiArrayType.shape)) for o in fp16.get_spec().description.output]
        size = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()) / 1e6
        print(f"{name}: in {inputs} out {outputs}, {size:.1f} MB")
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
