# Third-party notices

Clipsonar's own code is under the [MIT License](LICENSE). It relies on the projects below, which keep their own licenses. Check each project's current terms for anything that matters to your use.

## Face model: not MIT

Clipsonar downloads the InsightFace **`buffalo_l`** model the first time it runs a scan. The model is not part of this repository or of Clipsonar's release builds.

InsightFace's README says that its code is MIT, but that models downloaded through its Python library are "available for non-commercial research purposes only". For licensing the open-source recognition models, including `buffalo_l`, it directs people to contact `recognition-oss-pack@insightface.ai`.

**The MIT License on Clipsonar does not extend to this model.** If you want to use Clipsonar commercially, you need a license from InsightFace or a different model. See the [InsightFace license section](https://github.com/deepinsight/insightface#license).

## Direct dependencies

| Project | Used for | License |
|---|---|---|
| [InsightFace](https://github.com/deepinsight/insightface) (code) | Face detection and recognition | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Running the face model | MIT |
| [ml_dtypes](https://github.com/jax-ml/ml_dtypes) | Dependency support for ONNX | Apache 2.0 |
| [NumPy](https://numpy.org/) | Array math | BSD 3-Clause |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | Reading images and video frames | MIT (packaging), Apache 2.0 (OpenCV), LGPL 2.1 (the FFmpeg libraries it bundles) |
| [PySide6 / Qt](https://www.qt.io/qt-for-python) | Desktop UI | LGPL 3.0 |

InsightFace also pulls in further packages, including SciPy, scikit-image, scikit-learn, Albumentations and Matplotlib.

Each release build includes `THIRD_PARTY_LICENSES.md`, generated at build time, with the name, version and license of every Python package in that build. Qt's libraries ship as separate files in the build's folder, so you can replace them, as the LGPL allows.

## Installed separately

| Project | Used for | License |
|---|---|---|
| [FFmpeg](https://ffmpeg.org/legal.html) | Cutting clips and building montages | LGPL 2.1+ or GPL 2+, depending on the build |

Clipsonar runs FFmpeg as a separate program and does not include it. You install it yourself.

## GPU only

The `gpu` install option adds [NVIDIA's CUDA and cuDNN libraries](https://developer.nvidia.com/cuda-toolkit), which are covered by NVIDIA's license terms.
