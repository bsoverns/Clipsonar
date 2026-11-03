"""Face matching with InsightFace. Importing this module does not load the model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .common import IMAGE_EXTS, JobError, Reporter

DEFAULT_THRESHOLD = 0.40  # ArcFace cosine similarity. Higher is stricter.
MODEL_NAME = "buffalo_l"


def gpu_available() -> bool:
    import onnxruntime

    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def read_image(path: Path) -> np.ndarray | None:
    """Read an image, or return None. cv2.imread can't open non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def reference_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def reference_fingerprint(folder: Path) -> str:
    """Identify a set of sample images by name and size, to notice when it changes."""
    digest = hashlib.sha1()
    for path in reference_files(folder):
        digest.update(f"{path.relative_to(folder)}|{path.stat().st_size}\n".encode("utf-8", "replace"))
    return digest.hexdigest()


class FaceMatcher:
    """Scores images against the faces found in a folder of sample images."""

    def __init__(self, analyzer, reference: np.ndarray):
        self._analyzer = analyzer
        self._reference = reference

    @classmethod
    def load(cls, references: Path, use_gpu: bool, rep: Reporter) -> FaceMatcher:
        analyzer = _load_analyzer(use_gpu, rep)
        rep.log(f"Reading sample images from {references}")
        return cls(analyzer, _build_reference(analyzer, Path(references), rep))

    def score(self, image: np.ndarray) -> float | None:
        """Return the best score across all faces in the image, or None if there are no faces."""
        faces = self._analyzer.get(image)
        if not faces:
            return None
        return max(float((self._reference @ f.normed_embedding).max()) for f in faces)


def _load_analyzer(use_gpu: bool, rep: Reporter):
    import onnxruntime

    if use_gpu and hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls()

    from insightface.app import FaceAnalysis

    rep.log(f"Loading face model {MODEL_NAME} (about 300 MB is downloaded on first use). "
            "The model is licensed by InsightFace for non-commercial research only.")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    analyzer = FaceAnalysis(name=MODEL_NAME, providers=providers,
                            allowed_modules=["detection", "recognition"])
    analyzer.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    active = analyzer.models["detection"].session.get_providers()
    using_gpu = "CUDAExecutionProvider" in active
    rep.log(f"Face model ready, running on {'GPU' if using_gpu else 'CPU'}.")
    if use_gpu and not using_gpu:
        rep.log("WARNING: GPU was requested but CUDA is not available, so this will be slower.")
    return analyzer


def _build_reference(analyzer, folder: Path, rep: Reporter) -> np.ndarray:
    embeddings = []
    for path in reference_files(folder):
        rep.check()
        name = path.relative_to(folder)
        image = read_image(path)
        if image is None:
            rep.log(f"  skipped {name}: could not read it")
            continue
        faces = analyzer.get(image)
        if not faces:
            rep.log(f"  skipped {name}: no face found")
            continue
        # The largest face is assumed to be the subject of a sample image.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(face.normed_embedding)
    if not embeddings:
        raise JobError(f"No usable faces found in the sample images in {folder}.")
    rep.log(f"Using {len(embeddings)} sample face(s).")
    return np.stack(embeddings)
