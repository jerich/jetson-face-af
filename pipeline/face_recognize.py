"""MobileFaceNet face recognition via TensorRT."""

import numpy as np

import config
from utils.trt_inference import TRTInference
from utils.face_align import align_face


class FaceRecognizer:
    """Computes face embeddings and matches against target identity."""

    def __init__(self, target_embeddings_path: str = None):
        size = config.RECOGNITION_INPUT_SIZE
        self._model = TRTInference(
            config.RECOGNITION_MODEL_PATH,
            config.RECOGNITION_ENGINE_PATH,
            fp16=True,
            input_shape=(1, 3, size, size),
        )
        self._threshold = config.RECOGNITION_THRESHOLD

        # Load pre-computed target embeddings
        path = target_embeddings_path or config.TARGET_EMBEDDINGS_PATH
        self._target_embeddings = np.load(path)  # shape: (N, 512)
        # Normalize target embeddings
        norms = np.linalg.norm(self._target_embeddings, axis=1, keepdims=True)
        self._target_embeddings = self._target_embeddings / norms

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Compute a 512-dim embedding for a detected face.

        Args:
            image: Full frame (BGR, HxWx3).
            landmarks: 5x2 facial landmarks for this face.

        Returns:
            Normalized 512-dim embedding vector.
        """
        aligned = align_face(image, landmarks, output_size=config.RECOGNITION_INPUT_SIZE)
        blob = self._preprocess(aligned)
        outputs = self._model.infer(blob)
        embedding = outputs[0].flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def match_target(self, embedding: np.ndarray) -> float:
        """Compute max cosine similarity between embedding and target embeddings.

        Returns:
            Maximum cosine similarity score. Values above RECOGNITION_THRESHOLD
            indicate a match.
        """
        similarities = self._target_embeddings @ embedding
        return float(np.max(similarities))

    def is_target(self, embedding: np.ndarray) -> tuple[bool, float]:
        """Check if an embedding matches the target identity.

        Returns:
            Tuple of (is_match, similarity_score).
        """
        score = self.match_target(embedding)
        return score > self._threshold, score

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Normalize aligned face for MobileFaceNet input."""
        blob = (aligned_face.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        return blob
