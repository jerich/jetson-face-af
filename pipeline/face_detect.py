"""SCRFD face detection via TensorRT."""

import numpy as np
import cv2

import config
from utils.trt_inference import TRTInference

# SCRFD uses 3 feature strides with 2 anchors each
_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2


class FaceDetector:
    """SCRFD-based face detector with 5-point landmarks."""

    def __init__(self):
        self._input_size = config.SCRFD_INPUT_SIZE
        self._model = TRTInference(
            config.SCRFD_MODEL_PATH,
            config.SCRFD_ENGINE_PATH,
            fp16=True,
            input_shape=(1, 3, self._input_size, self._input_size),
        )

        # Log output tensor info for debugging
        print(f"SCRFD outputs ({len(self._model.outputs)} tensors):")
        for i, out in enumerate(self._model.outputs):
            print(f"  [{i}] {out['name']}: shape={list(out['shape'])}")

    def detect(self, image: np.ndarray,
               conf_threshold: float = None,
               nms_threshold: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect faces in an image.

        Args:
            image: BGR image (HxWx3).
            conf_threshold: Confidence threshold (default from config).
            nms_threshold: NMS IoU threshold (default from config).

        Returns:
            Tuple of:
                boxes: Nx4 array of [x1, y1, x2, y2] in original image coords.
                scores: N array of confidence scores.
                landmarks: Nx5x2 array of facial landmarks in original coords.
        """
        if conf_threshold is None:
            conf_threshold = config.SCRFD_CONF_THRESHOLD
        if nms_threshold is None:
            nms_threshold = config.SCRFD_NMS_THRESHOLD

        # Preprocess: letterbox resize to input_size x input_size
        img_h, img_w = image.shape[:2]
        input_blob, scale, pad_w, pad_h = self._preprocess(image)

        # Run inference
        outputs = self._model.infer(input_blob)

        # Postprocess: decode boxes, landmarks, apply NMS
        boxes, scores, landmarks = self._postprocess(
            outputs, scale, pad_w, pad_h, img_w, img_h, conf_threshold
        )

        if len(boxes) == 0:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty((0, 5, 2), dtype=np.float32))

        # NMS
        indices = self._nms(boxes, scores, nms_threshold)
        return boxes[indices], scores[indices], landmarks[indices]

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize and normalize for SCRFD input."""
        img_h, img_w = image.shape[:2]
        size = self._input_size

        scale = min(size / img_w, size / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to square
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        padded = np.full((size, size, 3), 0, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Normalize: (pixel - 127.5) / 128.0
        blob = (padded.astype(np.float32) - 127.5) / 128.0
        # HWC -> CHW -> NCHW
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        return blob, scale, pad_w, pad_h

    def _postprocess(self, outputs: list[np.ndarray],
                     scale: float, pad_w: int, pad_h: int,
                     img_w: int, img_h: int,
                     conf_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode SCRFD outputs into boxes, scores, and landmarks."""
        all_boxes = []
        all_scores = []
        all_landmarks = []

        # SCRFD outputs: 9 tensors grouped by type then stride
        # Order: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
        num_strides = len(_STRIDES)
        for idx, stride in enumerate(_STRIDES):
            score_out = outputs[idx]
            bbox_out = outputs[idx + num_strides]
            kps_out = outputs[idx + num_strides * 2]

            # Flatten to 2D if needed
            scores = score_out.reshape(-1)
            bboxes = bbox_out.reshape(-1, 4)
            kps = kps_out.reshape(-1, 10)

            # Filter by confidence
            mask = scores > conf_threshold
            if not mask.any():
                continue

            scores = scores[mask]
            bboxes = bboxes[mask]
            kps = kps[mask]

            # Generate anchor centers
            feat_h = self._input_size // stride
            feat_w = self._input_size // stride
            anchors = self._generate_anchors(feat_w, feat_h, stride)
            anchors = anchors.reshape(-1, 2)
            # Apply mask to anchors (accounting for _NUM_ANCHORS per position)
            anchor_centers = np.repeat(anchors, _NUM_ANCHORS, axis=0)[mask]

            # Decode bboxes: distance from anchor center
            x1 = anchor_centers[:, 0] - bboxes[:, 0] * stride
            y1 = anchor_centers[:, 1] - bboxes[:, 1] * stride
            x2 = anchor_centers[:, 0] + bboxes[:, 2] * stride
            y2 = anchor_centers[:, 1] + bboxes[:, 3] * stride
            decoded_boxes = np.stack([x1, y1, x2, y2], axis=1)

            # Decode landmarks
            decoded_kps = np.zeros_like(kps)
            for k in range(5):
                decoded_kps[:, k * 2] = anchor_centers[:, 0] + kps[:, k * 2] * stride
                decoded_kps[:, k * 2 + 1] = anchor_centers[:, 1] + kps[:, k * 2 + 1] * stride

            # Map back to original image coordinates
            decoded_boxes[:, [0, 2]] = (decoded_boxes[:, [0, 2]] - pad_w) / scale
            decoded_boxes[:, [1, 3]] = (decoded_boxes[:, [1, 3]] - pad_h) / scale
            decoded_kps[:, 0::2] = (decoded_kps[:, 0::2] - pad_w) / scale
            decoded_kps[:, 1::2] = (decoded_kps[:, 1::2] - pad_h) / scale

            # Clip to image boundaries
            decoded_boxes[:, [0, 2]] = np.clip(decoded_boxes[:, [0, 2]], 0, img_w)
            decoded_boxes[:, [1, 3]] = np.clip(decoded_boxes[:, [1, 3]], 0, img_h)

            all_boxes.append(decoded_boxes)
            all_scores.append(scores)
            all_landmarks.append(decoded_kps.reshape(-1, 5, 2))

        if not all_boxes:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty((0, 5, 2), dtype=np.float32))

        return (np.concatenate(all_boxes).astype(np.float32),
                np.concatenate(all_scores).astype(np.float32),
                np.concatenate(all_landmarks).astype(np.float32))

    def _generate_anchors(self, feat_w: int, feat_h: int, stride: int) -> np.ndarray:
        """Generate anchor center coordinates for a given feature map size."""
        shifts_x = np.arange(feat_w) * stride + stride // 2
        shifts_y = np.arange(feat_h) * stride + stride // 2
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        anchors = np.stack([shift_x.ravel(), shift_y.ravel()], axis=1)
        return anchors.astype(np.float32)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray,
             threshold: float) -> np.ndarray:
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int64)
