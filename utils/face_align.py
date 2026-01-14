"""Landmark-based affine alignment for face recognition (112x112 output)."""

import numpy as np
import cv2

# Standard ArcFace alignment reference landmarks for 112x112 output.
# These correspond to: left_eye, right_eye, nose, left_mouth, right_mouth.
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Estimate a 2x3 affine transform from src to dst using least squares.

    Both src_pts and dst_pts should be Nx2 arrays.
    """
    n = src_pts.shape[0]
    # Build system: for each point (x, y) -> (x', y')
    # [x y 1 0 0 0] [a] = [x']
    # [0 0 0 x y 1] [b]   [y']
    #                [c]
    #                [d]
    #                [e]
    #                [f]
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        A[2 * i] = [src_pts[i, 0], src_pts[i, 1], 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, src_pts[i, 0], src_pts[i, 1], 1]
        b[2 * i] = dst_pts[i, 0]
        b[2 * i + 1] = dst_pts[i, 1]

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
    ], dtype=np.float64)
    return M


def align_face(image: np.ndarray, landmarks: np.ndarray,
               output_size: int = 112) -> np.ndarray:
    """Warp a face region to a canonical 112x112 aligned image.

    Args:
        image: Full frame (BGR, HxWx3).
        landmarks: 5x2 array of facial landmarks
                   (left_eye, right_eye, nose, left_mouth, right_mouth).
        output_size: Output image size (square).

    Returns:
        Aligned face image (output_size x output_size x 3, BGR).
    """
    src_pts = landmarks.astype(np.float32)
    dst_pts = ARCFACE_REF_LANDMARKS.copy()

    # Scale reference landmarks if output size differs from 112
    if output_size != 112:
        scale = output_size / 112.0
        dst_pts *= scale

    M = estimate_affine(src_pts, dst_pts)
    aligned = cv2.warpAffine(image, M, (output_size, output_size),
                             borderValue=(0, 0, 0))
    return aligned
