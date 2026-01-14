"""Generate target face embeddings from a directory of photos."""

import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.face_detect import FaceDetector
from pipeline.face_recognize import FaceRecognizer
from utils.face_align import align_face
from utils.trt_inference import TRTInference
import config


def main():
    parser = argparse.ArgumentParser(description="Enroll target face embeddings")
    parser.add_argument("--images", required=True,
                        help="Directory containing target face photos")
    parser.add_argument("--output", default=config.TARGET_EMBEDDINGS_PATH,
                        help="Output path for embeddings .npy file")
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        print(f"Error: {args.images} is not a directory")
        sys.exit(1)

    # Initialize models
    print("Loading face detection model...")
    detector = FaceDetector()

    print("Loading face recognition model...")
    size = config.RECOGNITION_INPUT_SIZE
    rec_model = TRTInference(
        config.RECOGNITION_MODEL_PATH,
        config.RECOGNITION_ENGINE_PATH,
        fp16=True,
        input_shape=(1, 3, size, size),
    )

    # Process images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        print(f"Error: No image files found in {args.images}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    embeddings = []
    skipped = 0

    for filename in image_files:
        filepath = os.path.join(args.images, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"  Skipping {filename}: cannot read image")
            skipped += 1
            continue

        # Detect faces
        boxes, scores, landmarks = detector.detect(image)

        if len(boxes) == 0:
            print(f"  Skipping {filename}: no face detected")
            skipped += 1
            continue

        # Use the highest-confidence face
        best_idx = np.argmax(scores)
        lm = landmarks[best_idx]

        # Align and extract embedding
        aligned = align_face(image, lm, output_size=config.RECOGNITION_INPUT_SIZE)
        blob = (aligned.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        outputs = rec_model.infer(blob)
        embedding = outputs[0].flatten()
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)

        print(f"  {filename}: face detected (conf={scores[best_idx]:.3f})")

    if not embeddings:
        print("Error: No valid face embeddings extracted")
        sys.exit(1)

    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, embeddings_array)
    print(f"\nSaved {len(embeddings)} embeddings to {args.output}")
    print(f"Skipped {skipped} images")

    # Print statistics
    if len(embeddings) > 1:
        sims = embeddings_array @ embeddings_array.T
        np.fill_diagonal(sims, 0)
        avg_sim = sims.sum() / (len(embeddings) * (len(embeddings) - 1))
        min_sim = sims[sims > 0].min() if (sims > 0).any() else 0
        print(f"Inter-embedding similarity: avg={avg_sim:.3f}, min={min_sim:.3f}")


if __name__ == "__main__":
    main()
