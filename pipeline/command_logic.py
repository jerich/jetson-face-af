"""AF steering decision engine with debouncing and stuck detection."""

from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np

import config
from pipeline.overlay_detect import AFState

logger = logging.getLogger(__name__)


class Command(Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    FN1 = "fn1"


@dataclass
class FaceInfo:
    """Information about a detected face for command decisions."""
    box: np.ndarray           # [x1, y1, x2, y2]
    is_target: bool
    similarity: float
    af_state: AFState

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0


class CommandLogic:
    """Determines which commands to send based on face and AF state.

    Implements:
    - Debouncing: requires INITIAL_CONFIRM_FRAMES consistent decisions
      before sending the first command in a new sequence.
    - Stuck detection: after STUCK_THRESHOLD commands in the same direction
      without the AF moving to the target, tries an alternative.
    """

    def __init__(self):
        self._confirm_count = 0
        self._last_pending_command = Command.NONE
        self._confirmed = False  # True once debounce period passes

        self._same_dir_count = 0
        self._last_sent_command = Command.NONE

    def update(self, faces: list[FaceInfo]) -> Command:
        """Evaluate the current frame and return a command to send.

        Args:
            faces: List of FaceInfo for all detected faces in this frame.

        Returns:
            Command to send (Command.NONE if no action needed).
        """
        target = self._find_target(faces)
        af_face = self._find_af_face(faces)

        # Debug: log face states
        if logger.isEnabledFor(logging.DEBUG):
            face_summary = ", ".join(
                f"[{'T' if f.is_target else '_'}{f.af_state.value[0].upper()}@{int(f.center_x)}]"
                for f in faces
            )
            logger.debug(f"Faces: {face_summary}")

        # No target visible — nothing to do
        if target is None:
            self._reset()
            return Command.NONE

        # Target already has AF — success, reset state
        if target.af_state in (AFState.TRACKED, AFState.ACTIVE_FOCUS):
            self._reset()
            return Command.NONE

        # Check if AF is on a face that overlaps with target (same person, different detection)
        # This handles cases where AF is on an eye region while target is the full face
        if af_face is not None and self._boxes_overlap(target.box, af_face.box):
            self._reset()
            return Command.NONE

        # No active AF found anywhere — nothing to steer away from
        # (This includes the case where white is misclassified as gray)
        if af_face is None:
            self._reset()
            return Command.NONE

        # Only send commands if there's a gray (DETECTED) AF box inside target region
        # that we could potentially switch to
        if not self._has_detected_af_inside(faces, target):
            # No available AF point inside target to switch to
            self._reset()
            return Command.NONE

        # At this point: AF is outside target, and there's a DETECTED inside target
        desired = self._compute_direction(target, af_face)
        if desired == Command.FN1:
            logger.debug(
                f"FN1: vertical stack, target@({int(target.center_x)},{int(target.center_y)}), "
                f"af@({int(af_face.center_x)},{int(af_face.center_y)}), "
                f"dx={int(target.center_x - af_face.center_x)}, dy={int(target.center_y - af_face.center_y)}"
            )

        # Debouncing: require consistent frames before first command
        if not self._confirmed:
            if desired == self._last_pending_command and desired != Command.NONE:
                self._confirm_count += 1
            else:
                self._confirm_count = 1
                self._last_pending_command = desired

            if self._confirm_count >= config.INITIAL_CONFIRM_FRAMES:
                self._confirmed = True
            else:
                return Command.NONE

        # Stuck detection
        if desired == self._last_sent_command and desired != Command.NONE:
            self._same_dir_count += 1
        else:
            self._same_dir_count = 0

        if self._same_dir_count >= config.STUCK_THRESHOLD:
            # Try alternative: opposite direction or fn1
            self._same_dir_count = 0
            if desired == Command.LEFT:
                desired = Command.RIGHT
            elif desired == Command.RIGHT:
                desired = Command.LEFT
            else:
                desired = Command.FN1

        self._last_sent_command = desired
        return desired

    def _compute_direction(self, target: FaceInfo, af_face: FaceInfo) -> Command:
        """Determine left/right/fn1 based on relative face positions."""
        dx = target.center_x - af_face.center_x
        dy = target.center_y - af_face.center_y

        # If faces are more vertically stacked than horizontally separated,
        # use fn1 since left/right won't navigate correctly
        if abs(dy) > abs(dx) * 2 and abs(dx) < (target.box[2] - target.box[0]):
            return Command.FN1

        if dx < 0:
            return Command.LEFT
        else:
            return Command.RIGHT

    def _find_target(self, faces: list[FaceInfo]) -> FaceInfo | None:
        """Find the target face (highest similarity above threshold)."""
        targets = [f for f in faces if f.is_target]
        if not targets:
            return None
        return max(targets, key=lambda f: f.similarity)

    def _find_af_face(self, faces: list[FaceInfo]) -> FaceInfo | None:
        """Find the face that currently has AF (TRACKED or ACTIVE_FOCUS)."""
        for face in faces:
            if face.af_state in (AFState.TRACKED, AFState.ACTIVE_FOCUS):
                return face
        return None

    def _boxes_overlap(self, box1: np.ndarray, box2: np.ndarray,
                       min_iou: float = 0.3) -> bool:
        """Check if two bounding boxes overlap significantly.

        Args:
            box1, box2: Bounding boxes [x1, y1, x2, y2].
            min_iou: Minimum intersection-over-union to consider overlapping.

        Returns:
            True if boxes overlap by at least min_iou.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return False  # No intersection

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou >= min_iou

    def _has_detected_af_inside(self, faces: list[FaceInfo],
                                target: FaceInfo) -> bool:
        """Check if there's a DETECTED (gray) AF point inside the target region.

        We only want to send commands if there's an available AF point inside
        the target that we could switch to.

        Args:
            faces: All detected faces.
            target: The target face.

        Returns:
            True if there's a gray AF box overlapping with target.
        """
        for face in faces:
            if face.af_state == AFState.DETECTED:
                if self._boxes_overlap(target.box, face.box, min_iou=0.2):
                    return True
        return False

    def _reset(self):
        """Reset debouncing and stuck detection state."""
        self._confirm_count = 0
        self._last_pending_command = Command.NONE
        self._confirmed = False
        self._same_dir_count = 0
        self._last_sent_command = Command.NONE
