"""X-VLA HTTP client for Rainbow Robotics RBY1 dual-arm mobile manipulator.

Wraps the FastAPI ``/act`` endpoint exposed by ``deploy.py`` so that the
caller only needs to pass raw images (HWC uint8) and a 16-D proprio vector.
Handles 16-D ↔ 20-D zero-padding internally.
"""

from __future__ import annotations

from typing import Tuple

import json_numpy
import numpy as np
import requests


class XVLARby1Client:
    """Stateless HTTP client that talks to the X-VLA inference server."""

    REAL_DIM = 16
    MODEL_DIM = 20
    DOMAIN_ID = 19

    # Must match datasets/domain_handler/rby1.py used during training.
    # Indices 0–13 are trained as delta (action - proprio); 14–15 are absolute.
    IDX_FOR_DELTA = list(range(14))
    IDX_FOR_MASK_PROPRIO = [14, 15]

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8010,
        domain_id: int = DOMAIN_ID,
    ) -> None:
        self.url = f"http://{host}:{port}/act"
        self.domain_id = domain_id

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def infer(
        self,
        head_img: np.ndarray,
        left_img: np.ndarray,
        right_img: np.ndarray,
        proprio_16d: np.ndarray,
        instruction: str,
        steps: int = 10,
    ) -> np.ndarray:
        """Run one inference call and return **absolute** 16-D actions.

        The model outputs *delta* actions for indices 0–13 (matching training).
        This method converts them back to absolute joint targets by adding the
        current proprio, so the caller can send them directly to the robot.

        Parameters
        ----------
        head_img, left_img, right_img : np.ndarray
            RGB images in HWC uint8 format (e.g. 224x224x3).
        proprio_16d : np.ndarray
            Current joint state, shape ``(16,)``.
        instruction : str
            Natural-language task description.
        steps : int
            Number of action steps to predict (server default is 10).

        Returns
        -------
        np.ndarray
            Action trajectory of shape ``(T, 16)`` — **absolute** joint targets.
        """
        proprio_16d = np.asarray(proprio_16d, dtype=np.float32)
        proprio_20d = np.zeros(self.MODEL_DIM, dtype=np.float32)
        proprio_20d[: self.REAL_DIM] = proprio_16d.copy()

        # Mask proprio indices that were masked during training
        for idx in self.IDX_FOR_MASK_PROPRIO:
            if idx < self.REAL_DIM:
                proprio_20d[idx] = 0.0

        payload = {
            "domain_id": self.domain_id,
            "proprio": json_numpy.dumps(proprio_20d),
            "language_instruction": instruction,
            "image0": json_numpy.dumps(np.asarray(head_img, dtype=np.uint8)),
            "image1": json_numpy.dumps(np.asarray(left_img, dtype=np.uint8)),
            "image2": json_numpy.dumps(np.asarray(right_img, dtype=np.uint8)),
            "steps": steps,
        }

        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        action = np.asarray(resp.json()["action"], dtype=np.float64)

        if action.ndim == 1:
            action = action.reshape(-1, self.MODEL_DIM)
        action = action[:, : self.REAL_DIM]

        # Convert delta → absolute for the indices that were delta-trained
        for idx in self.IDX_FOR_DELTA:
            if idx < self.REAL_DIM:
                action[:, idx] += float(proprio_16d[idx])

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def split_action(
        action_16d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a 16-D action vector into per-limb components.

        Layout (matches the training dataset assembled in data_conversion.ipynb):
            [0:7]   right arm joints   (robot_target_joints[8:15])
            [7:14]  left arm joints    (robot_target_joints[15:22])
            [14]    right gripper      (gripper_target[0])
            [15]    left gripper       (gripper_target[1])

        Returns
        -------
        right_joints : np.ndarray  shape ``(..., 7)``
        right_grip   : np.ndarray  shape ``(...,)``
        left_joints  : np.ndarray  shape ``(..., 7)``
        left_grip    : np.ndarray  shape ``(...,)``
        """
        a = np.asarray(action_16d)
        return a[..., 0:7], a[..., 14], a[..., 7:14], a[..., 15]

    def health_check(self) -> bool:
        """Return ``True`` if the server is reachable (simple GET)."""
        try:
            base = self.url.rsplit("/", 1)[0]
            resp = requests.get(base + "/docs", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
