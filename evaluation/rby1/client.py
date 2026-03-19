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
        """Run one inference call and return *trimmed* 16-D actions.

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
            Action trajectory of shape ``(T, 16)`` — already trimmed from the
            model's 20-D output.
        """
        proprio_20d = np.zeros(self.MODEL_DIM, dtype=np.float32)
        proprio_20d[: self.REAL_DIM] = np.asarray(proprio_16d, dtype=np.float32)

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
        return action[:, : self.REAL_DIM]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def split_action(
        action_16d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a 16-D action vector into per-limb components.

        Layout (matches the training dataset):
            [0:7]   right arm joints
            [7]     right gripper
            [8:15]  left arm joints  (index 7:14 from 0-based 16-D)
            [15]    left gripper     (index 14:16 from 0-based 16-D)

        Note: The actual rby1 dataset layout is:
            right_arm(7) + right_gripper(1) + left_arm(7) + left_gripper(1)

        Returns
        -------
        right_joints : np.ndarray  shape ``(..., 7)``
        right_grip   : np.ndarray  shape ``(...,)``
        left_joints  : np.ndarray  shape ``(..., 7)``
        left_grip    : np.ndarray  shape ``(...,)``
        """
        a = np.asarray(action_16d)
        return a[..., 0:7], a[..., 7], a[..., 8:15], a[..., 15]

    def health_check(self) -> bool:
        """Return ``True`` if the server is reachable (simple GET)."""
        try:
            base = self.url.rsplit("/", 1)[0]
            resp = requests.get(base + "/docs", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
