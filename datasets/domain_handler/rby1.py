from __future__ import annotations
import numpy as np, torch, random
from mmengine import fileio
from scipy.interpolate import interp1d
from PIL import Image
from .base import DomainHandler
from ..utils import read_parquet


class RBY1Handler(DomainHandler):
    """
    Domain handler for Rainbow Robotics RBY1 dual-arm mobile manipulator.

    Dataset format: LeRobot v2.1 with images embedded in parquet (no MP4 videos).
    Action: 16D (right 7 joints + right gripper + left 7 joints + left gripper)
    State:  16D (14 arm joints + 2 grippers)
    Cameras: head_image, left_wrist_image, right_wrist_image (CHW uint8, 224x224)
    """

    IMAGE_KEYS = ["head_image", "left_wrist_image", "right_wrist_image"]
    ACTION_KEY = "actions"
    idx_for_delta = list(range(14))
    idx_for_mask_proprio = [14, 15]

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None, **kwargs):
        item = self.meta["datalist"][traj_idx]

        episode_index = item["episode_index"]
        episode_chunk = episode_index // self.meta["chunks_size"]
        data_path = fileio.join_path(
            self.meta["root_path"],
            self.meta["data_path"],
        ).format(episode_chunk=episode_chunk, episode_index=episode_index)

        data = read_parquet(data_path)

        all_action = np.asarray(data[self.ACTION_KEY], dtype=np.float32)

        images_by_view = []
        for key in self.IMAGE_KEYS:
            if key in data and data[key] is not None:
                raw = np.asarray(data[key], dtype=np.uint8)
                if raw.ndim == 4 and raw.shape[1] in (1, 3, 4):
                    raw = raw.transpose(0, 2, 3, 1)
                images_by_view.append(raw)
            else:
                images_by_view.append(None)

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        for v in range(min(self.num_views, len(images_by_view))):
            if images_by_view[v] is not None:
                image_mask[v] = True

        ins = item["tasks"][0] if "tasks" in item else data.get("prompt", [""])[0]

        freq = float(self.meta.get("fps", 15))
        qdur = 1.0
        T = all_action.shape[0]
        t = np.arange(T, dtype=np.float64) / freq

        margin = max(1, int(freq))
        idxs = list(range(1, T - margin))
        if training:
            random.shuffle(idxs)

        all_action_interp = interp1d(
            t, all_action, axis=0, bounds_error=False,
            fill_value=(all_action[0], all_action[-1]),
        )

        for idx in idxs:
            imgs = []
            for v in range(min(self.num_views, len(images_by_view))):
                if images_by_view[v] is not None:
                    frame = images_by_view[v][idx]
                    imgs.append(image_aug(Image.fromarray(frame)))
                else:
                    imgs.append(None)

            if imgs[0] is None:
                continue
            for i in range(len(imgs)):
                if imgs[i] is None:
                    imgs[i] = torch.zeros_like(imgs[0])
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            image_input = torch.stack(imgs, 0)

            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            cur_action = torch.tensor(all_action_interp(q), dtype=torch.float32)

            if (cur_action[1] - cur_action[0]).abs().max() < 1e-5:
                continue

            cur_action = torch.cat([
                cur_action,
                torch.zeros((cur_action.shape[0], 20 - cur_action.shape[1])),
            ], dim=-1)

            if lang_aug_map is not None and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": cur_action,
                "idx_for_delta": self.idx_for_delta,
                "idx_for_mask_proprio": self.idx_for_mask_proprio,
            }
