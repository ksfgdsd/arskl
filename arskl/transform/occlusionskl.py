import random

import numpy as np

from ..registry import TRANSFORM



@TRANSFORM.register_module()
class JointOcclusion:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, results):
        all_kps = results['keypoint']  # (num_people, T, V, position)
        kp_shape = all_kps.shape
        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']  # (num_people, T, V)
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)
        occluded_kpscores = all_kpscores.copy()

        num_people, T, V = all_kpscores.shape
        num_occlusion = int(V * self.p)
        occlusion_index = np.random.choice(V, num_occlusion, replace=False)
        occluded_kpscores[:, :, occlusion_index] = 0

        results['keypoint_score'] = occluded_kpscores
        return results



@TRANSFORM.register_module()
class JointJitter:
    def __init__(self, jitter_amount=0.05, p=0.4):
        self.jitter_amount = jitter_amount
        self.p = p

    def __call__(self, results):
        all_kps = results['keypoint']  # (num_people, T, V, position)
        kp_shape = all_kps.shape
        jitter_kps = all_kps.copy()

        if random.random() < self.p:
            min_coords = np.min(jitter_kps, axis=(0, 1, 2))
            max_coords = np.max(jitter_kps, axis=(0, 1, 2))
            coord_range = max_coords - min_coords
            jitter = (np.random.rand(*kp_shape) - 0.5) * 2 * self.jitter_amount * coord_range
            jitter_kps += jitter

        results['keypoint'] = jitter_kps
        return results

