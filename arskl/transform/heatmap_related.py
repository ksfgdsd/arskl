import numpy as np

from pyskl.datasets.builder import PIPELINES

EPS = 1e-3


@PIPELINES.register_module()
class GeneratePoseVector:

    def __init__(self,
                 sigma=0.6,
                 hw=(64, 64),
                 use_score=True,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 is_compression=False,
                 with_kp=True,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb=(0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb=(1, 3, 7, 8, 9, 13, 14, 15),
                 scaling=1.):

        self.sigma = sigma
        self.hw = hw
        self.use_score = use_score
        self.with_kp = with_kp

        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling
        self.is_compression = is_compression

    def generate_a_vector(self, arr, centers, max_values):
        """Generate vector for one keypoint in one frame.  Simple Disentagled coordinate Representation (SimDR)

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h + img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = self.hw
        arr_x, arr_y = arr[:img_w], arr[img_w:]
        w = 3

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - w * sigma), 0)
            ed_x = min(int(mu_x + w * sigma) + 1, img_w)
            st_y = max(int(mu_y - w * sigma), 0)
            ed_y = min(int(mu_y + w * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue

            sigma_sq = sigma ** 2

            patch1 = max_value * np.exp(-((x - mu_x) ** 2) / (2 * sigma_sq))
            patch2 = max_value * np.exp(-((y - mu_y) ** 2) / (2 * sigma_sq))

            np.maximum(arr_x[st_x:ed_x], patch1, out=arr_x[st_x:ed_x])
            np.maximum(arr_y[st_y:ed_y], patch2, out=arr_y[st_y:ed_y])

    def generate_a_vector2(self, arr, centers, max_values):

        sigma = self.sigma
        img_h, img_w = self.hw
        arr_x, arr_y = arr[:img_w], arr[img_w:]
        w = 3

        # Pre-calculation for performance
        sigma_sq = sigma ** 2
        coef_x = 1 / (2 * sigma_sq)
        coef_y = 1 / (2 * sigma_sq)

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - w * sigma), 0)
            ed_x = min(int(mu_x + w * sigma) + 1, img_w)
            st_y = max(int(mu_y - w * sigma), 0)
            ed_y = min(int(mu_y + w * sigma) + 1, img_h)

            # Using np.meshgrid to calculate x and y in one go
            x = np.arange(st_x, ed_x, dtype=np.float32)
            y = np.arange(st_y, ed_y, dtype=np.float32)

            if not (len(x) and len(y)):
                continue

            # Calculate patches using vectorized operations
            patch1 = max_value * np.exp(-((x - mu_x) ** 2) * coef_x)
            patch2 = max_value * np.exp(-((y - mu_y) ** 2) * coef_y)

            # Use np.maximum in a vectorized way
            arr_x[st_x:ed_x] = np.maximum(arr_x[st_x:ed_x], patch1)
            arr_y[st_y:ed_y] = np.maximum(arr_y[st_y:ed_y], patch2)


    def generate_a_limb_vector(self, arr, cors, scores, is_compression):
        if is_compression:
            x_coords = [cor[0] for cor in cors]
            y_coords = [cor[0] for cor in cors]
            M_x = sum(x * c for x, c in zip(x_coords, scores)) / sum(scores)
            M_y = sum(y * c for y, c in zip(y_coords, scores)) / sum(scores)
            center = np.array([M_x, M_y])
            self.generate_a_vector(arr, center,  np.array([np.min(scores)]))
        else:
            # hyper heatmap
            for cor, score in zip(cors, scores):
                self.generate_a_vector2(arr, cor, score)

    def generate_vector(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_vector(arr[i], kps[:, i], max_values[:, i])
        else:
            for i, limb in enumerate(self.skeletons):
                cors = []
                scores = []
                for idx in limb:
                    cors.append(kps[:, idx])
                    scores.append(max_values[:, idx])
                self.generate_a_limb_vector(arr[i], cors, scores, self.is_compression)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0

        num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, img_w + img_h], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else np.ones_like(all_kpscores[:, i])

            self.generate_vector(ret[i], kps, kpscores)
        return ret

    def __call__(self, results):
        vector = self.gen_an_aug(results)
        key = 'vectors'
        results[key] = vector
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str
