
import torch
import numpy as np

from mmedit.datasets.pipelines import blur_kernels as blur_kernels
from ..registry import PIPELINES
@PIPELINES.register_module()
class SplitIntoPatches:
    """Split images into patches of given size.

    Args:
        keys (Sequence[str]): Required keys to be split.
        patch_size (int): Size of each patch.
    """

    def __init__(self, keys, patch_size):
        self.keys = keys
        self.patch_size = patch_size

    def __call__(self, results):
        """Call function to split images into patches.

        Args:
            results (dict): A dict containing the necessary information and data.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if not isinstance(results[key], np.ndarray):
                raise TypeError(f'results["{key}"] should be a numpy array, '
                                f'but got {type(results[key])}')

            # Split the image into patches
            patches = []
            h, w, c = results[key].shape  # Assuming results[key] is of shape (H, W, C)
            for i in range(0, h, self.patch_size):
                for j in range(0, w, self.patch_size):
                    patch = results[key][i:i + self.patch_size, j:j + self.patch_size, :]
                    # Ensure the patch is of the correct size
                    if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                        patches.append(patch)

            # Convert patches to numpy array for further processing
            results[key] = np.array(patches)  # Shape will be (num_patches, patch_size, patch_size, channels)
        return results


@PIPELINES.register_module()
class MergePatches(object):
    def __init__(self, keys, patch_size):
        self.keys = keys
        self.patch_size = patch_size

    def __call__(self, results):
        for key in self.keys:
            patches = results[key]
            h_patches = int(len(patches) ** 0.5)  # assuming square grid of patches
            w_patches = h_patches
            full_img = np.zeros((h_patches * self.patch_size, w_patches * self.patch_size, 3))
            idx = 0
            for i in range(h_patches):
                for j in range(w_patches):
                    full_img[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = patches[idx]
                    idx += 1
            results[key] = full_img
        return results
