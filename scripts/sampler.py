from torch.utils.data import Sampler
import numpy as np


class SteeringBalancedSampler(Sampler):
    """
    This sampler is designed for steering-angle datasets where the majority of samples
    correspond to near-zero steering values (vehicle driving straight).
    """

    def __init__(self, dataset, low_fraction=0.10, threshold_ratio=0.10, shuffle=True):
        """
        @brief Constructor for SteeringBalancedSampler.

        The constructor analyzes steering angles, identifies low-steering 
        and high-steering samples, and keeps: all high-steering samples and at most 
        low_fraction * total dataset size of low-steering samples

        @param dataset The dataset to sample from.
        @param low_fraction The maximum fraction of low-steering samples to keep.
        @param threshold_ratio Fraction of max steering defining "low steering".
        @param shuffle Whether to shuffle the indices after balancing.
        """

        self.low_fraction = low_fraction
        self.shuffle = shuffle
        
        # Extract raw samples from the original dataset
        samples = [dataset.dataset.samples[i] for i in dataset.indices]

        # Collect steering data
        steer = np.array([s["steering_angle"] for s in samples])
        max_steer = np.array([s["max_steering_angle"] for s in samples])

        # Normalize steering angles to [-1, 1]
        norm_steer = steer / max_steer
        abs_norm = np.abs(norm_steer)

        # Identify low and high steering samples using threshold as fraction
        low_indices = np.where(abs_norm < threshold_ratio)[0]
        high_indices = np.where(abs_norm >= threshold_ratio)[0]

        # Keep at most (low_fraction * total_dataset) low-steering samples
        max_low_allowed = int(len(samples) * low_fraction)

        # How many low samples we will actually keep
        keep_low = min(len(low_indices), max_low_allowed)

        # Downsample low-steering indices
        if keep_low > 0:
            low_indices = np.random.choice(low_indices, size=keep_low, replace=False)
        else:
            low_indices = np.array([], dtype=np.int64)

        # Combine all high-steering samples with selected low-steering samples
        self.indices = np.concatenate([high_indices, low_indices])

        # Shuffle final indices if required
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
