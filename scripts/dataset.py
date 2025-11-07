import os
import torch
import csv
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image


def preprocessing_transform(height, width):
    """
    @brief Creates the preprocessing transformation pipeline.
    
    @param height Desired image height after resizing.
    @param width Desired image width after resizing.

    @return A torchvision Compose object with the preprocessing transforms.
    """
    return T.Compose([T.Grayscale(),
                      T.Resize((height, width)),
                      T.ToTensor(),
                      T.Normalize((0.5,), (0.5,))])


class Data(Dataset):
    """
    @brief Custom PyTorch dataset for autonomous driving images and labels.

    This dataset loads image paths and target values from a CSV file,
    applies optional data augmentation, normalizes steering and velocity using their 
    max values from the CSV, and returns (image_tensor, normalized_velocity, normalized_steering).

    @param dataset_path Path to the folder containing images and dataset.csv.
    @param height Preprocessing image height.
    @param width Preprocessing image width.
    @param augment Whether to apply online augmentation during training.
    """

    def __init__(self, dataset_path, height=96, width=128, augment=True):
        # Path to the dataset CSV file
        self.dataset_path = dataset_path
        self.csv_file_path = os.path.join(self.dataset_path, 'dataset.csv')
        
        self.augment = augment

        # Image augmentation pipeline (used only if augment=True)
        self.augmentation = T.Compose([
            T.ColorJitter(brightness=(0.5, 1.5),
                          contrast=(0.5, 1.5),
                          saturation=(0.5, 1.5)),
            
            T.RandomPerspective(distortion_scale=0.1, p=0.1),
            T.RandomRotation(degrees=(-3, 3)),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.8)),
        ])

        self.transform = preprocessing_transform(height, width)

        # Storage for loaded sample metadata
        self.samples = []
        self.load_samples() # Parse CSV and fill samples list


    def load_samples(self):
        """
        @brief Loads sample metadata from dataset.csv and populates self.samples.
        """
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        # Open and parse the CSV file
        with open(self.csv_file_path, 'r') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader, start=2):  # Start at line 2 (after header)

                # Skip rows that are empty or malformed
                if not any(row.values()):
                    continue

                # Build the full path to the corresponding image
                image_path = os.path.join(self.dataset_path, f"{row['image_id']}.png")

                # Append structured sample info
                self.samples.append({
                    'image_path': image_path,
                    'velocity': float(row['velocity']),
                    'steering_angle': float(row['steering_angle']),
                    'max_velocity': float(row['max_velocity']),
                    'max_steering_angle': float(row['max_steering_angle'])
                })


    def random_flip(self, image, steering, p=0.5):
        """
        @brief Horizontally flips the image and inverts steering with probability p.

        @param image Torch tensor image.
        @param steering Steering angle value.
        @param p Probability of flipping.

        @return (image, steering) Possibly flipped image and modified steering.
        """
        if np.random.rand() < p:       
            image = torch.flip(image, (2,))  # Flip along width dimension
            steering = -steering # Invert steering sign
            
        return image, steering


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        # Retrieve metadata for this index
        sample = self.samples[idx]

        # Load image from disk (RGB mode)
        image = Image.open(sample['image_path']).convert('RGB')

        # Raw velocity and steering values
        velocity = sample['velocity']
        steering = sample['steering_angle']

        # Apply augmentation if enabled (training only)
        if self.augment:
            image = self.augmentation(image)

        # Apply preprocessing (resize, grayscale, tensor, normalize)
        image = self.transform(image)

        # Apply random horizontal flip augment
        if self.augment:
            image, steering = self.random_flip(image, steering)

        # Normalize values using dataset-provided max values
        max_velocity = sample['max_velocity']
        max_steering_angle = sample['max_steering_angle']

        velocity /= max_velocity if max_velocity != 0 else 1
        steering /= max_steering_angle if max_steering_angle != 0 else 1

        # Convert to tensors
        velocity = torch.tensor(velocity, dtype=torch.float32)
        steering = torch.tensor(steering, dtype=torch.float32)

        return image, velocity, steering
