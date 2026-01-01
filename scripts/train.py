import os
import time
import argparse
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import AutopilotNet
from dataset import Data
from sampler import SteeringBalancedSampler


def setup_tensorboard(args):
    """
    @brief Initializes and configures a TensorBoard SummaryWriter.

    This function sets up a directory for TensorBoard logs and records the
    hyperparameters used for the training session.

    @param args Arguments containing hyperparameters.
    @return The configured TensorBoard writer object.
    """
    log_dir = os.path.join("runs", f"autopilot_neural_network_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    
    writer.add_text("Hyperparameters", 
                    f"learning_rate: {args.learning_rate}, "
                    f"epochs: {args.epochs}, " 
                    f"batch_size: {args.batch_size}")
    
    return writer


def save_model(model, optimizer, scheduler, epoch, path):
    """
    @brief Saves the model checkpoint.

    Saves the state of the model, optimizer, and scheduler to a file. 
    This function is used to save the best performing model based on validation loss.

    @param model The model to save.
    @param optimizer The optimizer state to save.
    @param scheduler The scheduler state to save.
    @param epoch The current epoch number.
    @param path The file path to save the checkpoint to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}
    
    torch.save(state, path)
    print(f"Model saved at epoch {epoch} to {path}")


def compute_weighted_loss(predicted_velocities, predicted_steering_angles, 
                          velocities, steering_angles, criterion, alpha=1, beta=1):
    """
    @brief Computes a weighted loss for velocity and steering angle.

    This function calculates the error for both predicted velocities
    and steering angles against their ground truth values, and then combines them
    into a single weighted loss.
    
    @param predicted_velocities The predicted velocities from the model.
    @param predicted_steering_angles The predicted steering angles from the model.
    @param velocities The ground truth velocities.
    @param steering_angles The ground truth steering angles.
    @param criterion The loss function.
    @param alpha The weight for the velocity loss component.
    @param beta The weight for the steering angle loss component.
    
    @return The final computed weighted loss.
    """
    # Compute losses separately
    velocity_loss = criterion(predicted_velocities, velocities)
    steering_loss = criterion(predicted_steering_angles, steering_angles)
    
    # Weighted total loss
    loss = (alpha * velocity_loss) + (beta * steering_loss)
    
    return loss


def validate(model, val_loader, criterion, device, alpha, beta):
    """
    @brief Evaluates the model on the validation dataset.

    This function iterates through the validation set, computes the loss, and
    calculates the mean absolute error for both velocity and steering angle predictions.

    @param model The model to be evaluated.
    @param val_loader DataLoader for the validation data.
    @param criterion The loss function used for evaluation.
    @param device The device (CPU or CUDA) to run evaluation on.
    @param alpha The weight for the velocity loss component.
    @param beta The weight for the steering angle loss component.
    
    @return A tuple containing the mean validation loss, mean velocity error,
            and mean steering angle error.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize accumulators for loss and error metrics
    val_loss = 0
    velocity_error = 0
    steering_angle_error = 0
    
    # Disable gradient calculations to save memory and computations
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Unpack data from the validation loader
            images, velocities, steering_angles = data
            
            # Reshape and cast labels to match the model's output format
            velocities = velocities.float().view(len(velocities), 1)
            steering_angles = steering_angles.float().view(len(steering_angles), 1)

            # Move data to the appropriate device (GPU or CPU)
            if device.type == "cuda":
                images = images.cuda()
                velocities = velocities.cuda()
                steering_angles = steering_angles.cuda()

            # Get model predictions
            outputs = model(images)
            
            # Extract predictions
            predicted_velocities = outputs[:, 0:1]
            predicted_steering_angles = outputs[:, 1:2]
            
            # Compute the weighted loss for the current batch
            loss = compute_weighted_loss(predicted_velocities=predicted_velocities,
                                         predicted_steering_angles=predicted_steering_angles,
                                         velocities=velocities,
                                         steering_angles=steering_angles,
                                         criterion=criterion,
                                         alpha=alpha,
                                         beta=beta)
            
            # Accumulate the validation loss
            val_loss += loss.item()
            
            # Accumulate the mean absolute error for velocity and steering angle
            velocity_error += torch.mean(
                torch.abs(velocities - predicted_velocities)).item()
            
            steering_angle_error += torch.mean(
                torch.abs(steering_angles - predicted_steering_angles)).item()
    
    # Calculate the mean validation loss and errors across all batches
    mean_val_loss = val_loss / (i + 1)
    mean_velocity_error = velocity_error / (i + 1)
    mean_steering_angle_error = steering_angle_error / (i + 1)
    
    return mean_val_loss, mean_velocity_error, mean_steering_angle_error


def train(args, model, train_dataset, val_dataset):
    """
    @brief Main training and validation loop.

    This function orchestrates the entire training process. It sets up the optimizer,
    learning rate scheduler, data loaders (including a balanced sampler for training),
    and the main training loop. For each epoch, it trains the model, evaluates it on
    the validation set, logs metrics to TensorBoard, and saves the best model
    checkpoint based on validation loss.

    @param args Command-line arguments for training configuration.
    @param model The neural network model to train.
    @param train_dataset The dataset for training.
    @param val_dataset The dataset for validation.
    """
    # Determine the device to use for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model to the selected device
    model.to(device)

    # Initialize the Adam optimizer with the model's parameters and learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize a learning rate scheduler that reduces the learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                  factor=args.lr_factor, 
                                  patience=args.lr_patience)
    
    # Define the loss function
    criterion = nn.MSELoss()
    
    # Set up a balanced sampler for the training data to handle imbalanced steering angles
    train_sampler = SteeringBalancedSampler(dataset=train_dataset,
                                            low_fraction=args.sampler_low_fraction,
                                            threshold_ratio=args.sampler_threshold_ratio,
                                            shuffle=True)
    
    print(f"Using {len(train_sampler)} training samples after balanced sampling.")
    
    # Create a DataLoader for the training set, using the balanced sampler
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              sampler=train_sampler, 
                              pin_memory=True, # Speeds up data transfer to GPU
                              num_workers=args.num_workers)
    
    # Create a DataLoader for the validation set
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.val_batch_size, 
                            shuffle=False, # No need to shuffle validation data
                            pin_memory=True, 
                            num_workers=args.num_workers)

    # Set up TensorBoard for logging
    writer = setup_tensorboard(args)
    
    # Initialize the best validation loss to infinity for tracking the best model
    best_val_loss = float('inf')
    best_model_path = args.model_path
    
    # Start the training loop for the specified number of epochs
    for epoch in range(args.epochs):
        # Set the model to training mode
        model.train()
        
        # Initialize accumulators for training metrics for the current epoch
        train_loss = 0
        velocity_error = 0
        steering_angle_error = 0
    
        # Iterate over the training data in batches
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Unpack data from the training loader
            images, velocities, steering_angles = data
            
            # Reshape and cast labels to match the model's output format
            velocities = velocities.float().view(len(velocities), 1)
            steering_angles = steering_angles.float().view(len(steering_angles), 1)

            # Move data to the appropriate device (GPU or CPU)
            if device.type == "cuda":
                images = images.cuda()
                velocities = velocities.cuda()
                steering_angles = steering_angles.cuda()
            
            # Get model predictions
            outputs = model(images)
            
            # Extract predictions
            predicted_velocities = outputs[:, 0:1]
            predicted_steering_angles = outputs[:, 1:2]
            
            # Compute the weighted loss for the current batch
            loss = compute_weighted_loss(predicted_velocities=predicted_velocities,
                                         predicted_steering_angles=predicted_steering_angles,
                                         velocities=velocities,
                                         steering_angles=steering_angles,
                                         criterion=criterion,
                                         alpha=args.alpha,
                                         beta=args.beta)
            
            # Backpropagate the loss and update model weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate the training loss and error metrics
            train_loss += loss.item()
            velocity_error += torch.mean(
                torch.abs(velocities - predicted_velocities)).item()
            
            steering_angle_error += torch.mean(
                torch.abs(steering_angles - predicted_steering_angles)).item()
            
        # Calculate the mean training loss and errors for the epoch
        mean_train_loss = train_loss / (i + 1)
        mean_velocity_error = velocity_error / (i + 1)
        mean_steering_angle_error = steering_angle_error / (i + 1)
        
        # Evaluate the model on the validation set
        mean_val_loss, mean_val_velocity_error, mean_val_steering_angle_error = \
            validate(model, val_loader, criterion, device, args.alpha, args.beta)

        print(f"Epoch {epoch} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", mean_train_loss, epoch)
        writer.add_scalar("Loss/Validation", mean_val_loss, epoch)
        writer.add_scalar("Velocity Error/Train", mean_velocity_error, epoch)
        writer.add_scalar("Steering Angle Error/Train", mean_steering_angle_error, epoch)
        writer.add_scalar("Velocity Error/Validation", mean_val_velocity_error, epoch)
        writer.add_scalar("Steering Angle Error/Validation", mean_val_steering_angle_error, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # Update the learning rate based on the validation loss
        scheduler.step(mean_val_loss)

        # Save the model if the validation loss has improved
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            save_model(model, optimizer, scheduler, epoch, best_model_path)

    writer.close()


def main(args):
    """
    @brief Entry point for the training script.

    Parses command-line arguments, initializes the model, creates the datasets,
    splits them into training and validation sets, and then starts the training
    process by calling the train function.

    @param args The parsed command-line arguments.
    """
    # Initialize model
    model = AutopilotNet(h=args.height, w=args.width, inputs=1)  
    
    # Training dataset with augmentation
    train_dataset = Data(args.dataset_path, 
                         width=args.width, 
                         height=args.height, 
                         augment=True)  
    
    # Validation dataset without augmentation
    val_dataset = Data(args.dataset_path,
                       width=args.width, 
                       height=args.height, 
                       augment=False)
    
    # Split the dataset between training and validation
    val_fraction = args.val_fraction # Validation percentage
    val_size = int(len(train_dataset) * val_fraction)  # Number of validation samples
    train_size = len(train_dataset) - val_size  # Number of training samples

    # Generate random train/val indices
    train_indices, val_indices = random_split(range(len(train_dataset)),
                                              [train_size, val_size])
    
    # Apply the generated train and validation indices to create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)  
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Start training
    train(args, model, train_dataset, val_dataset)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_path', type=str, default="/tmp/autopilot_neural_network/dataset",
                        help="Path to the directory containing the dataset.")

    parser.add_argument('--model_path', type=str, default="/tmp/autopilot_neural_network/model.pt",
                        help="Path to save the best model checkpoint.")
    
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs.")
    
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Batch size for the training data loader.")
    
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help="Batch size for the validation data loader.")
    
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help="Fraction of the dataset to use for validation (e.g., 0.2 for 20%).")

    parser.add_argument('--learning_rate', type=float, default=1.0e-3)
    
    parser.add_argument('--lr_patience', type=int, default=2, 
                        help="Epochs to wait for loss improvement before reducing learning rate.")
    
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help="Factor by which to reduce learning rate on plateau.")
    
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="Weight for the velocity component of the loss function.")
    
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Weight for the steering angle component of the loss function.")
    
    parser.add_argument('--num_workers', type=int, default=(multiprocessing.cpu_count() - 1),
                        help="Number of worker processes for data loading.")
    
    parser.add_argument('--height', type=int, default=96,
                        help="Height to which input images will be resized.")
    
    parser.add_argument('--width', type=int, default=128,
                        help="Width to which input images will be resized.")

    parser.add_argument('--sampler_low_fraction', type=float, default=0.05, 
                        help="Fraction of low-steering samples to keep.")
    
    parser.add_argument('--sampler_threshold_ratio', type=float, default=0.10,
                        help="Steering fraction of max steering to define low steering region.")

    args = parser.parse_args()

    main(args)