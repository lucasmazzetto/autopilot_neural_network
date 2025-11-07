import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import Data
from model import AutopilotNet


def evaluate(args, model, test_set):
    """
    Evaluates the model on the test set, displaying predictions and ground truth.
    
    @param args Parsed command-line arguments containing model_path and debug settings.
    @param model The moswl instance to evaluate.
    @param test_set Dataset object containing test samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load model checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_workers)

    model.eval()
    total_velocity_error = 0
    total_steering_error = 0

    with torch.no_grad():
        
        # Iterate through test samples
        for i, data in enumerate(tqdm(test_loader, desc="Evaluating")):
            image, gt_velocity, gt_steering = data

            # Move tensors to the correct device
            image = image.to(device)

            # Get model predictions
            outputs = model(image)
            pred_velocity = outputs[0, 0].item()
            pred_steering = outputs[0, 1].item()

            # Extract ground truth values
            gt_velocity = gt_velocity.item()
            gt_steering = gt_steering.item()
            
            # Accumulate errors
            total_velocity_error += abs(pred_velocity - gt_velocity)
            total_steering_error += abs(pred_steering - gt_steering)

            # Optional visualization for debugging
            if args.debug:
                print('--------------------------------------------------------------')
                print(f"Predicted Velocity: {pred_velocity:.4f}, " 
                      f"Ground Truth Velocity: {gt_velocity:.4f}")
                
                print(f"Predicted Steering: {pred_steering:.4f}, " 
                      f"Ground Truth Steering: {gt_steering:.4f}")

                # Prepare image for display
                img_display = image.squeeze(0)
                img_display = (img_display * 0.5) + 0.5  # Denormalize from [-1, 1]
                img_display = img_display.cpu().numpy().transpose(1, 2, 0)
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
                img_display = cv2.resize(img_display, (640, 480))

                # Add text to image
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 255, 0)  # Green
                
                # Ground Truth Text
                cv2.putText(img_display, 
                            f"GT Vel: {gt_velocity:.3f}", 
                            (20, 30), font, 0.8, color, 2, 1)
                
                cv2.putText(img_display, 
                            f"GT Steer: {gt_steering:.3f}", 
                            (20, 60), font, 0.8, color, 2, 1)

                # Predicted Text
                cv2.putText(img_display, 
                            f"Pred Vel: {pred_velocity:.3f}", 
                            (20, 100), font, 0.8, color, 2, 1)
                cv2.putText(img_display, 
                            f"Pred Steer: {pred_steering:.3f}", 
                            (20, 130), font, 0.8, color, 2, 1)

                cv2.imshow('Evaluation', img_display)

                # Press 'q' to quit, any other key to continue
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                
    # Close visualization window if debugging mode is active
    if args.debug:
        cv2.destroyAllWindows()

    # Print averaged errors
    print(f"\nMean Velocity Error: {total_velocity_error / len(test_loader):.4f}")
    print(f"Mean Steering Error: {total_steering_error / len(test_loader):.4f}")


def main(args):
    """
    Main function to set up model and dataset, then start evaluation.
    
    @param args Parsed command-line arguments.
    """
    model = AutopilotNet(h=args.height, w=args.width, inputs=1)
    
    # Augmentation is disabled for evaluation
    test_set = Data(args.dataset_path, 
                    height=args.height, 
                    width=args.width, 
                    augment=False)

    evaluate(args, model, test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the AutopilotNet model.")
    
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset directory.')
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (.pth file).')
    
    parser.add_argument('--debug', action='store_true', 
                        help='Show image visualizations.')

    parser.add_argument('--height', type=int, default=96,
                        help="Height to which input images will be resized.")
    
    parser.add_argument('--width', type=int, default=128,
                        help="Width to which input images will be resized.")

    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of worker processes for data loading.")
    
    args = parser.parse_args()

    main(args)
