import torch
import torch.nn as nn


class AutopilotNet(nn.Module):
    """
    This model takes a single-channel image (e.g., grayscale) as input and
    outputs two values: the predicted velocity and steering angle. It uses a
    multi-head architecture where a shared convolutional base extracts features
    from the input image, and two separate fully connected heads predict the
    final outputs.
    """
    def __init__(self, h=96, w=128, inputs=1):
        """
        @brief Initializes the AutopilotNet model layers.

        @param h The height of the input images.
        @param w The width of the input images.
        @param inputs The number of channels in the input images.
        """
        super(AutopilotNet, self).__init__()

        # Convolutional base for feature extraction
        self.convolutional_layers = nn.Sequential(nn.Conv2d(inputs, 6, kernel_size=5, stride=1),
                                                  nn.ReLU(),
                                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                                  nn.Conv2d(6, 16, kernel_size=5, stride=1),
                                                  nn.ReLU(),
                                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                                  nn.Dropout(0.5))

        # Get the number of features produced by the convolutional block
        conv_output_size = self._get_conv_output_size(h, w, inputs)

        # Shared fully connected layers to process features from the convolutional base
        self.shared_layers = nn.Sequential(nn.Linear(conv_output_size, 120),
                                           nn.ReLU())

        # Head for predicting velocity
        self.velocity_head = nn.Sequential(nn.Linear(120, 84),
                                           nn.ReLU(),
                                           nn.Linear(84, 1))
        
        # Head for predicting steering angle
        self.steering_head = nn.Sequential(nn.Linear(120, 84),
                                           nn.ReLU(),
                                           nn.Linear(84, 1))

    def _get_conv_output_size(self, h, w, inputs):
        """
        @brief Computes the number of output features produced by the convolutional layers.

        This method generates a dummy tensor with the same shape as the actual
        input images and feeds it through the convolutional base. By examining
        the resulting tensor size, it determines the exact number of features
        required by the first fully connected layer.

        @param h The input image height.
        @param w The input image width.
        @param inputs The number of input channels.
        @return The flattened feature size after all convolutional layers.
        """
        # Create a dummy tensor
        x = torch.zeros(1, inputs, h, w)
        
        # Pass dummy input through the convolutional layers
        x = self.convolutional_layers(x)
        
        # Get the total number of elements in x
        return x.numel()
    
    def forward(self, x):
        """
        @brief Defines the forward pass of the AutopilotNet.

        @param x The input tensor, which is a batch of images.
        @return A tensor containing the concatenated predictions for velocity and steering angle.
        """
        # Pass input through the convolutional layers
        x = self.convolutional_layers(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass flattened features through the shared layers
        x = self.shared_layers(x)

        # Get predictions from each head
        vel = self.velocity_head(x)
        steer = self.steering_head(x)

        # Concatenate the outputs into a single tensor
        return torch.cat([vel, steer], dim=1)