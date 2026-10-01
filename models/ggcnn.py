import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GGCNN(nn.Module):
    """
    Generative Grasping Convolutional Neural Network (GG-CNN).
    Maps depth/RGB input to pixel-wise grasp quality Q, orientation angle theta,
    and gripper width W.
    """
    def __init__(self, input_channels=1):
        super().__init__()
        self.input_channels = input_channels

        filter_sizes = [32, 16, 8, 8, 16, 32]
        kernel_sizes = [9, 5, 3, 3, 5, 9]
        strides = [3, 2, 2, 2, 2, 3]

        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

    def forward(self, x):
        # Forward through encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Forward through decoder
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        # Prediction heads
        pos_output = torch.sigmoid(self.pos_output(x))
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = torch.relu(self.width_output(x))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        total_loss = p_loss + cos_loss + sin_loss + width_loss
        return {
            'loss': total_loss,
            'losses': {
                'p_loss': p_loss.item(),
                'cos_loss': cos_loss.item(),
                'sin_loss': sin_loss.item(),
                'width_loss': width_loss.item()
            }
        }

def load_trained_model(checkpoint_path=None, input_channels=1, device='cpu'):
    """
    Factory to instantiate GGCNN and load weights from checkpoint.
    Supports 1-channel (Depth) or 3-channel (RGB) by adapting first conv layer.
    """
    model = GGCNN(input_channels=input_channels)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # If model expects 3 channels but checkpoint has 1 channel (or vice-versa)
        conv1_w = state_dict.get('conv1.weight')
        if conv1_w is not None and conv1_w.shape[1] != input_channels:
            if input_channels == 3 and conv1_w.shape[1] == 1:
                # Replicate 1-channel weights across 3 RGB channels (divided by 3)
                state_dict['conv1.weight'] = conv1_w.repeat(1, 3, 1, 1) / 3.0
            elif input_channels == 1 and conv1_w.shape[1] == 3:
                state_dict['conv1.weight'] = conv1_w.mean(dim=1, keepdim=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from: {checkpoint_path}")
    model.to(device)
    model.eval()
    return model
