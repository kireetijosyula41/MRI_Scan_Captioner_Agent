"""PyTorch implementation of the CNN stored in the legacy Keras HDF5 file."""

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
            nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 15 * 15, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
        )

    def forward(self, image, capture_layer=None):
        activation = None
        for layer in self.features:
            image = layer(image)
            if layer is capture_layer:
                activation = image

        # Keras Flatten uses NHWC order, so preserve that order for its dense weights.
        image = image.permute(0, 2, 3, 1).reshape(image.shape[0], -1)
        logits = self.classifier(image)
        if capture_layer is not None:
            return logits, activation
        return logits


def load_classifier(path):
    """Load Keras kernels into the equivalent PyTorch layers without TensorFlow."""
    model = BrainTumorCNN()
    conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    dense_layers = [layer for layer in model.classifier if isinstance(layer, nn.Linear)]

    with h5py.File(path, "r") as saved_model, torch.no_grad():
        weights = saved_model["model_weights"]
        for index, layer in enumerate(conv_layers):
            name = "conv2d" if index == 0 else f"conv2d_{index}"
            group = weights[name][name]
            kernel = torch.from_numpy(group["kernel:0"][()]).permute(3, 2, 0, 1)
            layer.weight.copy_(kernel)
            layer.bias.copy_(torch.from_numpy(group["bias:0"][()]))

        for index, layer in enumerate(dense_layers):
            name = "dense" if index == 0 else f"dense_{index}"
            group = weights[name][name]
            kernel = torch.from_numpy(group["kernel:0"][()]).T
            layer.weight.copy_(kernel)
            layer.bias.copy_(torch.from_numpy(group["bias:0"][()]))

    return model.eval()


def image_tensor(image_array):
    """Convert the existing NHWC preprocessed batch to PyTorch NCHW float32."""
    return torch.from_numpy(image_array).permute(0, 3, 1, 2).float()


def preprocess_image(image_file):
    """Apply the same RGB, 150x150, and 1/255 preprocessing as the app."""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_probabilities_batch(model, image_array):
    with torch.inference_mode():
        logits = model(image_tensor(image_array))
        return torch.softmax(logits, dim=1).cpu().numpy()


def predict_probabilities(model, image_array):
    return predict_probabilities_batch(model, image_array)[0]


def grad_cam_heatmap(model, image_array, class_index, output_size):
    """Return a normalized Grad-CAM map for the selected class and image size."""
    layers = list(model.features)
    last_conv_index = max(
        index for index, layer in enumerate(layers) if isinstance(layer, nn.Conv2d)
    )
    # The ReLU after the last convolution matches Keras Conv2D's activated output.
    target_layer = layers[last_conv_index + 1]
    logits, feature_maps = model(
        image_tensor(image_array), capture_layer=target_layer
    )
    gradients = torch.autograd.grad(logits[0, class_index], feature_maps)[0]
    channel_weights = gradients.mean(dim=(0, 2, 3), keepdim=True)
    heatmap = (feature_maps * channel_weights).sum(dim=1, keepdim=True).relu()
    heatmap = heatmap / heatmap.amax().clamp_min(1e-12)
    heatmap = F.interpolate(
        heatmap, size=output_size, mode="bilinear", align_corners=False
    )
    return heatmap[0, 0].detach().cpu().numpy()
