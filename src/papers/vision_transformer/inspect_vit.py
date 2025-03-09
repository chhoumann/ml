"""
Vision Transformer (ViT) Attention Visualizer
===========================================

This script visualizes attention maps from a Vision Transformer model for any given image.
It creates three visualizations:
1. The original image
2. The attention heatmap
3. A spotlight effect showing where the model is focusing

Requirements:
------------
- Python 3.6+
- Dependencies:
  - matplotlib
  - numpy
  - requests
  - Pillow
  - transformers
  
Installation:
------------
pip install matplotlib numpy requests Pillow transformers

Usage:
-----
Basic usage:
    python inspect_vit.py --image-url <URL_TO_IMAGE>

Advanced usage:
    python inspect_vit.py \
        --image-url <URL_TO_IMAGE> \
        --model-name <MODEL_NAME> \
        --output <OUTPUT_PATH>

Arguments:
  --image-url    URL of the image to analyze (required)
  --model-name   Name of the ViT model to use (default: google/vit-base-patch16-224)
  --output       Path to save the visualization (default: attention_visualization.png)

Example:
-------
python inspect_vit.py \
    --image-url https://example.com/cat.jpg \
    --output cat_attention.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def load_image_from_url(url):
    """Load an image from URL."""
    return Image.open(requests.get(url, stream=True).raw)


def get_attention_maps(model, processor, image):
    """Get attention maps from the model for a given image."""
    # Prepare image
    inputs = processor(images=image, return_tensors="pt")

    # Get model outputs with attention
    outputs = model(**inputs, output_attentions=True)

    # Get attention weights from last layer (typically most interpretable)
    # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    attention = outputs.attentions[-1].detach().numpy()

    return attention[0]  # Remove batch dimension


def visualize_attention(image, attention_maps, save_path=None):
    """Visualize attention maps on the image."""
    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Get attention for CLS token to all patches
    # Average over attention heads
    cls_attention = attention_maps.mean(0)[0, 1:]  # Skip CLS token

    # Reshape attention to match image patches
    num_patches = int(np.sqrt(len(cls_attention)))
    attention_map = cls_attention.reshape(num_patches, num_patches)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Plot original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot attention heatmap
    im = ax2.imshow(attention_map, cmap="hot")
    ax2.set_title("Attention Map")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2)

    # Create spotlight visualization
    # Resize attention map to match image dimensions
    attention_map_resized = Image.fromarray(attention_map).resize(
        (image.shape[1], image.shape[0]), resample=Image.Resampling.BILINEAR
    )
    attention_map_resized = np.array(attention_map_resized)

    # Normalize attention map
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (
        attention_map_resized.max() - attention_map_resized.min()
    )

    # Apply sigmoid to increase contrast in attention map
    attention_map_resized = 1 / (1 + np.exp(-10 * (attention_map_resized - 0.5)))

    # Create spotlight effect
    darkness = 0.1  # How dark the non-attended regions should be
    spotlight = np.maximum(attention_map_resized, darkness)

    # Apply spotlight effect to each channel
    spotlight_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # RGB channels
        spotlight_image[:, :, i] = image[:, :, i] * spotlight

    # Convert back to uint8
    spotlight_image = np.clip(spotlight_image, 0, 255).astype(np.uint8)

    # Plot spotlight effect
    ax3.imshow(spotlight_image)
    ax3.set_title("Attention Spotlight")
    ax3.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Vision Transformer attention maps for an image."
    )
    parser.add_argument(
        "--image-url", type=str, required=True, help="URL of the image to analyze"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Name of the ViT model to use (default: google/vit-base-patch16-224)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_visualization.png",
        help="Path to save the visualization (default: attention_visualization.png)",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load model and processor
    print(f"Loading model: {args.model_name}")
    processor = ViTImageProcessor.from_pretrained(args.model_name)
    model = ViTForImageClassification.from_pretrained(args.model_name)

    # Load image
    print(f"Loading image from: {args.image_url}")
    image = load_image_from_url(args.image_url)

    # Get attention maps
    print("Generating attention maps...")
    attention_maps = get_attention_maps(model, processor, image)

    # Visualize attention
    print(f"Saving visualization to: {args.output}")
    visualize_attention(image, attention_maps, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
