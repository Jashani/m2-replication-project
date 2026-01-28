import os
import numpy as np
from skimage import io, color, img_as_ubyte, img_as_float
from pathlib import Path

# Rotate images by 180 degrees

def rotate_image_colors_180(image):
    """
    Converts image to LAB space and rotates color channels (a, b) by 180 degrees.
    """
    # Convert to float and then to LAB space
    img_float = img_as_float(image)
    lab = color.rgb2lab(img_float)
    
    # Extract channels
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # In LAB space, a 180-degree rotation is mathematically equivalent 
    # to inverting the signs of the 'a' and 'b' color channels.
    new_a = -a_channel
    new_b = -b_channel
    
    # Merge channels and convert back to RGB
    new_lab = np.stack([l_channel, new_a, new_b], axis=2)
    new_rgb = color.lab2rgb(new_lab)
    
    # Ensure values are within valid 0.0-1.0 range and convert to 8-bit integer
    return img_as_ubyte(np.clip(new_rgb, 0, 1))

def process_directory(input_dir, output_dir):
    # Setup paths
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get valid image files
    valid_extensions = ('.png', '.jpg', '.jpeg')
    files = [f for f in in_path.iterdir() if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(files)} images...")

    for file_path in files:
        try:
            # Load image
            img = io.imread(str(file_path))
            
            # Remove alpha channel if present (RGBA -> RGB)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
                
            # Perform the 180-degree rotation
            rotated_img = rotate_image_colors_180(img)
            
            # Save to output directory
            save_path = out_path / file_path.name
            io.imsave(str(save_path), rotated_img)
            print(f"Rotated: {file_path.name}")
            
        except Exception as e:
            print(f"Could not process {file_path.name}: {e}")

if __name__ == "__main__":
    SOURCE_DIR = "../color_scrambled"
    TARGET_DIR = "../foil_scrambled"
    
    process_directory(SOURCE_DIR, TARGET_DIR)