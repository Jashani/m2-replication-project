import os
import numpy as np
from skimage import io, color, img_as_ubyte, img_as_float
import re
import random

# Rotate image colours in sets of 4, ensuring at least 30 deg between each image.
# Each set of 4 images is rotated into one of four 90 deg quarters out of 360 degrees
# to ensure significant separation.

def distinct_rotations(count, minimum, maximum, diff):
    """
    Get a set of semi-random numbers, each within a sub-range in a range.
    E.g., if we want 4 colours from a 360 colour wheel, at least 30 deg apart,
    we'll get one number between 0-90, one between 90-180, etc, while making sure
    that the ranges are capped by the 30 deg difference, so if the first number 
    is 85, the next one will be between 115-180. The numbers are then shuffled.
    """
    colors = []
    span = int((maximum - minimum) / count)
    for i in range(count):
        value = random.randrange(minimum, span * (i + 1))
        colors.append(value)
        minimum = max(minimum + span, value + diff)
        
    np.random.shuffle(colors)
    return colors

def rotate_image_colors(image, angle_degrees):
    """
    Converts image to LAB, rotates the color channels (a, b)
    around the L axis, and converts back to RGB.
    """
    # 1. Convert to float (0.0 to 1.0) and then to LAB space
    img_float = img_as_float(image)
    lab = color.rgb2lab(img_float)
    
    # 2. Extract channels
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # 3. Vectorize a and b for matrix multiplication
    # Shape becomes (2, Number of Pixels)
    ab_vectors = np.array([a_channel.flatten(), b_channel.flatten()])
    
    # 4. Create Rotation Matrix
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # 5. Apply Rotation
    rotated_vectors = rotation_matrix @ ab_vectors
    
    # 6. Reshape back to image dimensions
    height, width = l_channel.shape
    new_a = rotated_vectors[0, :].reshape(height, width)
    new_b = rotated_vectors[1, :].reshape(height, width)
    
    # 7. Merge channels and convert back to RGB
    new_lab = np.stack([l_channel, new_a, new_b], axis=2)
    new_rgb = color.lab2rgb(new_lab)
    
    # Ensure values are within valid range and convert to 8-bit integer
    return img_as_ubyte(np.clip(new_rgb, 0, 1))

def process_dir(input_dir, output_dir, foil_output_dir):
    # Create output dirs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(foil_output_dir):
        os.makedirs(foil_output_dir)

    # Get and sort all files in input dir
    natural_sort = lambda str: [int(char) if char.isdigit() else char.lower() for char in re.split(r'(\d+)', str)]
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files, key=natural_sort)

    print(f"Found {len(files)} images. Starting processing...")

    angles = []
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        
        try:
            img = io.imread(file_path)
            # Remove alpha channel if present (keep only RGB)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # New angles every 4 objects
        if len(angles) == 0:
            # 4 colours between 0-360 with at least 30 difference
            angles = distinct_rotations(4, 0, 360, 30)
        angle = angles.pop()
        rotated_img = rotate_image_colors(img, angle)
        # Flip foil image
        foil_img = rotate_image_colors(rotated_img, 180)
        
        # New file name
        name_no_ext = os.path.splitext(filename)[0]
        save_name = f"{name_no_ext}.jpg"
        save_path = os.path.join(output_dir, save_name)
        foil_save_path = os.path.join(foil_output_dir, save_name)
        
        io.imsave(save_path, rotated_img)
        io.imsave(foil_save_path, foil_img)
            
        print(f"Finished processing: {filename}")

input_dir = 'ColorRotationStimuli/TestObjectsScrambled'
output_dir = 'ColorRotationStimuli/TestScrambledColored'
foil_output_dir = 'ColorRotationStimuli/TestScrambledFoil'

if __name__ == "__main__":
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created dir '{input_dir}'. Please put images there.")
    else:
        process_dir(input_dir, output_dir, foil_output_dir)
