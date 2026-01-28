import cv2
import numpy as np
from pathlib import Path

# Scramble images based on Stojanoski, B., & Cusack, R. (2014). 
# Time to wave good-bye to phase scrambling: Creating controlled scrambled images using
# diffeomorphic transformations. Journal of Vision, 14(12), 6. doi:10.1167/14.12.6

def get_diffeo(image_size, max_distortion, nsteps):
    """Calculates the mathematical 'warp' field for scrambling."""
    ncomp = 6
    yi, xi = np.meshgrid(np.arange(image_size), np.arange(image_size))
    ph = np.random.rand(ncomp, ncomp, 4) * 2 * np.pi
    a = np.random.rand(ncomp, ncomp) * 2 * np.pi
    xn = np.zeros((image_size, image_size))
    yn = np.zeros((image_size, image_size))
    
    for xc in range(1, ncomp + 1):
        for yc in range(1, ncomp + 1):
            xn += a[xc-1, yc-1] * np.cos(xc * xi / image_size * 2 * np.pi + ph[xc-1, yc-1, 0]) * \
                                  np.cos(yc * yi / image_size * 2 * np.pi + ph[xc-1, yc-1, 1])
            yn += a[xc-1, yc-1] * np.cos(xc * xi / image_size * 2 * np.pi + ph[xc-1, yc-1, 2]) * \
                                  np.cos(yc * yi / image_size * 2 * np.pi + ph[xc-1, yc-1, 3])

    # Center the warp field to prevent the object from drifting off-screen
    xn -= np.mean(xn)
    yn -= np.mean(yn)
    
    # Normalize to RMS
    xn /= np.sqrt(np.mean(xn**2))
    yn /= np.sqrt(np.mean(yn**2))
    
    return (max_distortion * xn / nsteps).astype(np.float32), (max_distortion * yn / nsteps).astype(np.float32)

def run_scrambling(input_dir, output_dir):
    max_distortion = 20
    n_steps = 20  
    image_size = 400   
    
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    yi, xi = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

    for img_file in in_path.glob("*.jpg"):
        img = cv2.imread(str(img_file))
        if img is None: continue
        
        # Fit object within a 300px zone (75% of canvas) to provide a warp buffer
        buffer_size = 300
        h, w = img.shape[:2]
        scale = min(buffer_size / h, buffer_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Start with a white canvas and center the object
        padded = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        y_start = (image_size - new_h) // 2
        x_start = (image_size - new_w) // 2
        padded[y_start:y_start+new_h, x_start:x_start+new_w] = img_resized
        
        cx, cy = get_diffeo(image_size, max_distortion, n_steps)
        
        current_img = padded.copy()
        for step in range(n_steps):
            map_x = (xi + cx).astype(np.float32)
            map_y = (yi + cy).astype(np.float32)
            
            # Use BORDER_REFLECT to wrap object pixels back in if they hit the edge
            current_img = cv2.remap(current_img, map_x, map_y, cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REFLECT)
            
        out_name = f"{img_file.stem}.jpg"
        cv2.imwrite(str(out_path / out_name), current_img)

SOURCE_DIR = "ColorRotationStimuli/TestObjects"
TARGET_DIR = "ColorRotationStimuli/TestObjectsScrambled"

if __name__ == "__main__":
    run_scrambling(SOURCE_DIR, TARGET_DIR)