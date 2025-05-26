import cv2
import numpy as np

def enhance_frame_for_detection(frame, enhancement_level='medium'):
    """
    Enhance frame contrast and sharpness to improve vehicle detection,
    especially for distant objects.
    
    Args:
        frame: Input RGB frame (numpy array)
        enhancement_level: 'light', 'medium', 'strong' - controls intensity
    
    Returns:
        Enhanced RGB frame
    """
    
    # Define enhancement parameters based on level
    params = {
        'light': {
            'clahe_clip': 2.0,
            'clahe_grid': (8, 8),
            'sharpen_alpha': 0.3,
            'gamma': 1.1
        },
        'medium': {
            'clahe_clip': 3.0,
            'clahe_grid': (8, 8),
            'sharpen_alpha': 0.5,
            'gamma': 1.2
        },
        'strong': {
            'clahe_clip': 4.0,
            'clahe_grid': (6, 6),
            'sharpen_alpha': 0.7,
            'gamma': 1.3
        }
    }
    
    p = params.get(enhancement_level, params['medium'])
    
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=p['clahe_clip'], tileGridSize=p['clahe_grid'])
    l_channel_enhanced = clahe.apply(l_channel)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
    frame_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply gamma correction for brightness adjustment
    gamma_corrected = adjust_gamma(frame_contrast, p['gamma'])
    
    # Apply unsharp masking for sharpening
    frame_sharp = unsharp_mask(gamma_corrected, p['sharpen_alpha'])
    
    # Optional: Enhance edges specifically
    frame_final = enhance_edges(frame_sharp, intensity=0.3)
    
    return cv2.cvtColor(frame_final, cv2.COLOR_RGB2BGR)

def adjust_gamma(image, gamma=1.0):
    """Apply gamma correction to adjust brightness"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def unsharp_mask(image, alpha=0.5, blur_radius=1):
    """Apply unsharp masking for sharpening"""
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), blur_radius)
    
    # Create sharpened image
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    
    return sharpened

def enhance_edges(image, intensity=0.3):
    """Enhance edges to make object boundaries clearer"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Convert back to 3-channel
    laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    
    # Blend with original image
    enhanced = cv2.addWeighted(image, 1.0, laplacian_3ch, intensity, 0)
    
    return enhanced