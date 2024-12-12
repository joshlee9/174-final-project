import cv2
import numpy as np

def detect_and_compute_surf(image, hessian_threshold=400, n_octaves=4, n_octave_layers=3):
    """
    Detect keypoints and compute SURF descriptors for an image.

    Parameters:
    - image: Input image (grayscale or color).
    - hessian_threshold: Threshold for the Hessian keypoint detector.
    - n_octaves: Number of pyramid octaves the detector uses.
    - n_octave_layers: Number of octave layers within each octave.

    Returns:
    - keypoints: Detected keypoints.
    - descriptors: SURF descriptors.
    """
    if image is None:
        raise ValueError("Input image is None")

    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create(
        hessianThreshold=hessian_threshold,
        nOctaves=n_octaves,
        nOctaveLayers=n_octave_layers
    )

    # Detect keypoints and compute descriptors
    keypoints, descriptors = surf.detectAndCompute(gray, None)

    return keypoints, descriptors

def draw_keypoints(image, keypoints):
    """
    Draw keypoints on the image.

    Parameters:
    - image: Input image (grayscale or color).
    - keypoints: Detected keypoints.

    Returns:
    - image_with_keypoints: Image with keypoints drawn.
    """
    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return image_with_keypoints

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    img = cv2.imread('datasets/yaseen_panorama/sample.jpg')

    # Detect SURF features
    keypoints, descriptors = detect_and_compute_surf(img, hessian_threshold=500)

    print(f"Number of SURF keypoints detected: {len(keypoints)}")

    # Draw keypoints on the image
    img_with_keypoints = draw_keypoints(img, keypoints)

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('SURF Keypoints')
    plt.axis('off')
    plt.show()
