# src/stitching/graph_cut.py

import cv2
import numpy as np

def get_overlap_mask(panorama, img2):
    """
    Compute the overlap mask between two images.

    Parameters:
    - panorama: The current panorama image.
    - img2: The second image to be blended.

    Returns:
    - overlap_mask: Binary mask of the overlapping region.
    """
    # Create masks of where each image has content
    mask1 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0

    # Overlap where both masks are True
    overlap_mask = np.uint8(np.logical_and(mask1, mask2)) * 255
    return overlap_mask

def graph_cut_blending(panorama, img2, overlap_mask):
    """
    Blend two images using Graph Cut optimization within the overlap region.

    Parameters:
    - panorama: The current panorama image.
    - img2: The second image to be blended.
    - overlap_mask: Binary mask indicating the overlapping area.

    Returns:
    - blended_image: The blended panorama image.
    """
    # Convert images to float
    panorama_float = panorama.astype(float)
    img2_float = img2.astype(float)

    # Initialize blended image
    blended_image = np.copy(panorama_float)

    # Find where the overlap is
    overlap_indices = np.where(overlap_mask == 255)

    for y, x in zip(*overlap_indices):
        # Define neighborhood
        neighborhood_size = 5
        half_size = neighborhood_size // 2

        # Extract patches
        y_start = max(y - half_size, 0)
        y_end = min(y + half_size + 1, panorama.shape[0])
        x_start = max(x - half_size, 0)
        x_end = min(x + half_size + 1, panorama.shape[1])

        patch1 = panorama_float[y_start:y_end, x_start:x_end]
        patch2 = img2_float[y_start:y_end, x_start:x_end]

        # Compute difference
        difference = np.sum((patch1 - patch2) ** 2)

        # Simple decision: choose pixel with lower difference
        if difference < 1000:  # Threshold can be adjusted
            blended_image[y, x] = img2_float[y, x]

    blended_image = blended_image.astype(np.uint8)
    return blended_image

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.feature_detection.sift_detection import detect_and_compute_sift
    from src.stitching.homography import find_homography, draw_matches
    from src.stitching.warping import warp_images

    # Load two images
    img1 = cv2.imread('datasets/yaseen_panorama/image1.jpg')
    img2 = cv2.imread('datasets/yaseen_panorama/image2.jpg')

    # Detect SIFT features
    keypoints1, descriptors1 = detect_and_compute_sift(img1, n_features=500)
    keypoints2, descriptors2 = detect_and_compute_sift(img2, n_features=500)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Total matches found: {len(matches)}")

    # Compute homography
    try:
        H, status = find_homography(keypoints1, keypoints2, matches, reproj_thresh=5.0)
        print(f"Homography matrix:\n{H}")
    except ValueError as e:
        print(e)
        H, status = None, None

    if H is not None:
        # Warp images
        panorama = warp_images(img1, img2, H)

        # Compute overlap mask
        overlap_mask = get_overlap_mask(panorama, img2)

        # Apply Graph Cut Blending
        blended_panorama = graph_cut_blending(panorama, img2, overlap_mask)

        # Display the blended panorama
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(blended_panorama, cv2.COLOR_BGR2RGB))
        plt.title('Blended Panorama with Graph Cut')
        plt.axis('off')
        plt.show()
    else:
        print("Homography could not be computed.")
