# src/stitching/warping.py

import cv2
import numpy as np

def warp_images(img1, img2, H):
    """
    Warp img1 to img2 using the homography matrix H.

    Parameters:
    - img1: First image to be warped.
    - img2: Second image (reference image).
    - H: Homography matrix.

    Returns:
    - result: The stitched panorama image.
    """
    # Get image dimensions
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Get the canvas dimensions
    corners_img1 = np.float32([[0,0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1,1,2)
    corners_img1_transformed = cv2.perspectiveTransform(corners_img1, H)

    corners_img2 = np.float32([[0,0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1,1,2)

    all_corners = np.concatenate((corners_img1_transformed, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # Warp the first image
    result = cv2.warpPerspective(img1, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    # Paste the second image into the panorama
    result[translation_dist[1]:height2+translation_dist[1], translation_dist[0]:width2+translation_dist[0]] = img2

    return result

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.feature_detection.sift_detection import detect_and_compute_sift
    from src.stitching.homography import find_homography, draw_matches

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

        # Display the panorama
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title('Stitched Panorama')
        plt.axis('off')
        plt.show()
    else:
        print("Homography could not be computed.")
