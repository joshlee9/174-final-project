# src/stitching/homography.py

import cv2
import numpy as np

def find_homography(keypoints1, keypoints2, matches, reproj_thresh=5.0):
    """
    Compute the homography matrix using matched keypoints.

    Parameters:
    - keypoints1: Keypoints from the first image.
    - keypoints2: Keypoints from the second image.
    - matches: List of matched keypoints.
    - reproj_thresh: RANSAC reprojection threshold.

    Returns:
    - H: Homography matrix.
    - status: Mask of inliers and outliers.
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")

    # Extract the matched keypoints
    ptsA = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    ptsB = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Compute the homography matrix using RANSAC
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)

    return H, status

def draw_matches(img1, keypoints1, img2, keypoints2, matches, status=None):
    """
    Draw matches between two images.

    Parameters:
    - img1: First image.
    - keypoints1: Keypoints from the first image.
    - img2: Second image.
    - keypoints2: Keypoints from the second image.
    - matches: List of matched keypoints.
    - status: Optional mask of inliers and outliers.

    Returns:
    - matched_image: Image with matches drawn.
    """
    if status is not None:
        matches = [m for i, m in enumerate(matches) if status[i]]

    matched_image = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matched_image

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.feature_detection.sift_detection import detect_and_compute_sift

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
        # Draw inlier matches
        matched_img = draw_matches(img1, keypoints1, img2, keypoints2, matches, status)

        # Display the matched image
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.title('Inlier Matches')
        plt.axis('off')
        plt.show()
    else:
        print("Homography could not be computed.")
