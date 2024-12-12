# src/main.py

import os
import cv2
import numpy as np
import logging
import time
import gc
from typing import List, Tuple
from pathlib import Path
import psutil  # To monitor memory usage

def setup_logging(log_file='results/logs/main.log'):
    """
    Sets up logging configuration.
    Logs will be saved to log_file and output to the console.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite the log file each time
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def apply_gaussian_filter(image: np.ndarray, kernel_size=(3,3), sigma=1.0) -> np.ndarray:
    """
    Apply Gaussian Blur to the image.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_bilateral_filter(image: np.ndarray, d=9, sigmaColor=75, sigmaSpace=75) -> np.ndarray:
    """
    Apply Bilateral Filter to the image.
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def apply_median_filter(image: np.ndarray, kernel_size=3) -> np.ndarray:
    """
    Apply Median Filter to the image.
    """
    return cv2.medianBlur(image, kernel_size)

def apply_non_local_means_denoising(image: np.ndarray, h=10, templateWindowSize=7, searchWindowSize=21) -> np.ndarray:
    """
    Apply Non-Local Means Denoising to the image.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)

def apply_laplacian_filter(image: np.ndarray, kernel_size=3) -> np.ndarray:
    """
    Apply Laplacian Filter to the image.
    """
    return cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)

def apply_sobel_filter(image: np.ndarray, ksize=3) -> np.ndarray:
    """
    Apply Sobel Filter to the image.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    # Normalize to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)
    return magnitude

def apply_filters(image: np.ndarray, filters: List[str]) -> np.ndarray:
    """
    Apply a list of filters to the image in sequence.
    """
    for filt in filters:
        if filt == 'gaussian':
            image = apply_gaussian_filter(image)
        elif filt == 'bilateral':
            image = apply_bilateral_filter(image)
        elif filt == 'median':
            image = apply_median_filter(image)
        elif filt == 'non_local_means':
            image = apply_non_local_means_denoising(image)
        elif filt == 'laplacian':
            image = apply_laplacian_filter(image)
        elif filt == 'sobel':
            image = apply_sobel_filter(image)
        else:
            logging.warning(f"Unknown filter: {filt}. Skipping.")
    return image

def resize_image(image: np.ndarray, max_width=800, max_height=600) -> np.ndarray:
    """
    Resize image to fit within max dimensions while maintaining aspect ratio.
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height, 1.0)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def detect_and_compute_features(image: np.ndarray, feature_detector=cv2.SIFT_create(nfeatures=300)) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect and compute features using SIFT.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = feature_detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
    """
    Match features between two sets of descriptors using BFMatcher.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def find_homography(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch], reproj_thresh=5.0) -> Tuple[np.ndarray, List[int]]:
    """
    Compute homography using RANSAC.
    """
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    return H, status

def stitch_images_manual(images: List[np.ndarray], filters: List[str]) -> Tuple[np.ndarray, dict]:
    """
    Manually stitch images by applying filters, detecting features, matching, and warping.
    Returns the stitched panorama and metrics.
    """
    if len(images) < 2:
        logging.warning("Need at least two images to stitch a panorama.")
        return None, {}

    feature_detector = cv2.SIFT_create(nfeatures=300)  # Further limit the number of features
    metrics = {
        'Total_Feature_Matches': 0,
        'Average_Match_Distance': 0.0,
        'Average_Reprojection_Error': 0.0
    }

    # Apply filters to all images
    filtered_images = []
    for idx, img in enumerate(images):
        filtered = apply_filters(img, filters)
        filtered_images.append(filtered)
        del img  # Free original image memory

    # Initialize panorama with the first image
    panorama = filtered_images[0]
    del filtered_images[0]  # Free memory
    for i, img in enumerate(filtered_images, start=1):
        img1 = panorama
        img2 = img

        # Detect features
        kp1, desc1 = detect_and_compute_features(img1, feature_detector)
        kp2, desc2 = detect_and_compute_features(img2, feature_detector)

        if desc1 is None or desc2 is None:
            logging.warning(f"Descriptors not found for image pair {i-1} and {i}. Skipping.")
            continue

        # Match features
        matches = match_features(desc1, desc2)
        metrics['Total_Feature_Matches'] += len(matches)
        if len(matches) > 0:
            avg_distance = np.mean([m.distance for m in matches])
            metrics['Average_Match_Distance'] += avg_distance
        else:
            avg_distance = 0.0
            metrics['Average_Match_Distance'] += avg_distance

        # Find homography
        H, status = find_homography(kp1, kp2, matches)
        if H is not None:
            # Compute reprojection error
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts_estimated = cv2.perspectiveTransform(src_pts, H)
            errors = cv2.norm(dst_pts, dst_pts_estimated, cv2.NORM_L2) / len(matches) if len(matches) > 0 else 0.0
            metrics['Average_Reprojection_Error'] += errors

            # Warp image
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            # Get the canvas dimensions
            corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
            warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
            all_corners = np.concatenate((
                np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2),
                warped_corners_img2
            ), axis=0)
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            translation_dist = [-xmin, -ymin]
            H_translation = np.array([
                [1, 0, translation_dist[0]],
                [0, 1, translation_dist[1]],
                [0, 0, 1]
            ])

            # Warp the second image
            panorama_warped = cv2.warpPerspective(img2, H_translation.dot(H), (xmax - xmin, ymax - ymin))
            # Place the first image on the canvas
            panorama_translated = np.zeros_like(panorama_warped)
            panorama_translated[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = img1

            # Create masks for blending
            mask_pano = cv2.cvtColor(panorama_translated, cv2.COLOR_BGR2GRAY)
            _, mask_pano = cv2.threshold(mask_pano, 0, 255, cv2.THRESH_BINARY)
            mask_warped = cv2.cvtColor(panorama_warped, cv2.COLOR_BGR2GRAY)
            _, mask_warped = cv2.threshold(mask_warped, 0, 255, cv2.THRESH_BINARY)
            overlap = cv2.bitwise_and(mask_pano, mask_warped)
            non_overlap_pano = cv2.bitwise_and(panorama_translated, panorama_translated, mask=cv2.bitwise_not(overlap))
            non_overlap_warped = cv2.bitwise_and(panorama_warped, panorama_warped, mask=cv2.bitwise_not(overlap))
            overlap_pano = cv2.bitwise_and(panorama_translated, panorama_translated, mask=overlap)
            overlap_warped = cv2.bitwise_and(panorama_warped, panorama_warped, mask=overlap)

            # Simple blending by averaging
            blended_overlap = cv2.addWeighted(overlap_pano, 0.5, overlap_warped, 0.5, 0)

            # Combine all parts
            panorama = cv2.add(non_overlap_pano, non_overlap_warped)
            panorama = cv2.add(panorama, blended_overlap)
        else:
            logging.error(f"Homography could not be computed for image pair {i-1} and {i}. Skipping stitching for this pair.")
            continue

        # Free memory for img2
        del img2
        del H, status, matches, kp1, kp2, desc1, desc2
        gc.collect()

    # Compute average metrics
    if len(images) > 1:
        metrics['Average_Match_Distance'] /= (len(images) - 1)
        metrics['Average_Reprojection_Error'] /= (len(images) - 1)
    else:
        metrics['Average_Match_Distance'] = 0.0
        metrics['Average_Reprojection_Error'] = 0.0

    return panorama, metrics

def process_group(group_name: str, image_paths: List[str], output_dirs: dict, filter_configs: dict):
    """
    Process a single group of images with different filter configurations.
    """
    logging.info(f"Processing group: {group_name} with {len(image_paths)} images.")

    # Load and resize images as NumPy arrays
    loaded_images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            logging.warning(f"Unable to read image: {path}. Skipping.")
            continue
        # Resize image to reduce memory usage and processing time
        resized_image = resize_image(image)
        loaded_images.append(resized_image)
        del image  # Free original image memory

    if not loaded_images:
        logging.error(f"No valid images were loaded for group: {group_name}. Skipping group.")
        return

    for config_name, filters in filter_configs.items():
        try:
            logging.info(f"Applying filter configuration: {config_name}")
            start_time = time.time()

            # Stitch images and get metrics
            panorama, metrics = stitch_images_manual(images=loaded_images, filters=filters)

            if panorama is not None:
                # Save panorama
                output_path = os.path.join(output_dirs[config_name], f"{group_name}-stitched.jpg")
                cv2.imwrite(output_path, panorama)
                logging.info(f"Saved stitched panorama to: {output_path}")

                # Log metrics
                logging.info(f"Metrics for '{group_name}-stitched.jpg' under '{config_name}':")
                logging.info(f"  Total Feature Matches: {metrics['Total_Feature_Matches']}")
                logging.info(f"  Average Match Distance: {metrics['Average_Match_Distance']:.2f}")
                logging.info(f"  Average Reprojection Error: {metrics['Average_Reprojection_Error']:.2f}")
            else:
                logging.error(f"Stitching failed for group {group_name} under configuration {config_name}.")
        except MemoryError:
            logging.error(f"MemoryError encountered while processing group {group_name} under configuration {config_name}. Skipping this configuration.")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing group {group_name} under configuration {config_name}: {e}")
            continue

        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"  Processing Time: {processing_time:.2f} seconds")

        # Monitor and log memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
        logging.info(f"  Current Memory Usage: {memory_usage:.2f} MB")

    # Cleanup after processing the group
    del loaded_images
    gc.collect()
    logging.info(f"Completed processing group: {group_name}")

def main():
    """
    Main function to process all image groups with different filter configurations.
    """
    setup_logging()
    logging.info("Starting image stitching pipeline.")

    input_dir = 'datasets/input-42-data'
    output_base_dir = 'results/panoramas'
    common_output_dir = os.path.join(output_base_dir, 'common')
    less_common_output_dir = os.path.join(output_base_dir, 'less_common')
    all_filters_output_dir = os.path.join(output_base_dir, 'all_filters')

    # Create output directories
    os.makedirs(common_output_dir, exist_ok=True)
    os.makedirs(less_common_output_dir, exist_ok=True)
    os.makedirs(all_filters_output_dir, exist_ok=True)

    output_dirs = {
        'common': common_output_dir,
        'less_common': less_common_output_dir,
        'all_filters': all_filters_output_dir
    }

    # Define filter configurations
    filter_configs = {
        'common': ['gaussian', 'bilateral'],
        'less_common': ['median', 'non_local_means', 'laplacian', 'sobel'],
        'all_filters': ['gaussian', 'bilateral', 'median', 'non_local_means', 'laplacian', 'sobel']
    }

    # Iterate through each group
    groups = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not groups:
        logging.error(f"No image groups found in '{input_dir}'. Exiting.")
        return

    for group in groups:
        group_path = os.path.join(input_dir, group)
        # Collect image files (assuming common image formats)
        image_files = sorted([
            os.path.join(group_path, f) for f in os.listdir(group_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not image_files:
            logging.warning(f"No images found in group '{group}'. Skipping.")
            continue
        process_group(group, image_files, output_dirs, filter_configs)

    logging.info("Image stitching pipeline completed.")

if __name__ == "__main__":
    main()
