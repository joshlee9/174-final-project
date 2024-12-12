# src/evaluation/evaluate_stitching.py

import os
import cv2
import numpy as np
import logging
import time
from skimage import img_as_float
from skimage.restoration import estimate_sigma
from typing import List, Tuple
from pathlib import Path

def setup_logging(log_file='results/logs/evaluation.log'):
    """
    Sets up the logging configuration.
    Logs will be saved to the specified log_file and output to the console.
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

def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute the sharpness of the image using the variance of the Laplacian.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(grayscale, cv2.CV_64F).var()
    return laplacian_var

def compute_noise_level(image: np.ndarray) -> float:
    """
    Estimate the noise level in the image using skimage's estimate_sigma.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_float = img_as_float(grayscale)
    try:
        noise_level = estimate_sigma(grayscale_float, channel_axis=None, average_sigmas=True)
    except TypeError:
        # For older versions of scikit-image without 'channel_axis'
        noise_level = estimate_sigma(grayscale_float, multichannel=False, average_sigmas=True)
    return noise_level

def compute_color_statistics(image: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute mean and standard deviation for each color channel.
    Returns tuple of (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
    """
    mean_color = cv2.mean(image)[:3]  # (B, G, R)
    std_color = cv2.meanStdDev(image)[1].flatten()  # (B_std, G_std, R_std)
    mean_r, mean_g, mean_b = mean_color[2], mean_color[1], mean_color[0]
    std_r, std_g, std_b = std_color[2], std_color[1], std_color[0]
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def compute_edge_density(image: np.ndarray) -> float:
    """
    Compute the edge density of the image.
    Edge density is defined as the ratio of edge pixels to total pixels.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    edge_density = edge_pixels / total_pixels
    return edge_density

def detect_and_compute_features(image: np.ndarray, feature_detector=cv2.SIFT_create(nfeatures=500)) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
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

def compute_intrinsic_metrics(panorama_image: np.ndarray, reference_images: List[np.ndarray], feature_detector=cv2.SIFT_create(nfeatures=500)) -> Tuple[int, float, float]:
    """
    Compute intrinsic metrics: Total Feature Matches, Average Match Distance, and Average Reprojection Error.
    """
    total_matches = 0
    total_distance = 0.0
    total_reprojection_error = 0.0
    homography_count = 0

    for ref_img in reference_images:
        kp1, desc1 = detect_and_compute_features(panorama_image, feature_detector)
        kp2, desc2 = detect_and_compute_features(ref_img, feature_detector)

        if desc1 is None or desc2 is None:
            logging.warning("Descriptors not found for a pair. Skipping.")
            continue

        matches = match_features(desc1, desc2)
        total_matches += len(matches)

        if len(matches) > 0:
            avg_distance = np.mean([m.distance for m in matches])
            total_distance += avg_distance

            H, status = find_homography(kp1, kp2, matches)
            if H is not None:
                # Compute reprojection error
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts_estimated = cv2.perspectiveTransform(src_pts, H)
                errors = cv2.norm(dst_pts, dst_pts_estimated, cv2.NORM_L2) / len(matches) if len(matches) > 0 else 0.0
                total_reprojection_error += errors
                homography_count += 1
            else:
                logging.warning("Homography computation failed for a pair.")

    # Aggregate metrics
    average_match_distance = total_distance / homography_count if homography_count else 0.0
    average_reprojection_error = total_reprojection_error / homography_count if homography_count else 0.0

    return total_matches, average_match_distance, average_reprojection_error

def compute_iqa_metrics(stitched_image: np.ndarray):
    """
    Compute no-reference IQA metrics:
    - Sharpness (Variance of Laplacian)
    - Noise Level (Estimated noise in the image)
    - Color Consistency (Mean and Standard Deviation of Color Channels)
    - Edge Density (Proportion of Edge Pixels)
    """
    sharpness = compute_sharpness(stitched_image)
    noise_level = compute_noise_level(stitched_image)
    (mean_r, mean_g, mean_b), (std_r, std_g, std_b) = compute_color_statistics(stitched_image)
    edge_density = compute_edge_density(stitched_image)

    return {
        'Sharpness': round(sharpness, 2),
        'Noise_Level': round(noise_level, 4),
        'Mean_Color_R': round(mean_r, 2),
        'Mean_Color_G': round(mean_g, 2),
        'Mean_Color_B': round(mean_b, 2),
        'Std_Color_R': round(std_r, 2),
        'Std_Color_G': round(std_g, 2),
        'Std_Color_B': round(std_b, 2),
        'Edge_Density': round(edge_density, 4)
    }

def compute_intrinsic_and_iqa(panorama_path: str, reference_dir: str) -> dict:
    """
    Compute both intrinsic and IQA metrics for a stitched panorama.
    """
    panorama_image = cv2.imread(panorama_path)
    if panorama_image is None:
        logging.warning(f"Unable to read panorama image: {panorama_path}. Skipping.")
        return {}

    # Load reference images
    reference_images = [cv2.imread(os.path.join(reference_dir, f)) for f in os.listdir(reference_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    reference_images = [img for img in reference_images if img is not None]

    if not reference_images:
        logging.warning(f"No valid reference images found in '{reference_dir}'. Skipping intrinsic metrics.")
        total_feature_matches = 0
        average_match_distance = 0.0
        average_reprojection_error = 0.0
    else:
        total_feature_matches, average_match_distance, average_reprojection_error = compute_intrinsic_metrics(panorama_image, reference_images)

    # Compute IQA metrics
    iqa_metrics = compute_iqa_metrics(panorama_image)

    # Aggregate all metrics
    metrics = {
        'Panorama': os.path.basename(panorama_path),
        'Total_Feature_Matches': total_feature_matches,
        'Average_Match_Distance': round(average_match_distance, 2),
        'Average_Reprojection_Error': round(average_reprojection_error, 2),
        'Sharpness': iqa_metrics['Sharpness'],
        'Noise_Level': iqa_metrics['Noise_Level'],
        'Mean_Color_R': iqa_metrics['Mean_Color_R'],
        'Mean_Color_G': iqa_metrics['Mean_Color_G'],
        'Mean_Color_B': iqa_metrics['Mean_Color_B'],
        'Std_Color_R': iqa_metrics['Std_Color_R'],
        'Std_Color_G': iqa_metrics['Std_Color_G'],
        'Std_Color_B': iqa_metrics['Std_Color_B'],
        'Edge_Density': iqa_metrics['Edge_Density']
    }

    return metrics

def main():
    """
    Main function to evaluate stitched panoramas across all filter configurations.
    """
    setup_logging()
    logging.info("Starting evaluation of stitched panoramas.")

    output_base_dir = 'results/panoramas'
    metrics_dir = 'results/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_output_path = os.path.join(metrics_dir, 'stitching_metrics.txt')

    configurations = ['common', 'less_common', 'all_filters']
    output_dirs = {
        'common': os.path.join(output_base_dir, 'common'),
        'less_common': os.path.join(output_base_dir, 'less_common'),
        'all_filters': os.path.join(output_base_dir, 'all_filters')
    }

    # Initialize metrics storage
    all_metrics = []

    # Iterate through each configuration
    for config in configurations:
        logging.info(f"Evaluating configuration: {config}")
        pano_dir = output_dirs[config]
        if not os.path.exists(pano_dir):
            logging.warning(f"Panorama directory '{pano_dir}' does not exist. Skipping.")
            continue
        panoramas = [f for f in os.listdir(pano_dir) if f.endswith('-stitched.jpg')]
        if not panoramas:
            logging.warning(f"No stitched panoramas found in '{pano_dir}'. Skipping.")
            continue
        for pano_file in panoramas:
            pano_path = os.path.join(pano_dir, pano_file)
            logging.info(f"Computing metrics for '{pano_file}' under '{config}' configuration.")

            # Extract group_name by removing the '-stitched.jpg' suffix
            group_name = pano_file.replace('-stitched.jpg', '')
            
            # **Key Correction Below**
            # Map the full group_name directly to the input_group_dir
            # Previously, it was incorrectly splitting at the first hyphen
            input_group_dir = os.path.join('datasets/input-42-data', group_name)
            # **End of Key Correction**

            if not os.path.exists(input_group_dir):
                logging.warning(f"Input group directory '{input_group_dir}' does not exist for panorama '{pano_file}'. Skipping intrinsic metrics.")
                total_feature_matches = 0
                average_match_distance = 0.0
                average_reprojection_error = 0.0
            else:
                # Collect all images in the group as reference images
                reference_images = sorted([
                    os.path.join(input_group_dir, f) for f in os.listdir(input_group_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                reference_images = [f for f in reference_images if os.path.isfile(f)]
                if not reference_images:
                    logging.warning(f"No images found in '{input_group_dir}' for panorama '{pano_file}'. Skipping intrinsic metrics.")
                    total_feature_matches = 0
                    average_match_distance = 0.0
                    average_reprojection_error = 0.0
                else:
                    # Read all reference images
                    reference_images_data = []
                    for ref_path in reference_images:
                        ref_img = cv2.imread(ref_path)
                        if ref_img is None:
                            logging.warning(f"Unable to read reference image: {ref_path}. Skipping.")
                            continue
                        reference_images_data.append(ref_img)
                    
                    if not reference_images_data:
                        logging.warning(f"All reference images in '{input_group_dir}' could not be read. Skipping intrinsic metrics.")
                        total_feature_matches = 0
                        average_match_distance = 0.0
                        average_reprojection_error = 0.0
                    else:
                        # Compute intrinsic metrics
                        total_feature_matches, average_match_distance, average_reprojection_error = compute_intrinsic_metrics(
                            cv2.imread(pano_path), 
                            reference_images_data
                        )

            # Compute IQA metrics
            panorama_image = cv2.imread(pano_path)
            if panorama_image is not None:
                iqa_metrics = compute_iqa_metrics(panorama_image)
            else:
                logging.warning(f"Unable to read panorama image: {pano_path}. Skipping IQA metrics.")
                iqa_metrics = {}

            # Aggregate all metrics
            panorama_metrics = {
                'Panorama': pano_file,
                'Configuration': config,
                'Total_Feature_Matches': total_feature_matches,
                'Average_Match_Distance': round(average_match_distance, 2),
                'Average_Reprojection_Error': round(average_reprojection_error, 2),
                'Sharpness': iqa_metrics.get('Sharpness', 0.0),
                'Noise_Level': iqa_metrics.get('Noise_Level', 0.0),
                'Mean_Color_R': iqa_metrics.get('Mean_Color_R', 0.0),
                'Mean_Color_G': iqa_metrics.get('Mean_Color_G', 0.0),
                'Mean_Color_B': iqa_metrics.get('Mean_Color_B', 0.0),
                'Std_Color_R': iqa_metrics.get('Std_Color_R', 0.0),
                'Std_Color_G': iqa_metrics.get('Std_Color_G', 0.0),
                'Std_Color_B': iqa_metrics.get('Std_Color_B', 0.0),
                'Edge_Density': iqa_metrics.get('Edge_Density', 0.0)
            }

            all_metrics.append(panorama_metrics)

            # Log metrics
            logging.info(f"  Total Feature Matches: {panorama_metrics['Total_Feature_Matches']}")
            logging.info(f"  Average Match Distance: {panorama_metrics['Average_Match_Distance']:.2f}")
            logging.info(f"  Average Reprojection Error: {panorama_metrics['Average_Reprojection_Error']:.2f}")
            logging.info(f"  Sharpness (Variance of Laplacian): {panorama_metrics['Sharpness']:.2f}")
            logging.info(f"  Noise Level (Estimated Sigma): {panorama_metrics['Noise_Level']:.4f}")
            logging.info(f"  Mean Color (R, G, B): ({panorama_metrics['Mean_Color_R']:.2f}, {panorama_metrics['Mean_Color_G']:.2f}, {panorama_metrics['Mean_Color_B']:.2f})")
            logging.info(f"  Std Color (R, G, B): ({panorama_metrics['Std_Color_R']:.2f}, {panorama_metrics['Std_Color_G']:.2f}, {panorama_metrics['Std_Color_B']:.2f})")
            logging.info(f"  Edge Density: {panorama_metrics['Edge_Density']:.4f}")

    # Write metrics to file
    if all_metrics:
        with open(metrics_output_path, 'w') as f:
            header = "Panorama,Configuration,Total_Feature_Matches,Average_Match_Distance,Average_Reprojection_Error,Sharpness,Noise_Level,Mean_Color_R,Mean_Color_G,Mean_Color_B,Std_Color_R,Std_Color_G,Std_Color_B,Edge_Density\n"
            f.write(header)
            for m in all_metrics:
                line = (
                    f"{m['Panorama']},"
                    f"{m['Configuration']},"
                    f"{m['Total_Feature_Matches']},"
                    f"{m['Average_Match_Distance']:.2f},"
                    f"{m['Average_Reprojection_Error']:.2f},"
                    f"{m['Sharpness']:.2f},"
                    f"{m['Noise_Level']:.4f},"
                    f"{m['Mean_Color_R']:.2f},"
                    f"{m['Mean_Color_G']:.2f},"
                    f"{m['Mean_Color_B']:.2f},"
                    f"{m['Std_Color_R']:.2f},"
                    f"{m['Std_Color_G']:.2f},"
                    f"{m['Std_Color_B']:.2f},"
                    f"{m['Edge_Density']:.4f}\n"
                )
                f.write(line)
        logging.info(f"All metrics have been written to '{metrics_output_path}'.")
    else:
        logging.warning("No metrics were computed. Ensure that stitched panoramas exist and are correctly named.")

    logging.info("Evaluation of stitched panoramas completed.")

if __name__ == "__main__":
    main()
