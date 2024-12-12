import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from src.evaluation.metrics import calculate_ssim

def main():
    stitched_dir = 'results/stitched/train'
    reference_dir = 'datasets/yaseen_panorama/reference_panorama/train'
    output_file = 'results/metrics/ssim_metrics.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for panorama_name in os.listdir(stitched_dir):
            stitched_path = os.path.join(stitched_dir, panorama_name)
            reference_path = os.path.join(reference_dir, panorama_name)
            
            if not os.path.exists(reference_path):
                print(f"Reference image not found for {panorama_name}. Skipping.")
                continue
            
            stitched_image = cv2.imread(stitched_path, cv2.IMREAD_GRAYSCALE)
            reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
            
            if stitched_image is None or reference_image is None:
                print(f"Error reading images for {panorama_name}. Skipping.")
                continue
            
            ssim_value = calculate_ssim(reference_image, stitched_image)
            f.write(f"{panorama_name}: SSIM = {ssim_value:.4f}\n")
            print(f"{panorama_name}: SSIM = {ssim_value:.4f}")

if __name__ == "__main__":
    main()
