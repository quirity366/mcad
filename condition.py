import os
import cv2
import numpy as np
from tqdm import tqdm

def process_image_to_three_channel(img_path = './good.png', mask_path='./mask.png', anomaly_id=3):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    LOW_THRESHOLD = 50
    HIGH_THRESHOLD = 100
    
    output_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    if mask is not None:
        _, binary_mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        masked_gray = gray_img.copy()
        masked_gray[binary_mask == 255] = 0
        
        edges = cv2.Canny(masked_gray, LOW_THRESHOLD, HIGH_THRESHOLD)
        
        binary_mask[binary_mask == 255] = 256 - anomaly_id
        
        output_image[:, :, 0] = masked_gray  
        output_image[:, :, 1] = edges        
        output_image[:, :, 2] = binary_mask  
        
    else:
        edges = cv2.Canny(gray_img, LOW_THRESHOLD, HIGH_THRESHOLD)
        output_image[:, :, 0] = gray_img  
        output_image[:, :, 1] = edges     
        output_image[:, :, 2] = 0    

    output_img_path = './condition.png'

    cv2.imwrite(output_img_path, output_image)

    return output_image

if __name__ == '__main__':
    process_image_to_three_channel()