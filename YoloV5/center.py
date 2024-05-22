import cv2
import numpy as np

def draw_center_on_photo(photo_path, output_path):
    # Load the image
    image = cv2.imread(photo_path)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate center coordinates
    center_x = int(width / 2)
    center_y = int(height / 2)
    
    # Define the size of the center mark
    mark_size = 20
    
    # Draw the center mark (crosshair)
    cv2.line(image, (center_x - mark_size, center_y), (center_x + mark_size, center_y), (0, 255, 0), 2)
    cv2.line(image, (center_x, center_y - mark_size), (center_x, center_y + mark_size), (0, 255, 0), 2)
    
    # Save the image with the center mark
    cv2.imwrite(output_path, image)

# Example usage:
input_photo_path = "input_photo.jpg"
output_photo_path = "output_photo_with_center_mark.jpg"
draw_center_on_photo(input_photo_path, output_photo_path)
