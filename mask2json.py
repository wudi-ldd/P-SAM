import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Set the path for the image folder
image_folder = 'prompt_mask'
# Create a path for the folder where JSON files will be saved
json_folder = 'json'

# Ensure the folder for saving JSON files exists
os.makedirs(json_folder, exist_ok=True)

# Get all image file names in the image folder
image_files = os.listdir(image_folder)

for image_file in image_files:
    # Build the path for the image file
    image_path = os.path.join(image_folder, image_file)

    # Read the mask image
    mask_image = cv2.imread(image_path)

    # Convert the mask image to grayscale
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask_image_gray[mask_image_gray > 1] = 255
    # Use opening operation to remove small noise
    kernel = np.ones((5,5), np.uint8)
    opened_mask = cv2.morphologyEx(mask_image_gray, cv2.MORPH_OPEN, kernel)

    # Extract contours of pores (white areas)
    mask = opened_mask == 255
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create Labelme JSON data structure
    labelme_data = {
        "version": "LabelMe",
        "imagePath": image_file,
        "imageData": None,
        "imageHeight": mask_image.shape[0],
        "imageWidth": mask_image.shape[1],
        "flags": {},
        "shapes": []
    }

    # Create contours for each region in Labelme
    for i, contour in enumerate(contours):
        # Simplify contour
        epsilon = 0.000001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        # Check number of contour points
        if len(contour) < 15:
            continue

        contour = contour.flatten().tolist()
        shape = {
            "label": "pore",
            "points": [],
            "group_id": i + 1,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        }

        # Convert contour coordinates to [x, y] format
        for j in range(0, len(contour), 2):
            shape["points"].append([contour[j], contour[j + 1]])

        labelme_data["shapes"].append(shape)

    # Build the path for saving the JSON file
    json_file_path = os.path.join(json_folder, os.path.splitext(image_file)[0] + '.json')

    # Save Labelme data as a JSON file and format it
    with open(json_file_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)
    print("Saved to {}".format(json_file_path))