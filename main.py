import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Input and output directories
input_dir = r'C:\Users\pasin\Desktop\COURSES\PYTHON-B2\EXTRA\HELMET_DETECT\SAMPLE_PHOTOS'
output_dir = r'C:\Users\pasin\Desktop\COURSES\PYTHON-B2\EXTRA\HELMET_DETECT\NOT_A_HELMET'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_and_detect(image_path):
    # Load image and resize it to 600x600 pixels
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 600))

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray_image.shape[:2]

    # Process the image for pose detection
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def get_pixel_coords(landmark):
            return int(landmark.x * image_width), int(landmark.y * image_height)

        # Landmark indices for arms, legs, shoulders, waist, and ankles
        arm_indices = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
        leg_indices = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles

        # Draw light green lines for arms and legs
        light_green = (144, 238, 144)
        radius = 5
        for idx in arm_indices + leg_indices:
            if landmarks[idx].visibility > 0.5:  # Only draw visible landmarks
                coord = get_pixel_coords(landmarks[idx])
                cv2.circle(image, coord, radius, light_green, -1)

        # Get shoulder coordinates to estimate the head position
        left_shoulder = get_pixel_coords(landmarks[11])
        right_shoulder = get_pixel_coords(landmarks[12])

        # Compute midpoint between shoulders
        shoulder_midpoint = (
            (left_shoulder[0] + right_shoulder[0]) // 2,
            (left_shoulder[1] + right_shoulder[1]) // 2,
        )

        # Estimate head position based on shoulder width
        shoulder_width = np.sqrt(
            (left_shoulder[0] - right_shoulder[0]) ** 2
            + (left_shoulder[1] - right_shoulder[1]) ** 2
        )

        # Estimate head size using arm length and leg length (simplified)
        left_arm_length = np.sqrt(
            (landmarks[11].x - landmarks[15].x) ** 2 + (landmarks[11].y - landmarks[15].y) ** 2
        )
        right_arm_length = np.sqrt(
            (landmarks[12].x - landmarks[16].x) ** 2 + (landmarks[12].y - landmarks[16].y) ** 2
        )
        avg_arm_length = (left_arm_length + right_arm_length) / 2

        # Define head box dimensions (adjusting size based on estimated arm length)
        head_width = shoulder_width * 0.75
        head_height = avg_arm_length * 0.75

        # Coordinates of the head box
        top_left = (
            max(int(shoulder_midpoint[0] - head_width / 2), 0),  # Ensuring top_left stays within bounds
            max(int(shoulder_midpoint[1] - head_height * 1.5), 0)
        )
        bottom_right = (
            min(int(shoulder_midpoint[0] + head_width / 2), image_width),  # Ensuring bottom_right stays within bounds
            min(int(shoulder_midpoint[1] - head_height / 3), image_height)
        )

        # Check if the cropped head region has valid dimensions
        if bottom_right[0] > top_left[0] and bottom_right[1] > top_left[1]:
            # Draw the blue rectangle for the head on the grayscale image
            cv2.rectangle(gray_image, top_left, bottom_right, (255, 0, 0), 2)

            # Crop the head region
            head_region = gray_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Resize the cropped head region to 600x600 pixels
            cropped_resized_head = cv2.resize(head_region, (600, 600))

            # Save the cropped and resized head image
            image_filename = os.path.basename(image_path)
            cropped_file_path = os.path.join(output_dir, f"cropped_head_{image_filename}")
            cv2.imwrite(cropped_file_path, cropped_resized_head)
        else:
            print(f"Invalid head region for {image_path}, skipping head cropping.")

    else:
        print(f"No pose landmarks detected in {image_path}")

    return gray_image

def process_all_images_in_directory(input_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_directory, filename)
            print(f"Processing {image_path}...")
            processed_image = process_and_detect(image_path)
            cv2.imshow('Processed Image', processed_image)
            cv2.waitKey(500)  # Display the image for 500ms (adjust as needed)
            cv2.destroyAllWindows()

# Process all images in the input directory
process_all_images_in_directory(input_dir)

print("Processing complete.")
