import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function for race detection using DeepFace with face detection
def detect_race(image_path):
    try:
        # Analyze the image to detect race and get face location
        obj = DeepFace.analyze(img_path=image_path, actions=['race'], enforce_detection=False, detector_backend='opencv')
        if isinstance(obj, list):
            obj = obj[0]
        
        # Get face region coordinates
        face_region = obj.get('region', None)
        return obj.get('dominant_race', None), obj.get('race', None), face_region
    except Exception as e:
        logging.error(f"Error in race detection: {e}")
        return None, None, None

# Function to estimate height and body type using MediaPipe
def estimate_height_and_body_type(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Get body bounding box coordinates
        landmarks = results.pose_landmarks.landmark
        x_coordinates = [landmark.x for landmark in landmarks]
        y_coordinates = [landmark.y for landmark in landmarks]
        
        # Calculate bounding box
        x_min = int(min(x_coordinates) * width)
        x_max = int(max(x_coordinates) * width)
        y_min = int(min(y_coordinates) * height)
        y_max = int(max(y_coordinates) * height)
        
        head_y = landmarks[0].y * height
        foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * height
        approx_height = abs(foot_y - head_y)
        height_category = "tall" if approx_height > 170 else "short"

        shoulder_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x -
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * width
        body_type = "slim" if shoulder_width < 0.5 * width else "wide"

        return {
            "height": height_category,
            "body_type": body_type,
            "bbox": (x_min, y_min, x_max, y_max)
        }
    else:
        return None

# Function to detect dominant color
def detect_dominant_color(image, n_clusters=1):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(np.float32(img_reshaped), n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = centers[0].astype(int)
    return dominant_color.tolist()

# Function to detect clothing based on dominant color
def simple_clothing_detection(dominant_color):
    color_labels = {
        (255, 0, 0): 'Red T-shirt',
        (0, 255, 0): 'Green Dress',
        (0, 0, 255): 'Blue Pants',
        (255, 255, 0): 'Yellow Shirt',
        (0, 255, 255): 'Cyan Jacket',
        (255, 0, 255): 'Magenta Coat',
        (0, 0, 0): 'Black Outfit',
        (255, 255, 255): 'White Outfit'
    }
    dominant_color_tuple = tuple(dominant_color)
    return color_labels.get(dominant_color_tuple, 'Unknown Clothing')

# Function to draw annotations on the image
def draw_annotations(image, face_region, body_info, race_info, clothing):
    annotated_img = image.copy()
    
    # Draw face box and race info
    if face_region:
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        race_text = f"Race: {race_info}"
        cv2.putText(annotated_img, race_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw body box and size info
    if body_info and 'bbox' in body_info:
        x_min, y_min, x_max, y_max = body_info['bbox']
        cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        body_text = f"Height: {body_info['height']}, Type: {body_info['body_type']}"
        cv2.putText(annotated_img, body_text, (x_min, y_max+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw clothing info
    clothing_text = f"Clothing: {clothing}"
    cv2.putText(annotated_img, clothing_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return annotated_img

# Main function to create profile and display visualization
def create_profile(image_path):
    logging.info("Creating profile...")
    
    # Read the image first
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Error: Could not read image")
        return None
    
    with tqdm(total=4, desc="Processing", unit="step") as pbar:
        # Detect race and face region
        race, race_details, face_region = detect_race(image_path)
        pbar.update(1)

        # Get body measurements and bounding box
        size = estimate_height_and_body_type(image_path)
        pbar.update(1)

        # Detect dominant color and clothing
        dominant_color = detect_dominant_color(image)
        clothing = simple_clothing_detection(dominant_color)
        pbar.update(1)

        # Draw annotations on the image
        annotated_image = draw_annotations(image, face_region, size, race, clothing)
        pbar.update(1)

    if race and clothing and size:
        feature_vector = {
            "race": race,
            "clothing": clothing,
            "color": dominant_color,
            "height": size['height'],
            "body_type": size['body_type']
        }
        logging.info("Profile created successfully: %s", feature_vector)
        
        # Display the annotated image
        cv2.imshow('Profile Analysis', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return feature_vector
    else:
        logging.error("Failed to create profile.")
        return None

# Example usage
if __name__ == "__main__":
    profile = create_profile('path_to_image.jpeg')