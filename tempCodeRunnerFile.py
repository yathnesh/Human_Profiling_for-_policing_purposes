import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from tqdm import tqdm

# Function for race detection using DeepFace
def detect_race(image_path):
    try:
        # Analyze the image to detect race, set enforce_detection to False
        obj = DeepFace.analyze(img_path=image_path, actions=['race'], enforce_detection=False)
        if isinstance(obj, list):
            obj = obj[0]
        return obj.get('dominant_race', None), obj.get('race', None)
    except Exception as e:
        print(f"Error in race detection: {e}")
        return None, None

# Function to estimate height and body type using MediaPipe
def estimate_height_and_body_type(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        head_y = results.pose_landmarks.landmark[0].y * height
        foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * height
        approx_height = abs(foot_y - head_y)
        height_category = "tall" if approx_height > 170 else "short"
        
        shoulder_width = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x - 
                             results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * width
        body_type = "slim" if shoulder_width < 0.5 * width else "wide"

        return {"height": height_category, "body_type": body_type}
    else:
        return None

# A simple function to detect the dominant color
def detect_dominant_color(image):
    # Convert to RGB and reshape for K-means clustering
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape((-1, 3))

    # Perform K-means clustering
    kmeans = cv2.kmeans(np.float32(img_reshaped), 1, None, 
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                        10, cv2.KMEANS_RANDOM_CENTERS)[2]

    dominant_color = kmeans[0].astype(int)
    return dominant_color.tolist()  # Return as a list for easy processing

# Simple clothing detection based on dominant color
def simple_clothing_detection(dominant_color):
    # Convert dominant color to a string for basic classification
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

    # Default to 'Unknown' if color is not in the predefined list
    return color_labels.get(dominant_color_tuple, 'Unknown Clothing')

# Main function to create the profile
def create_profile(image_path):
    print("Creating profile...")
    with tqdm(total=3, desc="Processing", unit="step") as pbar:
        race, race_details = detect_race(image_path)
        pbar.update(1)

        size = estimate_height_and_body_type(image_path)
        pbar.update(1)

        img = cv2.imread(image_path)
        dominant_color = detect_dominant_color(img)
        clothing = simple_clothing_detection(dominant_color)
        pbar.update(1)

    if race and clothing and size:
        feature_vector = {
            "race": race,
            "clothing": clothing,
            "color": dominant_color,
            "height": size['height'],
            "body_type": size['body_type']
        }
        print("Profile created successfully:", feature_vector)
        return feature_vector
    else:
        print("Failed to create profile.")
        return None

# Example usage
if __name__ == "__main__":
    profile = create_profile('D:\\pending_projects\\Machine_learning\\ML_LAB\\Person-profiling\\path_to_image.jpeg')
