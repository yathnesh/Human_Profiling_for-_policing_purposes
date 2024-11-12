import cv2
import numpy as np
import requests
import os

# URLs for YOLO files
CONFIG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
CLASSES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# File paths
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    # Print progress
                    done = int(50 * downloaded / total_size)
                    print(f"\rProgress: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
            print("\nDownload completed!")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return False
    else:
        print(f"{filename} already exists.")
    return True

# Download necessary files
if not all([
    download_file(CONFIG_URL, CONFIG_PATH),
    download_file(WEIGHTS_URL, WEIGHTS_PATH),
    download_file(CLASSES_URL, CLASSES_PATH)
]):
    print("Failed to download one or more required files. Exiting.")
    exit(1)

# Load YOLO
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

# Load classes
with open(CLASSES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the neural network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# List of clothing-related classes we want to detect
clothing_classes = ['person', 'tie', 'backpack', 'umbrella', 'handbag', 'suitcase']

def detect_objects(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display
    class_ids = []
    confidences = []
    boxes = []

    # Showing information on the screen
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in clothing_classes:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for the box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y + 30), font, 2, color, 2)

    # Display the result
    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, img)
    print(f"Result saved as {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace this with your image path
    image_path = "D:\\pending_projects\\Machine_learning\\ML_LAB\\mini-project\\path_to_image.jpeg"
    detect_objects(image_path)