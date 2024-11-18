# Human Profiling for Policing Purposes

This project leverages advanced Machine Learning techniques and Computer Vision libraries to analyze images and create detailed human profiles. The system extracts information such as race, height, body type, clothing, and dominant color, aimed at assisting law enforcement with profiling capabilities.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Race Detection**: Determines the dominant race and a detailed breakdown using DeepFace.
- **Height and Body Type Estimation**: Uses MediaPipe Pose to estimate approximate height and classify body type.
- **Clothing Detection**: Detects the dominant color and identifies potential clothing types.
- **Efficient Image Processing**: Incorporates `cv2`, `numpy`, and `MediaPipe` to optimize image processing.
- **Progress Tracking**: Utilizes `tqdm` to provide visual feedback during processing.

---

## Technologies Used

- [Python](https://www.python.org/)
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [NumPy](https://numpy.org/)
- [Tqdm](https://tqdm.github.io/)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/human-profiling.git
   cd human-profiling
2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed, then run:
   ```bash
   pip install -r requirements.txt
3.**Set Up the Environment**: 
  Ensure all required libraries, such as OpenCV, DeepFace, MediaPipe, and NumPy, are installed.

---

## Usage

1. **Prepare the Image**:
  Provide a path to a high-quality image of the person you want to profile. The image should clearly show the full body and face for accurate results.
2.**Run the Script**:
  Execute the script by passing the image path:
  ```bash
  python main.py --image_path path_to_image.jpeg

