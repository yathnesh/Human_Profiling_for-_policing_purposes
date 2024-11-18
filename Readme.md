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

1. **Prepare the Image**:Provide a path to a high-quality image of the individual you want to profile. The image should clearly show the full body and face to ensure accurate results.
2. **Run the Script**: Execute the script by specifying the image path:
   ```bash
   python main.py --image_path path_to_image.jpeg

3. **View the Results**: The program will process the image and output a detailed profile, including race, height, body type, clothing type, and dominant color.

---

## How It Works

### Race Detection
- **DeepFace** analyzes the image for facial features and predicts the dominant race along with its probabilities.

### Height and Body Type Estimation
- **MediaPipe Pose** identifies key body landmarks to estimate approximate height and categorize body type.

### Clothing Detection
- Uses **K-means clustering** to identify the dominant color of the clothing and maps it to predefined categories.

### Profile Compilation
- Combines all extracted data into a structured profile for easy interpretation.

---

## Limitations

### Image Dependency
- Results are highly dependent on the quality and clarity of the image provided.
- Obstructions or poor lighting may reduce accuracy.

### Bias in Models
- Pre-trained models used for race detection may contain inherent biases.

### Simplistic Clothing Detection
- The clothing detection system relies on the dominant color, limiting its ability to handle patterns or multiple colors.

---

## Future Work

- Integrate more advanced deep learning models for height and body type estimation.
- Expand clothing detection to include fabric patterns and multi-color identification.
- Add real-time profiling capabilities using video input.
- Introduce additional features such as emotion recognition and gesture analysis.

---

## Contributing

Contributions are welcome! Follow these steps to get started:

1. **Fork the Repository**:  
   Click the "Fork" button on the repository page.

2. **Clone Your Fork**:  
   ```bash
   git clone https://github.com/your-username/human-profiling.git
3. **Create a Feature Branch**:  
   ```bash
   git checkout -b feature-name
4. **Commit and Push Changes**:  
   ```bash
   git commit -m "Add new feature"
   git push origin feature-name
5.**Open a Pull Request**:
   Submit your changes for review.

---

## License

This project is licensed under the [MIT License](LICENSE).
