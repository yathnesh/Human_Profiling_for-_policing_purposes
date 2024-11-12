import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm  # Import tqdm for progress bar

def load_images_and_labels(directory, labels_df):
    images = []
    labels = []
    for index, row in labels_df.iterrows():
        img_path = os.path.join(directory, row['file'])
        image = cv2.imread(img_path)
        if image is not None:  # Check if the image was loaded properly
            image = cv2.resize(image, (128, 128))  # Resize to match model input
            images.append(image)
            labels.append(row['race'])  # Adjust based on your CSV column name
    return np.array(images), np.array(labels)

def build_race_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load dataset
    train_labels_path = 'FairFace/fairface_label_train.csv'
    train_images_path = 'FairFace/'

    # Read CSV file
    labels_df = pd.read_csv(train_labels_path)

    # Load images and labels
    images, labels = load_images_and_labels(train_images_path, labels_df)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

    # Build the model
    model = build_race_model(num_classes=labels_categorical.shape[1])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with progress bar
    epochs = 30
    for epoch in tqdm(range(epochs), desc='Model Training', unit='epoch'):
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, callbacks=[early_stopping], verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))

if __name__ == '__main__':
    main()
