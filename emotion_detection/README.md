# Emotion Detection Web App

A Streamlit web application for detecting emotions from facial images using a trained deep learning model.

## Features

- 🎭 Detects 7 different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 📸 Easy image upload interface
- 📊 Shows confidence scores for all emotions
- 🎨 Interactive and user-friendly design
- 📱 Responsive layout

## Model Details

- **Input**: Grayscale images resized to 48×48 pixels
- **Output**: Emotion classification with confidence scores
- **Emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise


## Model Code Reference

The model expects images to be preprocessed as follows:

```python
import pickle
import cv2

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Preprocess image
test_img = cv2.imread("image_path.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (48, 48))
test_input = test_img.reshape((1, 48, 48, 1))

# Make prediction
prediction = model.predict(test_input)
emotion_index = prediction[0].argmax()
```

## Tips for Best Results

- Use clear, well-lit face images
- Ensure the face is clearly visible
- Frontal view works best
- Single face per image preferred
