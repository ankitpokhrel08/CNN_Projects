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

## Installation

1. **Clone or navigate to this directory**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   Make sure `model.pkl` is in the same directory as `app.py`

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Upload an image** containing a face and get emotion predictions!

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

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Pillow
- TensorFlow/Keras (for model loading)

## Troubleshooting

If you encounter issues:

1. **Model not found**: Ensure `model.pkl` is in the correct directory
2. **Import errors**: Install all requirements using `pip install -r requirements.txt`
3. **Image processing errors**: Make sure uploaded images are valid image files (jpg, png, jpeg)

## License

This project is for educational purposes.
