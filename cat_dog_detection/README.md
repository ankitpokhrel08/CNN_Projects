# 🐱🐶 Cat vs Dog Image Classifier

A simple web application built with Streamlit that uses a pre-trained machine learning model to classify images as either cats or dogs.

## Features

- **Easy Image Upload**: Support for JPG, JPEG, and PNG formats
- **Real-time Prediction**: Instant classification results
- **Confidence Score**: Shows how confident the model is in its prediction
- **Clean Interface**: User-friendly web interface with emojis and visual feedback

## Model Information

- **Input Size**: 256x256 pixels
- **Classes**: 0 = Cat 🐱, 1 = Dog 🐶
- **Preprocessing**: Images are resized to 256x256 and reshaped for model input
- **Output**: Float value between 0 and 1, rounded to nearest integer for classification

# How to use
1. **Upload an Image**: Click "Choose an image..." and select a photo of a cat or dog
2. **View Results**: The app will display:
   - The uploaded image
   - Prediction (Cat or Dog) with emoji
   - Confidence percentage
   - Raw prediction value for transparency
3. **Try Different Images**: Upload as many images as you want to test the model


### Preprocessing Pipeline

```python
# Resize image to model input size
test_img = cv2.resize(img_array, (256, 256))
# Reshape for model input (batch_size, height, width, channels)
test_input = test_img.reshape((1, 256, 256, 3))
```

### Prediction Logic

- Raw output: Float between 0 and 1
- Classification: `round(prediction[0][0])`
- 0 → Cat 🐱
- 1 → Dog 🐶
