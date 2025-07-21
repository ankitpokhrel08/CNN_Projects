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

## Setup and Installation

1. **Navigate to the project directory**:

   ```bash
   cd cat-dog-prediction
   ```

2. **Create and activate virtual environment** (recommended):

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On macOS/Linux
   # or
   myenv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Ensure you have the trained model**:

   - Make sure `model.pkl` is in the project directory
   - This file contains the pre-trained cat vs dog classifier

2. **Start the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

3. **Open your browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## How to Use

1. **Upload an Image**: Click "Choose an image..." and select a photo of a cat or dog
2. **View Results**: The app will display:
   - The uploaded image
   - Prediction (Cat or Dog) with emoji
   - Confidence percentage
   - Raw prediction value for transparency
3. **Try Different Images**: Upload as many images as you want to test the model

## Technical Details

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

## Dependencies

- **streamlit**: Web application framework
- **opencv-python**: Image processing
- **numpy**: Numerical computations
- **pillow**: Image handling
- **scikit-learn**: Model utilities (if needed)

## Project Structure

```
cat-dog-prediction/
├── app.py              # Main Streamlit application
├── model.pkl           # Pre-trained classifier model
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Troubleshooting

**Model not found error?**

- Ensure `model.pkl` exists in the project directory
- Check that the model was trained and saved properly

**Import errors?**

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment if using one

**Image not loading?**

- Supported formats: JPG, JPEG, PNG
- Try a different image or check image file integrity

## Future Improvements

- [ ] Add support for multiple image formats (GIF, WEBP)
- [ ] Implement batch prediction for multiple images
- [ ] Add model confidence visualization
- [ ] Include example images for testing
- [ ] Add data augmentation preview
