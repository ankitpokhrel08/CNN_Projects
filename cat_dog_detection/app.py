import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from model_utils import load_model_from_parts
import os

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the model from compressed parts or fallback to regular pickle file."""
    try:
        # First try to load from compressed parts
        if os.path.exists('model_part1.pkl.gz') and os.path.exists('model_part2.pkl.gz'):
            st.info("🔄 Loading model from compressed parts (first time may take a moment)...")
            model = load_model_from_parts('model_part1.pkl.gz', 'model_part2.pkl.gz')
            st.success("✅ Model loaded successfully from compressed parts!")
            return model
        # Fallback to regular pickle file
        elif os.path.exists('model.pkl'):
            model = pickle.load(open('model.pkl', 'rb'))
            return model
        else:
            raise FileNotFoundError("No model files found")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure model files are present in the directory.")
        st.stop()

model = load_model()

# Sidebar with model description
with st.sidebar:
    st.header("🤖 Model Information")
    st.markdown("""
    ### About This Model
    
    **Architecture**: Convolutional Neural Network (CNN)
    
    **Dataset**: [Dogs vs Cats - Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats/suggestions)
    
    **Training Details**:
    - Input size: 256×256 pixels
    - Color channels: RGB (3 channels)
    - Dataset: 25,000 images of cats and dogs
    - Source: Kaggle Competition Dataset
    
    **How it works**:
    1. Preprocesses uploaded image to 256×256
    2. Feeds through neural network layers
    3. Outputs probability (0-1 scale)
    
    **Prediction Scale**:
    - 0.0 = 100% Cat 🐱
    - 0.5 = Uncertain 🤔
    - 1.0 = 100% Dog 🐶
    
    **Accuracy**: The model rounds predictions to nearest integer for final classification.
    
    **Data Source**: This model was trained on the famous Kaggle Dogs vs Cats dataset, containing high-quality images of cats and dogs.
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ using Streamlit**")

# Main content area
st.title('🐱🐶 Cat vs Dog Image Classifier')
st.markdown("### *Is it a fluffy cat or a playful dog? Let our AI detective solve the mystery!* 🕵️‍♂️")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Choose an image to classify...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Read and display the image (smaller size)
        image = Image.open(uploaded_file)
        st.subheader("📸 Uploaded Image")
        st.image(image, caption='Your uploaded image', width=300)
    
    with col2:
        with st.spinner('🔍 Analyzing image...'):
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Convert RGB to BGR (if needed for OpenCV)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Preprocess the image exactly as in your example
            test_img = cv2.resize(img_array, (256, 256))
            test_input = test_img.reshape((1, 256, 256, 3))
            
            # Make prediction
            prediction = model.predict(test_input)
            
            # Get the predicted class (0 for cat, 1 for dog)
            # Round to nearest integer to handle values like 0.99999
            predicted_class = round(prediction[0][0])
            
            # Display the result
            st.subheader("🎯 Prediction Results")
            
            if predicted_class == 1:
                st.success("🐶 **It's a Dog!**")
                st.balloons()
            else:
                st.success("🐱 **It's a Cat!**")
                st.balloons()
            
            # Show confidence and raw prediction
            confidence = abs(prediction[0][0] - 0.5) * 200
            st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Raw prediction value
            st.info(f"**Raw Prediction Value**: {prediction[0][0]:.6f}")
            
            # Add a fun interpretation
            if confidence > 80:
                st.markdown("🎉 *I'm very confident about this prediction!*")
            elif confidence > 60:
                st.markdown("😊 *Pretty sure about this one!*")
            else:
                st.markdown("🤔 *This one's a bit tricky, but here's my best guess!*")

