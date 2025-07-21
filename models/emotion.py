import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle

# Define emotion labels
def label(num):
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return labels[num]

# Page configuration
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('emotion.pkl', 'rb'))
        return model
    except FileNotFoundError:
        st.error("Error: emotion.pkl not found. Please ensure the model file is in the same directory.")
        st.stop()

model = load_model()

# Sidebar with model information
with st.sidebar:
    st.header("🧠 Model Information")
    st.markdown("""
    ### About This Model
    
    **Task**: Facial Emotion Recognition
    
    **Input**: Grayscale images (48×48 pixels)
    
    **Emotions Detected**:
    - 😠 Angry
    - 🤢 Disgust  
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    
    **How it works**:
    1. Converts uploaded image to grayscale
    2. Resizes to 48×48 pixels
    3. Feeds through neural network
    4. Predicts emotion category
    
    **Tips for best results**:
    - Use clear face images
    - Ensure good lighting
    - Face should be clearly visible
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Dataset Information
    **Source**: [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
    
    **Details**: 48×48 grayscale images of facial expressions
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ using Streamlit**")

# Main content area
st.title('😊 Facial Emotion Detection')
st.markdown("### *Upload a face image or take a photo to detect emotions!* 🎭")

# Add collapsible instructions
with st.expander("📋 Click here for instructions"):
    st.markdown("""
    ### How to use:
    1. **Upload Image**: Choose a JPG, JPEG, or PNG file with a clear face
    2. **Camera**: Enable camera and take a photo
    3. **Best Results**: Use well-lit, frontal face images
    """)

st.markdown("---")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Read and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Your uploaded image', width=300)
        
        with col2:
            with st.spinner('🔍 Analyzing...'):
                # Convert PIL image to numpy array
                img_array = np.array(image)
                
                # Convert to grayscale if it's a color image
                if len(img_array.shape) == 3:
                    # Convert RGB to BGR for OpenCV
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    # Convert to grayscale
                    test_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                else:
                    test_img = img_array
                
                # Resize to 48x48 as required by the model
                test_img = cv2.resize(test_img, (48, 48))
                
                # Reshape for model input
                test_input = test_img.reshape((1, 48, 48, 1))
                
                # Normalize pixel values (if your model expects normalized input)
                test_input = test_input.astype('float32') / 255.0
                
                # Make prediction
                prediction = model.predict(test_input)
                predicted_emotion_index = prediction[0].argmax()
                predicted_emotion = label(predicted_emotion_index)
                confidence = prediction[0][predicted_emotion_index] * 100
                
                # Create emotion emoji mapping
                emotion_emojis = {
                    'Angry': '😠',
                    'Disgust': '🤢',
                    'Fear': '😨',
                    'Happy': '😊',
                    'Neutral': '😐',
                    'Sad': '😢',
                    'Surprise': '😲'
                }
                
                emotion_emoji = emotion_emojis.get(predicted_emotion, '😐')
                
                # Display prediction result
                st.markdown("### 🎯 Prediction")
                st.success(f"{emotion_emoji} **{predicted_emotion}**")
                st.metric("Confidence", f"{confidence:.1f}%")
        
        # Show horizontal probability bars below the image
        st.markdown("### 📊 All Emotions")
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        probabilities = prediction[0] * 100
        emotion_emojis = {
            'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
            'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
        }
        
        for emotion, prob in zip(emotions, probabilities):
            emoji = emotion_emojis[emotion]
            st.progress(float(prob)/100, text=f"{emoji} {emotion}: {prob:.1f}%")

with tab2:
    st.subheader("📷 Take a Photo")
    
    # Enable camera checkbox
    enable_camera = st.checkbox("Enable camera", key="enable_cam")
    
    # Camera input
    camera_picture = st.camera_input("Take a picture", disabled=not enable_camera)
    
    if camera_picture is not None:
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(camera_picture, caption='Your captured image', width=300)
        
        with col2:
            with st.spinner('🔍 Analyzing...'):
                # Read image file buffer as bytes
                bytes_data = camera_picture.getvalue()
                
                # Convert bytes to numpy array for OpenCV processing
                nparr = np.frombuffer(bytes_data, np.uint8)
                img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Convert to grayscale
                test_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                
                # Resize to 48x48 as required by the model
                test_img = cv2.resize(test_img, (48, 48))
                
                # Reshape for model input
                test_input = test_img.reshape((1, 48, 48, 1))
                
                # Normalize pixel values
                test_input = test_input.astype('float32') / 255.0
                
                # Make prediction
                prediction = model.predict(test_input)
                predicted_emotion_index = prediction[0].argmax()
                predicted_emotion = label(predicted_emotion_index)
                confidence = prediction[0][predicted_emotion_index] * 100
                
                # Create emotion emoji mapping
                emotion_emojis = {
                    'Angry': '😠',
                    'Disgust': '🤢',
                    'Fear': '😨',
                    'Happy': '😊',
                    'Neutral': '😐',
                    'Sad': '😢',
                    'Surprise': '😲'
                }
                
                emotion_emoji = emotion_emojis.get(predicted_emotion, '😐')
                
                # Display prediction result
                st.markdown("### 🎯 Prediction")
                st.success(f"{emotion_emoji} **{predicted_emotion}**")
                st.metric("Confidence", f"{confidence:.1f}%")
        
        # Show horizontal probability bars below the image
        st.markdown("### 📊 All Emotions")
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        probabilities = prediction[0] * 100
        emotion_emojis = {
            'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
            'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
        }
        
        for emotion, prob in zip(emotions, probabilities):
            emoji = emotion_emojis[emotion]
            st.progress(float(prob)/100, text=f"{emoji} {emotion}: {prob:.1f}%")

# Add some sample images section
st.markdown("---")
st.subheader("💡 Tips for Better Results")
st.markdown("""
- **Face visibility**: Ensure the face is clearly visible and well-lit
- **Image quality**: Higher resolution images generally work better
- **Single face**: The model works best with images containing one clear face
- **Frontal view**: Face looking towards the camera gives better results
- **Natural expressions**: Genuine emotions are detected more accurately
""")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Powered by Deep Learning*")
