import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    # Convert BytesIO to PIL Image
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to match model's expected input size
    
    # Convert PIL Image to array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Set background color to black
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "DONE BY"])

# Display Team Details in the Sidebar
if app_mode == "":
    st.sidebar.header("DONE BY")
    st.sidebar.markdown("""
    ### About
    **Name:** M. Shirisha Mahadevuni<br>
    **Roll Number:** 21311A1935<br>
    **College:** Sreenidhi Institute of Science and Technology<br>
    **Branch:** Electronics and Computer Engineering<br><br>
    """, unsafe_allow_html=True)
    st.sidebar.write("### Project Description")
    st.sidebar.write("""
    Our team is dedicated to developing and optimizing the ResNet50 model for
    various computer vision tasks.
    ResNet50 is a powerful convolutional neural network architecture that has proven
    highly effective in image classification and other vision-related applications.
    """)

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌱
    Our mission is to help in identifying plant diseases efficiently. Upload an
    image of a plant, and our system will analyze it to detect any signs of diseases.
    Together, let's protect our crops and ensure a healthier harvest!
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image
    of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to
    identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques
    and deep learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick
    decision-making.
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and
    experience the power of our Plant Disease Recognition System!
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])
    
    if test_image is not None:
        st.image(test_image)
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            
            # Predict the disease
            result_index = model_prediction(test_image)
            
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot',
                           'Apple___Cedar_apple_rust', 'Apple___healthy', 'Potato___Early_blight',
                           'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch',
                           'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                           'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                           'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
                           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']
            st.success(f"Model predicts it's a {class_name[result_index]}")

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the
    original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and
    diseased crop leaves categorized into 38 different classes. The total dataset is
    divided into an 80/20 ratio of training and validation set preserving the directory
    structure.
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)
