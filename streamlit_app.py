import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = load_model('flower_multiclass_model.keras')

# Define class labels
class_labels = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Set the title of the app
st.title('Flower Class Prediction')
st.write('This application predicts the class of a flower image using a deep learning model. The model is based on the VGG16 architecture, pre-trained on the ImageNet dataset, and fine-tuned to classify images of five types of flowers: Lilly, Lotus, Orchid, Sunflower, and Tulip.')

# Create an option menu for sample images
selected = option_menu(
    menu_title=None,
    options=class_labels,
    icons=['-','-','-','-','-'],
    default_index=0,
    orientation="horizontal"
)
flower_images = {
    'Lily': 'images/lily.jpg',
    'Lotus': 'images/lotus.jpg',
    'Orchid': 'images/orchid.jpg',
    'Sunflower': 'images/sunflower.jpg',
    'Tulip': 'images/tulip.jpg'
}
columns = st.columns(len(class_labels))
# Display the corresponding image below the selected option
if selected:
    # Get the selected column index
    selected_index = class_labels.index(selected)
    
    # Get the corresponding column
    col = columns[selected_index]
    
    # Display the image in the selected column
    col.image(Image.open(flower_images[selected]).resize((800, 800)), caption=selected, use_column_width=True)

# Header for prediction section
st.header('Upload an image to predict')

# File uploader for users to upload their own images
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    st.write("")
    with st.spinner('Classifying...'):
        
        # Preprocess the image
        image = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0  # Rescale the image
        
        # Predict the class
        
        predictions = model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        
    # Display the prediction
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
