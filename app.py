import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('C:\\Users\\SRIRAM\\Documents\\Image Classification\\FINAL_MODEL.keras')

# Title of the app
st.title("DISEASE DETECTION n CLASSIFICATION")

# Upload an image file
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    img_opened = Image.open(uploaded_image).convert('RGB')
    #image_opened = Image.open(uploaded_image)
    image_pred = np.array(img_opened)
    image_pred = cv2.resize(image_pred, (150, 150))

    # Convert the image to a numpy array
    image_pred = np.array(image_pred)

    # Rescale the image (if the model was trained with rescaling)
    image_pred = image_pred / 255.0

    # Add an extra dimension to match the input shape (1, 150, 150, 3)
    image_pred = np.expand_dims(image_pred, axis=0)

    # Print the shape of the preprocessed image
    print("Shape of the preprocessed image:", image_pred.shape)

    # Predict using the model
    prediction = model.predict(image_pred)
    
    # Example prediction output
    prediction = np.array(prediction)

    #print(f"Prediction_Classes for different types\ncovid: 0\nnormal_chestray: 1\npneumonia: 2")
    
    # Get the predicted class
    predicted_ = np.argmax(prediction)
    # Display the image
    print(f"Predicted array for : {predicted_}")

    # Decode the prediction
    if predicted_ == 0:
        predicted_class= "Covid"
    elif predicted_ == 1:
        predicted_class= "Normal_chestray"
    else:
        predicted_class= "Pneumonia"

    # Print the predicted class
    print(f'The predicted class is: {predicted_class}')
    #print(prediction)


    st.image(image_pred, caption='Input image by user', use_column_width=True)
    st.write("CLasses description for understanding")
    st.write("Prediction Classes for different types:")
    st.write("COVID: 0")
    st.write("Normal Chest X-ray: 1")
    st.write("Pneumonia: 2")
    st.write("\n")
    # Display some text
    st.write("DETECTED DISEASE DISPLAY")
    st.write(f"Predicted Class : {predicted_}")
    st.write(predicted_class)
else:
    st.write("Please upload an image file.")



