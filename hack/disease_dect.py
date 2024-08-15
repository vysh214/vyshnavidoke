import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

def app():
    # Load the model
    model = tf.keras.models.load_model('C:\\Users\\SRIRAM\\Documents\\New Folder\\FINAL_MODEL.keras')

    # Title of the app
    st.title("DISEASE DETECTION AND CLASSIFICATION")

    # Upload an image file
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open and preprocess the image
        img_opened = Image.open(uploaded_image).convert('RGB')
        image_pred = np.array(img_opened)
        image_pred = cv2.resize(image_pred, (150, 150))

        # Rescale the image (if the model was trained with rescaling)
        image_pred = image_pred / 255.0

        # Add an extra dimension to match the input shape (1, 150, 150, 3)
        image_pred = np.expand_dims(image_pred, axis=0)

        # Predict using the model
        prediction = model.predict(image_pred)

        # Get the predicted class
        predicted_ = np.argmax(prediction)

        # Decode the prediction
        if predicted_ == 0:
            predicted_class = "Covid"
        elif predicted_ == 1:
            predicted_class = "Normal Chest X-ray"
        else:
            predicted_class = "Pneumonia"

        # Display the original image
        st.image(img_opened, caption='Input image by user', use_column_width=True)

        # Display class descriptions
        st.write("Classes description for understanding:")
        st.write("COVID: 0")
        st.write("Normal Chest X-ray: 1")
        st.write("Pneumonia: 2")

        # Display the detected disease
        st.write("DETECTED DISEASE DISPLAY")
        st.write(f"Predicted Class : {predicted_}")
        st.write(predicted_class)
    else:
        st.write("Please upload an image file.")



