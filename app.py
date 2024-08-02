
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    # Load the image
    #image = Image.open(image_path)
    #image = cv2.imread(image_path)
    image = np.array(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(3, 4, 1)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    
    # Remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.subplot(3, 4, 2)
    plt.title("Blurred")
    plt.imshow(blurred, cmap='gray')
    
    # Thresholding/Binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.subplot(3, 4, 3)
    plt.title("Binary")
    plt.imshow(binary, cmap='gray')
    
    # Dilation and Erosion
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    plt.subplot(3, 4, 4)
    plt.title("Eroded")
    plt.imshow(eroded, cmap='gray')

    # Display the original image and the edge-detected image
    edges = cv2.Canny(eroded, 100, 200)
    plt.subplot(3,4,5) 
    plt.title('Edge Image')
    plt.imshow(edges, cmap='gray')

    
    # Deskewing
    coords = np.column_stack(np.where(edges > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print(f"Detected angle: {angle}") 
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    angle = 0
    print(f"Corrected angle: {angle}") 
    (h, w) = edges.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(edges, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    plt.subplot(3, 4, 6)
    plt.title("Deskewed")
    plt.imshow(deskewed, cmap='gray')

    # Convert to grayscale
    #gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, hierarchy = cv2.findContours(deskewed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    plt.subplot(3, 4, 7)
    plt.title('Contours')
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

    plt.show()
    
    return contour_image

##########################################################################################################################

import os
from PIL import Image
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow
from PIL import Image
import supervision as sv
import cv2


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="LSbJ0tl3WTLn4Aqar0Sp"
)




def creating_display_image(preprocessed_image):
    # Perform inference
    result_doch1 = CLIENT.infer(preprocessed_image, model_id="doctor-s-handwriting/1")

    # Print or process the result
    #print(result_doch1)

    labels = [item["class"] for item in result_doch1["predictions"]]

    detections = sv.Detections.from_inference(result_doch1)

    image_np = np.array(preprocessed_image)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()
    annotated_image = bounding_box_annotator.annotate(
        scene=image_np, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    annotated_image_pil = Image.fromarray(annotated_image)
    sv.plot_image(image=annotated_image_pil, size=(16, 16))

    return annotated_image_pil

######################################################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv

def process_and_plot_image(preprocessed_image):
    # Convert preprocessed image to numpy array
    image_np = np.array(preprocessed_image)

    # Perform inference
    result_doch1 = CLIENT.infer(preprocessed_image, model_id="doctor-s-handwriting/1")

    # Extract labels and detections
    labels = [item["class"] for item in result_doch1["predictions"]]
    detections = sv.Detections.from_inference(result_doch1)

    # Debug: Print unsorted detections and labels
    print("Unsorted Detections and Labels:")
    for i, detection in enumerate(detections):
        print(f"Detection {i}: {detection} - Label: {labels[i]}")

    # Function to extract the x1 coordinate from the detection
    def get_x1(detection):
        return detection.xyxy[0][0]  # Access the first element of the bounding box array

    # Sort detections and labels by the x-coordinate of the bounding box
    sorted_indices = sorted(range(len(detections)), key=lambda i: get_x1(detections[i]))
    sorted_detections = [detections[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Debug: Print sorted detections and labels
    print("Sorted Detections and Labels:")
    for i, detection in enumerate(sorted_detections):
        print(f"Detection {i}: {detection} - Label: {sorted_labels[i]}")

    # Function to plot bounding boxes
    def plot_bounding_boxes(image_np, detections):
        image_with_boxes = image_np.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0]  # Extract bounding box coordinates
            cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        return image_with_boxes

    # Function to plot labels
    def plot_labels(image_np, detections, labels):
        image_with_labels = image_np.copy()
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy[0]  # Extract bounding box coordinates
            label = labels[i]
            cv2.putText(image_with_labels, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image_with_labels

    # Plot bounding boxes with sorted detections
    image_with_boxes = plot_bounding_boxes(image_np, sorted_detections)

    # Plot labels with sorted detections and labels
    image_with_labels = plot_labels(image_np, sorted_detections, sorted_labels)

    # Convert images to RGB for plotting with matplotlib
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    image_with_labels_rgb = cv2.cvtColor(image_with_labels, cv2.COLOR_BGR2RGB)

    # Plot results using matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Bounding Boxes")
    plt.imshow(image_with_boxes_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Labels")
    plt.imshow(image_with_labels_rgb)
    plt.axis('off')

    plt.show()
    return sorted_labels


##########################################################################################################################

def image_result(sorted_labels):
    # Convert list to string
    resulting_string = ''.join(sorted_labels)
    return resulting_string

############################################################################################################################

import streamlit as st
from PIL import Image

# Title of the app
st.title("DOCTOR HANDWRITING DETECTION")

# Upload an image file
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    preprocessed_image_for_streamlit = preprocess_image(image)

    display_boundingbox = creating_display_image(preprocessed_image_for_streamlit)

    result = process_and_plot_image(preprocessed_image_for_streamlit)
    
    input_image_result = image_result(result)
    

    cv2.imwrite('preprocessed_image_2.jpg', preprocessed_image_for_streamlit)

    st.image(image, caption='Input image by user', use_column_width=True)

    st.image(display_boundingbox, caption='Displayed image through bounding boxes', use_column_width=True)

    
    # Display some text
    st.write("Detected text on the image uploaded by the user")
    st.write(input_image_result)
else:
    st.write("Please upload an image file.")

