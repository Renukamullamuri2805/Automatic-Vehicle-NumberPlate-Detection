import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
from PIL import Image

def process_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Reduce noise with bilateral filter
    bfliter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection with Canny(for entire image)
    edged = cv2.Canny(bfliter, 30, 200)
    
    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    
    # Loop over the contours
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is None:
        return "No license plate found", img  # No license plate found
    
    # Create mask for the license plate  (get only plate)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    # Crop the license plate area   (mainly focus on plate)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]
    
    # Use easyocr to read the text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped)
    
    #Extracting and Displaying the Text
    text = ""
    for res in result:
        text += res[1] + " "
    
    # Draw the text and rectangle on the image
    font = cv2.FONT_HERSHEY_SIMPLEX     #font
    img = cv2.putText(img, text, (location[0][0][0], location[1][0][1] + 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    img = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    
    return text, img

def process_frame(frame):
    return process_image(frame)

def process_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    license_plates = []
    frame_count = 0
    
    #for continuously read frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # processing every Nth frame
        if frame_count % frame_skip == 0:
            text, img_processed = process_frame(frame)
            if text and "No license plate found" not in text:
                license_plates.append(text.strip())
            out.write(img_processed)
            st.image(img_processed, channels="BGR")
        else:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open('detected_plates.txt', 'w') as f:
        for plate in license_plates:
            f.write(plate + '\n')

    st.write("Processed video saved to: output_video.mp4")
    st.write("Detected license plates saved to: detected_plates.txt")
    st.write("Detected License Plates:")
    for plate in license_plates:
        st.write(plate)

def main():
    st.title("License Plate Detection")

    choice = st.sidebar.selectbox("Choose Application", ["Image", "Video"])

    if choice == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Display original image
            st.subheader("Original Image")
            st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
            # Process the image
            text, img_processed = process_image(img)

            # Display processed image
            st.subheader("Processed Image")
            st.image(img_processed, channels="BGR", caption="Processed Image", use_column_width=True)
            
            # Display the result text
            st.subheader("Detected License Plate")
            st.write(text)
    elif choice == "Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        frame_skip = st.number_input("Frame Skip", min_value=1, value=10)

        if uploaded_file is not None:
            video_path = 'uploaded_video.mp4'
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            process_video(video_path, frame_skip)

if __name__ == "__main__":
    main()