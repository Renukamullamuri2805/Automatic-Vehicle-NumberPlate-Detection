# Automatic-Vehicle-NumberPlate-Detection

This project is a **Streamlit-based Web app** that allows users to detect and extract license number plate information from **images and videos** using computer vision techniques with **OpenCV**

## Features
- Upload an image and extract the license plate number.
- Upload a video and process frames to extyract license plates.
- View the processes image/video with the detected plate highlighted.
- Save results to:
  - 'output_video.mp4' (video output with bounding boxes)
  - 'detected_plates.txt' (text file of detected plates)
     
## Technologies Used

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [NumPy](https://numpy.org/)
- [imutils](https://github.com/jrosebr1/imutils)
- [Pillow](https://python-pillow.org/)

# 2.Install dependencies:
pip install streamlit opencv-python-headless easyocr imutils numpy Pillow

# 3.Run the app:
streamlit run stream6.py

## 📁 Project Structure
- ├── stream6.py         # Main Streamlit app
- ├── output_video.mp4           # Processed video output (generated after video upload)
- ├── detected_plates.txt        # Detected license plate numbers (generated)
- └── README.md                  # Project documentation


