# Streamlit Self-Driving Car Demo

## Description
This is a demo of Yolov3 Real-Time Object Detection with Streamlit on the Udacity Self-driving-car dataset.                        
                                                                                                                       
The demo runs the YOLO object detection on images from the Udacity dataset. The demo allows the user to select images it shows a reference 'ground image' and an image with the    
detected objects. The demo generates a Streamlit app with the following main characteristics:                                             
- Needed data files are downloaded from the internet.        
- A sidebar allows a user to selects important parameters.   
- A ground image with boxes representing the objects detected by a uman is shown.                                          
- The same image with boxes computed by running the YOLO model is shown for easy comparison.                          
- Some parameters of the YOLO model can be tuned by the use of silders.

## Links
This demo is available at https://github.com/streamlit/demo-self-driving
Yolo: https://pjreddie.com/darknet/yolo/                     
Udacity dataset: https://github.com/udacity/self-driving-car 
Streamlit: https://github.com/streamlit/streamlit  

## How to run this demo
```
pip install streamlit opencv-python
streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/app.py
```
