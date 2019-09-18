# Streamlit Demo: The Udacity Self-driving Car Image Browser

This project demonstrates the [Udacity self-driving-car dataset](https://github.com/udacity/self-driving-car) and [YOLO object detection](https://pjreddie.com/darknet/yolo) into an interactive [Streamlit](https://streamlit.io) app.

The complete demo is [implemented in less than 300 lines of Python]() and illustrates all the major building blocks of Streamlit.

![Making-of Animation](https://raw.githubusercontent.com/streamlit/demo-self-driving/master/udacity_demo_making_of.gif "Making-of Animation")

## How to run this demo
```
pip install streamlit opencv-python altair
streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/app.py
```

❗️**NOTE: This demo requires version 0.46 of Streamlit which has not yet been released!**

### Questions? Comments?

Please ask in the [Streamlit community](https://discuss.streamlit.io).


<!-- 
The demo consists of an app built with Streamlit. The app runs the YOLO model
on images from the Udacity dataset in real time. The demo allows the user to select an image. It shows
a reference 'ground image' for the detected objects and an image with the objects detected by YOLO.

## Where to go from here
Once the app has loaded the model data, use the selection box in the sidebar to the left to either see the source code,
run the app, or see this README

For more details on this app see the [extended README](https://github.com/streamlit/demo-self-driving/blob/master/README-extended.md)

## Links
- This demo is available on Github at https://github.com/streamlit/demo-self-driving
- Yolo: 
- Udacity dataset: 
- Streamlit: https://github.com/streamlit/streamlit -->
