# Streamlit Self-Driving Car Demo

## Description
This is a demo of the YOLO real-time object detection model with Streamlit on the Udacity self-driving-car dataset.

The demo consists of an app built with Streamlit. The app runs the YOLO model
on images from the Udacity dataset in real time. The demo allows the user to select an image. It shows
a reference 'ground image' for the detected objects and an image with the objects detected by YOLO.

These are the main characteristics of the app:
- All needed data files are downloaded from the internet when the app starts. The app shows
a progress bar for each download and store the data locally.
- A sidebar allows for a user to selects an image on which YOLO will run.
- A ground image with boxes representing the objects detected by a human is shown.
- The same image with boxes computed by the YOLO model is shown for easy comparison.
- The YOLO model is run in real time and it is possible to change its parameter in real time
as well.

## Links
- This demo is available at https://github.com/streamlit/demo-self-driving
- Yolo: https://pjreddie.com/darknet/yolo/
- Udacity dataset: https://github.com/udacity/self-driving-car
- Streamlit: https://github.com/streamlit/streamlit

## How to run this demo
```
pip install streamlit opencv-python
streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/app.py
```

The `streamlit` command opens a browser window and automatically starts the app.

## How to use the app
Once the app has started, you will see it consists in a sidebar and a main column.
You can tweak the parameters in the sidebar and use the main column to see how this affects the image
selection and the outcome on the YOLO model. This is where the app shows the ground image and
the results of the YOLO model run.
You can use the sliders and the picker in the sidebar to select an image. The app will show the
ground image in the main column and it will prompt you to run the YOLO model. You can do this by unchecking
the checkbox that now appears in the sidebar. Once you uncheck the checkbox, YOLO will run and an image with
the objects detected by YOLO will be shown. You can compare this image with the ground image and, by tweaking
the model parameters in the sidebar, see if you can get the two images to match.
