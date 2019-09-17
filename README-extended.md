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
pip install streamlit opencv-python altair pandas numpy
streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/app.py
```

The `streamlit` command opens a browser window and automatically starts the app.

## How to use the app
The app consists of a sidebar, used mostly for controls, and a main column that shows content.

The first thing that you will see is the README in the main column and a selection box in the
sidebar. You can use the selection box to switch between the README, the source code, and the core part of the app.

When we choose to run the app in the selection box, some controls will appear in the sidebar. These controls allow you to select
a frame to show in the main column as _ground image_. Tweak the controls and see how different ground images
are shown. For example, select `pedestrian` in the picker. Tune the minimum and
maximum values in the `How many pedestrians` slider to `10.0` and `20.0`, respectively, and just pick any frame
by sliding the `Choose a frame` slider. This means that you want to select an image with _pedestrians_ and you want between
10 and 20 of them. The _frame_ slider allows you to select a single frame - i.e. an image - out of the images satifying
the above criteria. A small chart below the slider shows how many of the target objects (_pedestrians_ in this case)
for each frame and a red vertical line highlights the frame being selected.

Below the ground image the app shows the very same image with the objects detected by the YOLO model.
You can compare this image with the ground image and, by tweaking the model parameters in the sidebar,
see if you can get the two images to match! Note that YOLO runs in real time and every time you change
the parameters YOLO runs again.

