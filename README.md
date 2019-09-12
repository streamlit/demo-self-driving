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
The app consists of a sidebar, used mostly for controls and a main column that shows content.

The first thing that you will see once the app has started is the README in the main column and a checkbox in the
sidebar. You can use the checkbox to hide the README. At any time, you can go back to the README by checking the 
checkbox. 

When you dismiss the README, some controls will appear in the sidebar. These controls allow you to select a frame
to show in the main column as _ground image_. Tweak the controls and see how different ground images
are shown. For example, select the `label_pedestrian` in the `label` picker. Tune the minimum and
maximum values on `label_pedestrian` slider to `10.0` and `20.0`, respectively, and just pick any frame by sliding the
`label_pedestrian frame` slider.

At this point, the app shows the ground image in the main column and prompts you to run the YOLO model. 
You can do this by unchecking the checkbox that now appears in the sidebar. Once you uncheck the checkbox, 
YOLO will run and an image with the objects detected by YOLO will be shown. You can compare this image with 
the ground image and, by tweaking the model parameters in the sidebar and see if you can get the two images to match!
