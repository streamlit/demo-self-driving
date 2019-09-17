# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#################################################################
# Self-driving car demo                                         #
#################################################################
# Available at https://github.com/streamlit/demo-self-driving   #
#                                                               #
# Demo of Yolov3 real-time object detection with Streamlit on   #
# the Udacity self-driving-car dataset.                         #
#                                                               #
# Yolo: https://pjreddie.com/darknet/yolo/                      #
# Udacity dataset: https://github.com/udacity/self-driving-car  #
# Streamlit: https://github.com/streamlit/streamlit             #
#                                                               #
# See README.md for more details                                #
#################################################################

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import urllib
import cv2
from collections import OrderedDict

#################
#   Constants   #
#################

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files with url and optionally md5
EXTERNAL_FILES = OrderedDict({
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    },
    "README.md": {
        "url": "https://raw.githubusercontent.com/streamlit/demo-self-driving/master/README.md"
    },
    "app.py": {
        "url": "https://raw.githubusercontent.com/streamlit/demo-self-driving/master/app.py"
    }
})

# Colors for the boxes
LABEL_COLORS = {
    "car": [255, 0, 0],
    "pedestrian": [0, 255, 0],
    "truck": [0, 0, 255],
    "trafficLight": [255, 255, 0],
    "biker": [255, 0, 255],
}

MEGABYTES = 2.0 ** -20.0


#################
#   Functions   #
#################

# Check is a file has been downloaded.
def is_downloaded(file_path):
    if not os.path.exists(file_path):
        return False
    if "size" not in EXTERNAL_FILES[file_path]:
        return True
    return os.path.getsize(file_path) == EXTERNAL_FILES[file_path]["size"]


# Download a file. Report progress using st.progress that draws a progress bar on the Streamlit UI.
def download_file(file_path):
    if file_path not in EXTERNAL_FILES:
        raise Exception("Unknown file: %s" % file_path)
    # Declare some variable used for Streamlit commands here so that we can use them in the try finally.
    # This is for a markdown title.
    title = None
    # This is to show a warning if the download is in progress.
    weights_warning = None
    # This is for the Streamlit progress bar.
    progress_bar = None
    try:
        download_message = "Downloading %s..." % file_path
        title = st.markdown("## Getting data files")
        weights_warning = st.warning(download_message)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as fp:
            with urllib.request.urlopen(EXTERNAL_FILES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    progress = counter / length
                    weights_warning.warning("%s (%6.2f/%6.2f MB)" %
                                            (download_message, counter * MEGABYTES, length * MEGABYTES))
                    progress_bar.progress(progress if progress <= 1.0 else 1.0)
                    fp.write(data)
    finally:
        if title:
            title.empty()
        if weights_warning:
            weights_warning.empty()
        if progress_bar:
            progress_bar.empty()


# Download a single file if not on the filesystem and make its content available as a string.
def get_file_content_as_string(filename):
    if not is_downloaded(filename):
        download_file(filename)
    with open(filename, "r") as fp:
        return fp.read() + "\n\n"


# The preamble shows the README or the source code of the app and takes care of downloading the data files.
def preamble():
    # Show README by calling st.markdown. This is a Streamlit command that shows markdown text.
    # We get a handle of this, so that we can make it disappear later when we need it by calling
    # readme_text.empty().
    readme_text = st.markdown(get_file_content_as_string("README.md"))

    # Download data files.
    for filename in EXTERNAL_FILES.keys():
        if filename != "README.md" and not is_downloaded(filename):
            download_file(filename)

    # Select box to choose whether to show the README, the source code, or run the app
    # Note that the select is in a sidebar under the title "About". We also select option 0
    # to be the default and we pass in a function to format the labels to our liking.
    st.sidebar.title("What to do")
    option = st.sidebar.selectbox(
        "Choose",
        ["readme", "code", "app"],
        0,
        lambda opt: {
            "readme": "Show the README",
            "code": "Show the source code",
            "app": "Run the App"
        }[opt])
    if option == "readme":
        return False
    if option == "code":
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))
        return False

    readme_text.empty()
    return True

# st.cache allows us to reuse computation across runs, making Streamlit really fast.
# This is a common usage, where we load data from an endpoint once and then reuse
# it across runs.
@st.cache
def load_metadata(url):
    metadata = pd.read_csv(url)
    return metadata

# An amazing property of st.cached functions is that you can pipe them into
# each other, creating a computation DAG (directed acyclic graph). Streamlit
# automatically recomputes only the *subset* of the DAG required to get the
# right answer!
@st.cache
def create_summary(metadata):
    one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
    summary = one_hot_encoded.groupby(["frame"]).sum()
    return summary

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image


# This function add the boxes of the detected objects to an image.
# It returns an image with the added boxes.
def add_boxes(image, boxes):
    image = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image[ymin:ymax,xmin:xmax,:] += LABEL_COLORS[label]
        image[ymin:ymax,xmin:xmax,:] /= 2
    return image.astype(np.uint8)


# This function select frames from the summary based on label, min_elts, and max_elts.
@st.cache
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index


# Load our YOLO object detector trained on COCO dataset (80 classes).
@st.cache(ignore_hash=True)
def load_network(config_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # determine only the *output* layer names that we need from YOLO
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layer_names


# Run the YOLO model to detect objects.
def yolo_v3(image,
            confidence_threshold=0.5,
            overlap_threshold=0.3):
    # Load the network. Because this is cached it will only happen once.
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # load our input image and grab its spatial dimensions
    H, W = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    class_IDs = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes" width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
        overlap_threshold)

    # remap labels
    new_labels = [None, ] * 80
    for e in [
        (0, "pedestrian"),
        (1, "biker"),
        (2, "car"),
        (3, "biker"),
        (5, "truck"),
        (7, "truck"),
        (9, "trafficLight"),

    ]:
        new_labels[e[0]] = e[1]

    # ensure at least one detection exists
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            label = new_labels[class_IDs[i]]
            if label is None:
                continue
            label = new_labels[class_IDs[i]]

            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    return pd.DataFrame({
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "labels": labels
    })


########
# Main #
########
def main():
    # Preamble to the app (show code, README, download files)
    if not preamble():
        return

    # Do some preparation by loading metadata from S3...
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    # ... and creating a summary our of the metadata
    summary = create_summary(metadata)

    # Add a title to the sidebar for the selection of the frame.
    st.sidebar.title("Frame")

    # Formatter for the labels.
    def formatter(label):
        return {
            "label_biker": "biker",
            "label_car": "car",
            "label_pedestrian": "pedestrian",
            "label_trafficLight": "traffic light",
            "label_truck": "truck"
        }[label]

    # Draw a selection box for the labels
    label = st.sidebar.selectbox(
        "Pick a label",
        summary.columns,
        2,
        formatter
    )
    # Draw a slider to select the range of objects we want in the image
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % formatter(label), 0, 25, [10, 20])

    # Select frames based on the selection in the sidebar
    selected_frames = get_selected_frames(summary, label, min_elts, max_elts)
    if len(selected_frames) < 1:
        st.error("No frames fit the criteria. 😳 Please select different label or number. ✌️")
        return

    objects_per_frame = summary.loc[selected_frames, label].reset_index(drop=True).reset_index()

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)
    selected_frame = selected_frames[selected_frame_index]
    # Compose the image url for the frame.
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    # Load the image from S3.
    image = load_image(image_url)

    # Draw an altair chart in the sidebar with information on the frame.
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % label))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(
        alt.X("selected_frame:Q", axis=None)
    )
    st.sidebar.altair_chart(alt.layer(chart, vline))

    # Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])

    # Create an header for the ground image.
    st.write("### Ground Truth (frame #`%i`)" % selected_frame_index)

    # Draw the ground image with the boxes that show the objects.
    image_with_boxes = add_boxes(image, boxes)
    st.image(image_with_boxes, use_column_width=True)

    # Add a section in the sidebar for the model.
    st.sidebar.markdown("----\n # Model")
    # This is an empty line
    st.sidebar.markdown("")

    # The following code runs the YOLO model.
    # It also adds two sliders in the sidebar for confidence threshold and overlap threshold.
    # These are parameters of the models. When the user changes these sliders, the model re-runs.
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    yolo_boxes = yolo_v3(image,
                         overlap_threshold=overlap_threshold,
                         confidence_threshold=confidence_threshold)
    # Add the boxes to the image.
    image_yolo = add_boxes(image, yolo_boxes)
    # Add an header.
    st.write("### YOLO Detection (overlap `%3.1f`) (confidence `%3.1f`)" %
             (overlap_threshold, confidence_threshold))
    # Draw the image with the boxes computed by YOLO. This image has the detected objects.
    st.image(image_yolo, use_column_width=True)


if __name__ == "__main__":
    main()
