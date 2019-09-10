# import the necessary packages
import numpy as np
import pandas as pd
import argparse
import cv2
import os
import streamlit as st

# load the COCO class labels our YOLO model was trained on
LABELS_PATH = "coco.names"
WEIGHTS_PATH = "yolov3.weights"
my_path = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(my_path, "yolov3.cfg")

@st.cache
def get_labels_and_colors(labels_path):
    labels = open(labels_path).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3),
        dtype="uint8")

    return labels, colors

# load our YOLO object detector trained on COCO dataset (80 classes)
@st.cache(ignore_hash=True)
def load_network(config_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # determine only the *output* layer names that we need from YOLO
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layer_names

def yolo_v3(image, confidence_threshold=0.5, overlap_threshold=0.3, weights_path=WEIGHTS_PATH, config_path=CONFIG_PATH):
    # Load the network. Because this is cached it will only happen once.
    net, output_layer_names = load_network(config_path, weights_path)

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
                # box followed by the boxes' width and height
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

    new_labels = [
        'pedestrian', #person
        'biker', # bicycle'
        'car', #car
        'biker', #motorbike
        None, # aeroplane
        'truck', #bus
        None, #train
        'truck', #truck
        None, # boat
        'trafficLight', # traffic light
        None, # fire hydrant
        None, # stop sign
        None, # parking meter
        None, # bench
        None, # bird
        None, # cat
        None, # dog
        None, # horse
        None, # sheep
        None, # cow
        None, # elephant
        None, # bear
        None, # zebra
        None, # giraffe
        None, # backpack
        None, # umbrella
        None, # handbag
        None, # tie
        None, # suitcase
        None, # frisbee
        None, # skis
        None, # snowboard
        None, # sports ball
        None, # kite
        None, # baseball bat
        None, # baseball glove
        None, # skateboard
        None, # surfboard
        None, # tennis racket
        None, # bottle
        None, # wine glass
        None, # cup
        None, # fork
        None, # knife
        None, # spoon
        None, # bowl
        None, # banana
        None, # apple
        None, # sandwich
        None, # orange
        None, # broccoli
        None, # carrot
        None, # hot dog
        None, # pizza
        None, # donut
        None, # cake
        None, # chair
        None, # sofa
        None, # pottedplant
        None, # bed
        None, # diningtable
        None, # toilet
        None, # tvmonitor
        None, #
        None, # mouse
        None, # remote
        None, # keyboard
        None, # cell phone
        None, # microwave
        None, # oven
        None, # toaster
        None, # sink
        None, # refrigerator
        None, # book
        None, # clock
        None, # vase
        None, # scissors
        None, # teddy bear
        None, # hair drier
        None, # toothbrush
    ]

    # ensure at least one detection exists
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            label = new_labels[class_IDs[i]]
            if label == None:
                continue

                # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    return pd.DataFrame({
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'labels': labels
    })
