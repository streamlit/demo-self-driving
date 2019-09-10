# -*- coding: utf-8 -*-
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import urllib
import cv2
import streamlit_demo.self_driving.yolov3
import time

DATA_URL_ROOT = 'https://streamlit-self-driving.s3-us-west-2.amazonaws.com/'
LABELS_FILENAME = os.path.join(DATA_URL_ROOT, 'labels.csv.gz')
LABEL_COLORS = {
    'car': [255, 0, 0],
    'pedestrian': [0, 255, 0],
    'truck': [0, 0, 255],
    'trafficLight': [255, 255, 0],
    'biker': [255, 0, 255],
}

# st.cache allows us to reuse computation across runs, making Streamlit really fast.
# This is a comman usage, where we load data from an endpoint once and then reuse
# it across runs.
@st.cache
def load_metadata(url):
    print('url:' + url)
    metadata = pd.read_csv(url)
    # one_hot_encoded = pd.get_dummies(metadata[['frame', 'label']], columns=['label'])
    # summary = one_hot_encoded.groupby(['frame']).sum()
    return metadata

# An amazing property of st.cache'ed functions is that you can pipe them into
# each other, creating a computaiton DAG (directed acyclic graph). Streamlit
# automatically recomputes only the *subset* of the DAG required to get the
# right answer!
@st.cache
def create_summary(metadata):
    one_hot_encoded = pd.get_dummies(metadata[['frame', 'label']], columns=['label'])
    summary = one_hot_encoded.groupby(['frame']).sum()
    return summary

@st.cache# (show_spinner=False)
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:,:,[2,1,0]] # BGR -> RGB
    return image

def add_boxes(image, boxes):
    image = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image[ymin:ymax,xmin:xmax,:] += LABEL_COLORS[label]
        image[ymin:ymax,xmin:xmax,:] /= 2
    return image.astype(np.uint8)

def main():
    metadata = load_metadata(LABELS_FILENAME)
    summary = create_summary(metadata)
    # st.show(metadata[:1000])
    # st.show(summary[:1000])

    st.sidebar.title('Frame')
    label = st.sidebar.selectbox('label', summary.columns)
    min_elts, max_elts = st.sidebar.slider(label, 0, 25, [10, 20])

    @st.cache
    def get_selected_frames(summary, label, min_elts, max_elts):
        return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index
    selected_frames = get_selected_frames(summary, label, min_elts, max_elts)
    if len(selected_frames) < 1:
        st.error('No frames fit the criteria. 😳 Please select different label or number. ✌️')
        return

    # st.show(selected_frames)

    frames = metadata.frame.unique()
    objects_per_frame = summary.loc[selected_frames, label].reset_index(drop=True).reset_index()

    selected_frame_index = st.sidebar.slider(label + ' frame', 0, len(selected_frames) - 1, 0)
    selected_frame = selected_frames[selected_frame_index]
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X('index:Q', scale=alt.Scale(nice=False)),
        alt.Y('%s:Q' % label))
    selected_frame_df = pd.DataFrame({'selected_frame': [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color='red').encode(
        alt.X('selected_frame:Q',axis=None)
    )
    st.sidebar.altair_chart(alt.layer(chart, vline))
    boxes = metadata[metadata.frame == selected_frame].drop(columns=['frame'])

    # st.image(image, use_column_width=True)

    "### Ground Truth `%i`/`%i` : `%s`" % (selected_frame_index, len(selected_frames), selected_frame)

    image_with_boxes = add_boxes(image, boxes)
    st.image(image_with_boxes, use_column_width=True)

    st.sidebar.markdown('----\n # Model')
    st.sidebar.markdown('')
    if st.sidebar.checkbox('Run Yolo Detection', True):
        confidence_threshold = st.sidebar.slider('confidence_threshold', 0.0, 1.0, 0.5, 0.01)
        overlap_threshold = st.sidebar.slider('overlap threshold', 0.0, 1.0, 0.3, 0.01)

        yolo_boxes = yolov3.yolo_v3(image,
            overlap_threshold=overlap_threshold,
            confidence_threshold=confidence_threshold)
        image3 = add_boxes(image, yolo_boxes)
        st.write('### YOLO Detection (overlap `%3.1f`) (confidence `%3.1f`)' % \
            (overlap_threshold, confidence_threshold))
        st.image(image3, use_column_width=True)
    else:
        st.warning('Click _Run Yolo Detection_ on the left to compare with ground truth.')

