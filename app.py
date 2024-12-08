import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from yt_dlp import YoutubeDL

# Load the model with custom objects (if needed)
model = load_model('siamese_model.h5', custom_objects=None)

# Save the model in TensorFlow's SavedModel format
model.save('saved_model/')
model = load_model('saved_model/')


# Get the streaming URL from a YouTube video ID
def get_youtube_stream_url(video_id):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
        return info['url']  # Streaming URL

# Extract frames from a video stream
def extract_frames_from_stream(video_url, interval=1):
    cap = cv2.VideoCapture(video_url)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (int(frame_rate) * interval) == 0:
            frames.append(frame)  # Append frame to the list
        frame_count += 1
    cap.release()
    return frames

# Find the most similar frames
def find_similar_frames(reference_image, candidate_frames, top_n=5):
    reference_image = cv2.resize(reference_image, (224, 224)) / 255.0  # Normalize
    reference_image = reference_image[np.newaxis, ...]

    similarities = []
    for i, frame in enumerate(candidate_frames):
        candidate_image = cv2.resize(frame, (224, 224)) / 255.0  # Normalize
        candidate_image = candidate_image[np.newaxis, ...]
        similarity = model.predict([reference_image, candidate_image])[0][0]
        similarities.append((i, similarity))

    # Sort by similarity in descending order
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit app
st.title("Match Cutting with YouTube or Uploaded Video")

# Input for YouTube Video ID
video_id = st.text_input("Enter YouTube Video ID (must be under 10 minutes):")
frames = []

if video_id:
    st.write("Processing YouTube video on the fly...")
    stream_url = get_youtube_stream_url(video_id)
    frames = extract_frames_from_stream(stream_url, interval=1)
    st.write(f"Extracted {len(frames)} frames from the video.")

# Select and display a frame
if frames:
    st.write("Select a frame to use as the reference:")
    frame_indices = list(range(len(frames)))
    selected_frame_index = st.slider("Select a frame index:", min_value=0, max_value=len(frames) - 1, value=0)
    selected_frame = frames[selected_frame_index]
    st.image(selected_frame, caption="Selected Reference Frame", use_column_width=True)

    # Perform similarity analysis
    st.write("Finding similar frames...")
    top_matches = find_similar_frames(selected_frame, frames, top_n=5)

    # Display similar frames
    st.write("Top Similar Frames:")
    for rank, (frame_idx, similarity) in enumerate(top_matches, 1):
        st.image(frames[frame_idx], caption=f"Match {rank} - Similarity Score: {similarity:.2f}", use_column_width=True)
