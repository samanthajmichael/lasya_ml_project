import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub import login
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from yt_dlp import YoutubeDL

load_dotenv()
token = os.getenv('HUGGING_FACE_HUB_TOKEN')
login(token=token)

def build_base_network(input_shape):
    """Create base CNN for feature extraction"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return x

def get_youtube_stream_url(video_id):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
        return info['url']  # Streaming URL

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

def find_similar_frames(reference_image, candidate_frames, top_n=5):
    reference_image = cv2.resize(reference_image, (224, 224)) / 255.0  # Normalize
    reference_image = reference_image[np.newaxis, ...]
    
    similarities = []
    for i, frame in enumerate(candidate_frames):
        candidate_image = cv2.resize(frame, (224, 224)) / 255.0  # Normalize
        candidate_image = candidate_image[np.newaxis, ...]
        similarity = siamese_model.predict([reference_image, candidate_image])[0][0]
        similarities.append((i, similarity))
    
    # Sort by similarity in descending order
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

try:
    input_shape = (224, 224, 3)
    
    # Create inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Share base network weights for both inputs
    tower = tf.keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu')
    ])
    
    # Get embeddings
    embedding_a = tower(input_a)
    embedding_b = tower(input_b)
    
    # Compute absolute difference between embeddings
    distance = Lambda(lambda x: tf.math.abs(x[0] - x[1]))([embedding_a, embedding_b])
    
    # Add prediction layer
    output = Dense(1, activation='sigmoid')(distance)
    
    # Create model
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)
    
    # Compile model
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Download and load weights
    model_path = hf_hub_download(
        repo_id="samanthajmichael/siamese_model.h5",
        filename="siamese_model.h5"
    )
    
    # Load weights
    siamese_model.load_weights(model_path)
    st.success("Model loaded successfully!")

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
        st.image(selected_frame, caption="Selected Reference Frame", use_container_width=True)
        
        # Perform similarity analysis
        st.write("Finding similar frames...")
        top_matches = find_similar_frames(selected_frame, frames, top_n=5)
        
        # Display similar frames
        st.write("Top Similar Frames:")
        for rank, (frame_idx, similarity) in enumerate(top_matches, 1):
            st.image(frames[frame_idx], caption=f"Match {rank} - Similarity Score: {similarity:.2f}", use_container_width=True)

except Exception as e:
    st.error(f"Error loading model: {str(e)}")