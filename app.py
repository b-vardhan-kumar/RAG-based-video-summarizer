import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
import os
import yt_dlp
import subprocess
import whisper
import streamlit as st
import re  # Regular expressions for sanitizing filenames
import torch

def sanitize_filename(filename):
    """
    Sanitizes the filename by replacing special characters with underscores.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    """
    Downloads a YouTube video and extracts audio. Renames the file to avoid special characters.
    """
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download the best video and audio streams
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),  # Save with title as filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_title = info.get('title', 'video')

    # Sanitize the video title to avoid special characters in filenames
    sanitized_video_title = sanitize_filename(video_title)

    video_file = os.path.join(video_folder, f"{sanitized_video_title}.mp4")
    audio_file = os.path.join(audio_folder, f"{sanitized_video_title}.mp3")

    extract_audio(video_file, audio_file)
    return audio_file, sanitized_video_title

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file using FFmpeg.
    """
    command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribes audio to text using Whisper, optimized for GPU if available.
    """
    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = whisper.load_model(model_name, device=device)
    
    # Perform transcription
    result = model.transcribe(audio_path)
    return result['text']

def initialize_faiss_database(embedding_dim):
    """
    Initializes a FAISS database for storing embeddings.
    """
    return faiss.IndexFlatL2(embedding_dim)

def generate_embeddings(text, model_name="all-MiniLM-L6-v2"):
    """
    Generates embeddings for the given text using SentenceTransformers.
    """
    model = SentenceTransformer(model_name)
    return np.array(model.encode([text]), dtype='float32')

def add_to_faiss(index, embeddings, metadata, metadata_file="metadata.json"):
    """
    Adds embeddings and metadata to the FAISS database and saves metadata to a file.
    """
    index.add(embeddings)
    with open(metadata_file, "w") as file:
        json.dump(metadata, file)

def search_faiss(index, query_embedding, top_k=3):
    """
    Searches the FAISS database for the closest embeddings to the query.
    """
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

def load_metadata(metadata_file="metadata.json"):
    """
    Loads metadata from a JSON file.
    """
    with open(metadata_file, "r") as file:
        metadata = json.load(file)
    return metadata

def retrieve_context(index, query, metadata_file="metadata.json", model_name="all-MiniLM-L6-v2"):
    """
    Retrieves the most relevant context for a given user query.
    """
    query_embedding = generate_embeddings(query, model_name)
    indices, _ = search_faiss(index, query_embedding)
    metadata = load_metadata(metadata_file)
    results = [metadata[idx]['transcription'] for idx in indices]
    return " ".join(results)

def generate_answer_with_huggingface(context, query):
    """
    Generates an answer using Hugging Face Transformers.
    """
    model_name = "google/flan-t5-base"  # Change to a model that suits your task
    question_answering_pipeline = pipeline("text2text-generation", model=model_name)

    # Prepare input for the model
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = question_answering_pipeline(prompt, max_length=100, num_return_sequences=1)

    return response[0]["generated_text"]

# Streamlit App Interface
st.title("RAG-based Video Q&A System")
st.subheader("Enter YouTube Video URL and ask questions")

# User Input: Video URL and Query
youtube_url = st.text_input("Enter YouTube Video URL")
query = st.text_input("Ask a question about the video:")

# Add button to trigger processing
if st.button("Start Processing"):

    # Check if both fields are filled
    if youtube_url and query:
        with st.spinner('Processing...'):
            # Download and Transcribe (transcription is only used for embedding and context retrieval)
            audio_file, video_title = download_video(youtube_url)
            
            # Transcription with GPU optimization
            transcription = transcribe_audio(audio_file, model_name="base")

        # Initialize FAISS and Add Embedding
        faiss_index = initialize_faiss_database(384)
        embedding = generate_embeddings(transcription)
        add_to_faiss(faiss_index, embedding, [{"video_title": video_title, "transcription": transcription}])

        context = retrieve_context(faiss_index, query)

        # Generate answer
        answer = generate_answer_with_huggingface(context, query)
        st.write(f"Answer: {answer}")

    else:
        st.warning("Please fill in both the YouTube video URL and your query.")