import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
import os
import yt_dlp
import whisper
import streamlit as st
import re  # Regular expressions for sanitizing filenames
import torch

# Helper Functions
def sanitize_filename(filename):
    """
    Sanitizes the filename by replacing special characters with underscores.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def download_audio(youtube_url, audio_folder="audio_downloads"):
    """
    Downloads only the audio from a YouTube video and saves it as an MP3 file.
    """
    os.makedirs(audio_folder, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(audio_folder, '%(title)s.%(ext)s'),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_title = info.get('title', 'video')
    
    # Sanitize the video title to avoid special characters in filenames
    sanitized_video_title = sanitize_filename(video_title)
    audio_file = os.path.join(audio_folder, f"{sanitized_video_title}.mp3")
    return audio_file, sanitized_video_title

def transcribe_audio(audio_path, model_name="medium"):
    """
    Transcribes audio to text using Whisper, optimized for GPU if available.
    """
    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Running Whisper on: {device}")
    
    model = whisper.load_model(model_name, device=device)
    
    # Perform transcription
    result = model.transcribe(audio_path)
    return result['text']

def initialize_faiss_database(embedding_dim):
    """
    Initializes a FAISS database for storing embeddings.
    """
    return faiss.IndexFlatL2(embedding_dim)

def save_faiss_index(index, index_file="faiss_index.index"):
    """
    Saves the FAISS index to disk.
    """
    faiss.write_index(index, index_file)

def load_faiss_index(index_file="faiss_index.index"):
    """
    Loads the FAISS index from disk.
    """
    if os.path.exists(index_file):
        return faiss.read_index(index_file)
    return None

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
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            existing_metadata = json.load(file)
        existing_metadata.extend(metadata)
        metadata = existing_metadata
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
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            metadata = json.load(file)
        return metadata
    return []

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
    model_name = "google/flan-t5-large"  # Use a more powerful model
    question_answering_pipeline = pipeline("text2text-generation", model=model_name)

    # Prepare input for the model
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = question_answering_pipeline(prompt, max_length=100, num_return_sequences=1)

    return response[0]["generated_text"]

# Streamlit App Interface
st.title("RAG-based Video Q&A System")
st.subheader("Enter YouTube Video URL and ask questions")

# Initialize session state for FAISS and metadata
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = load_faiss_index() or initialize_faiss_database(384)
    st.session_state.metadata = load_metadata()

# User Input: Video URL and Query
youtube_url = st.text_input("Enter YouTube Video URL")
query = st.text_input("Ask a question about the video:")

# Add button to trigger processing
if st.button("Start Processing"):
    if youtube_url and query:
        try:
            with st.spinner('Downloading audio...'):
                audio_file, video_title = download_audio(youtube_url)
            
            with st.spinner('Transcribing audio...'):
                transcription = transcribe_audio(audio_file, model_name="medium")
            
            # Add to FAISS
            embedding = generate_embeddings(transcription)
            add_to_faiss(st.session_state.faiss_index, embedding, [{"video_title": video_title, "transcription": transcription}])
            save_faiss_index(st.session_state.faiss_index)  # Save FAISS index to disk
            
            # Retrieve context and generate answer
            context = retrieve_context(st.session_state.faiss_index, query)
            answer = generate_answer_with_huggingface(context, query)
            
            # Display results
            st.subheader("Video Details")
            st.write(f"Title: {video_title}")
            st.subheader("Transcription")
            st.write(transcription)
            st.subheader("Answer")
            st.write(answer)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please fill in both the YouTube video URL and your query.")

# Allow multiple queries
while True:
    query = st.text_input("Ask another question (or type 'exit' to quit):")
    if query.lower() == "exit":
        break
    context = retrieve_context(st.session_state.faiss_index, query)
    answer = generate_answer_with_huggingface(context, query)
    st.write(f"Answer: {answer}")
