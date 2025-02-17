import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer
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
    Uses 'restrictfilenames' for safe filenames and yt-dlp's prepare_filename method
    to determine the final output file name.
    """
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'restrictfilenames': True,  # Ensure safe filenames
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(audio_folder, '%(title)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        file_path = ydl.prepare_filename(info)
        audio_file = os.path.splitext(file_path)[0] + ".mp3"
        video_title = info.get('title', 'video')

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Expected audio file '{audio_file}' not found.")

    return audio_file, video_title


def transcribe_audio(audio_path, model_name="medium"):
    """
    Transcribes audio to text using Whisper, optimized for GPU if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Running Whisper on: {device}")
    model = whisper.load_model(model_name, device=device)
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
    Validates indices to ensure they are within the bounds of the metadata list.
    """
    query_embedding = generate_embeddings(query, model_name)
    indices, _ = search_faiss(index, query_embedding)
    metadata = load_metadata(metadata_file)

    results = []
    for idx in indices:
        if idx < len(metadata):
            results.append(metadata[idx]['transcription'])
        else:
            st.warning(f"Index {idx} is out of range. Metadata has {len(metadata)} entries.")
    return " ".join(results)


def generate_answer_with_huggingface(context, query):
    """
    Generates an answer using Hugging Face Transformers.
    This function truncates the context as needed to ensure the final prompt
    does not exceed the model's maximum token length.
    """
    model_name = "google/flan-t5-large"  # Use a powerful model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("text2text-generation", model=model_name)

    static_prompt = f"Question: {query}\nAnswer:"
    static_tokens = tokenizer.encode(static_prompt, add_special_tokens=False)
    max_length = tokenizer.model_max_length

    allowed_context_tokens = max_length - len(static_tokens) - 5
    context_tokens = tokenizer.encode(context, truncation=True, max_length=allowed_context_tokens,
                                      add_special_tokens=False)
    truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)

    prompt = f"Context: {truncated_context}\nQuestion: {query}\nAnswer:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) > max_length:
        prompt_tokens = prompt_tokens[:max_length]
        prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

    response = qa_pipeline(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]


# Streamlit App Interface
st.title("RAG Driven Video Summarization with Context Aware Chatbot")
st.subheader("Enter YouTube Video URL and ask questions")

# Unique keys for each text_input to avoid duplication errors
youtube_url = st.text_input("Enter YouTube Video URL", key="video_url")
initial_query = st.text_input("Ask a question about the video:", key="initial_query")

if st.button("Start Processing", key="start_processing"):
    if youtube_url and initial_query:
        try:
            with st.spinner('Downloading audio...'):
                audio_file, video_title = download_audio(youtube_url)

            with st.spinner('Transcribing audio...'):
                transcription = transcribe_audio(audio_file, model_name="medium")

            embedding = generate_embeddings(transcription)
            add_to_faiss(st.session_state.faiss_index, embedding,
                         [{"video_title": video_title, "transcription": transcription}])
            save_faiss_index(st.session_state.faiss_index)

            # Retrieve context and generate answer for the initial query
            context = retrieve_context(st.session_state.faiss_index, initial_query)
            answer = generate_answer_with_huggingface(context, initial_query)

            st.subheader("Video Details")
            st.write(f"Title: {video_title}")
            st.subheader("Transcription")
            st.write(transcription)
            st.subheader("Answer")
            st.write(answer)

            # Mark that video processing is complete
            st.session_state.video_processed = True
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please fill in both the YouTube video URL and your query.")

# Additional query section for follow-up questions after processing
if st.session_state.get("video_processed", False):
    st.subheader("Additional Queries")
    followup_query = st.text_input("Ask another question (or type 'exit' to quit):", key="followup_query")
    if followup_query:
        if followup_query.lower() == "exit":
            st.write("Exiting additional queries.")
        else:
            context = retrieve_context(st.session_state.faiss_index, followup_query)
            answer = generate_answer_with_huggingface(context, followup_query)
            st.write(f"Answer: {answer}")

# Initialize session state if not already done
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = load_faiss_index() or initialize_faiss_database(384)
    st.session_state.metadata = load_metadata()
