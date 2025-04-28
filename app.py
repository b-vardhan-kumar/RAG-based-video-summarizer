import os
import re
import tempfile
import torch
import whisper
import yt_dlp
import streamlit as st
import nltk
import shutil
from hashlib import md5
from transformers import pipeline
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# -------------------- 1. PAGE CONFIG --------------------
st.set_page_config(page_title="RAG Driven Video Summarization with Context Aware Chatbot", layout="wide")

# -------------------- 2. NLTK SETUP --------------------
# Remove corrupt or partial punkt data if present
shutil.rmtree('/usr/local/nltk_data/tokenizers/punkt', ignore_errors=True)

# Attempt to download required NLTK resources, including 'punkt_tab'
def initialize_nltk():
    resources = ["punkt", "stopwords", "punkt_tab"]
    nltk.data.path.append('/usr/local/nltk_data')
    for resource in resources:
        try:
            if resource in ["punkt", "punkt_tab"]:
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir='/usr/local/nltk_data', quiet=True)

initialize_nltk()

# -------------------- 3. UTILITY FUNCTIONS --------------------
def sanitize_filename(name):
    return re.sub(r'[\\/*?:\"<>|]', '_', name)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower()

# -------------------- 4. CHUNK-BASED SUMMARIZATION --------------------
def chunk_text(text, max_chunk_len=800):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        sent_len = len(sentence.split())
        if current_len + sent_len > max_chunk_len:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [sentence], sent_len
        else:
            current_chunk.append(sentence)
            current_len += sent_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -------------------- 5. MODEL LOADING --------------------
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_qa_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model="google/flan-t5-large", device=device)

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# -------------------- 6. TRANSCRIPTION --------------------
def transcribe_audio(audio_path, whisper_model):
    result = whisper_model.transcribe(audio_path)
    return " ".join([seg['text'] for seg in result['segments']])

# -------------------- 7. YOUTUBE DOWNLOAD --------------------
def download_audio(url, output_dir):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
        return audio_path, info.get("title", "Unknown Title")

# -------------------- 8. STREAMLIT APP --------------------
def main():
    st.title("RAG Driven Video Summarization with Context Aware Chatbot")
    whisper_model = load_whisper_model()
    summarizer = load_summarizer()
    qa_pipeline = load_qa_model()

    if "cache" not in st.session_state:
        st.session_state.cache = {"hash": None, "transcription": None, "title": None}

    tab_input, tab_analysis = st.tabs(["Input", "Analysis"])

    # Input Tab
    with tab_input:
        source_option = st.radio("Select Input Source", ["YouTube", "File"], horizontal=True)
        content_hash = None

        if source_option == "YouTube":
            youtube_url = st.text_input("Enter YouTube URL")
            if youtube_url:
                content_hash = md5(youtube_url.encode()).hexdigest()
        else:
            uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
            if uploaded_file:
                content_hash = md5(uploaded_file.getvalue()).hexdigest()

        if content_hash and st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        if source_option == "YouTube":
                            audio_path, title = download_audio(youtube_url, tmp_dir)
                        else:
                            title = uploaded_file.name
                            audio_path = os.path.join(tmp_dir, sanitize_filename(title))
                            with open(audio_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                        transcription = transcribe_audio(audio_path, whisper_model)

                    st.session_state.cache = {"hash": content_hash, "transcription": transcription, "title": title}
                    st.success("Processing Complete!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    # Analysis Tab
    with tab_analysis:
        if st.session_state.cache.get("hash"):
            transcription = st.session_state.cache["transcription"]
            title = st.session_state.cache["title"]

            st.subheader(f"Title: {title}")
            st.text_area("Transcript", transcription, height=200)

            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    chunks = chunk_text(transcription)
                    summary_text = " ".join([summarizer(chunk, max_length=150, min_length=50, truncation=True, do_sample=False)[0]["summary_text"] for chunk in chunks])
                    st.subheader("Summary")
                    st.write(summary_text)

            question = st.text_input("Ask a question about the video content")
            if question and st.button("Get Answer"):
                with st.spinner("Analyzing question..."):
                    response = qa_pipeline(f"Question: {question} Context: {transcription[:3000]} Answer:")[0]["generated_text"]
                    st.markdown(f"**Answer:** {response}")
        else:
            st.warning("No content processed yet. Please provide input.")

if __name__ == "__main__":
    main()
