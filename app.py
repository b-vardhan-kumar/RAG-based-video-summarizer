import os
import re
import tempfile
import torch
import whisper
import yt_dlp
import streamlit as st
import nltk
from hashlib import md5
from transformers import pipeline
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# -------------------- 1. PAGE CONFIG --------------------
st.set_page_config(page_title="Video Content Analyzer", layout="wide")

# -------------------- 2. NLTK SETUP --------------------
def initialize_nltk():
    """Download required NLTK data if not already present."""
    nltk_resources = ["punkt", "stopwords"]
    for resource in nltk_resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

initialize_nltk()

# -------------------- 3. UTILITY FUNCTIONS --------------------
def sanitize_filename(name: str) -> str:
    """Replaces invalid filename characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def clean_text(text: str) -> str:
    """Remove special chars and extra spaces, convert to lowercase."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\\s]", "", text)
    return text.lower()

def preprocess_words(text: str) -> str:
    """Basic text normalization: remove stopwords, apply stemming."""
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    words = re.findall(r"\\w+", text.lower())
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------- 4. CHUNK-BASED SUMMARIZATION --------------------
def chunk_text(text: str, max_chunk_len=800):
    """
    Splits a large text into chunks of ~max_chunk_len words each,
    to avoid exceeding BART's 1024-token limit.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence.split())
        if current_len + sent_len > max_chunk_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sent_len
        else:
            current_chunk.append(sentence)
            current_len += sent_len

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -------------------- 5. SUMMARIZATION & Q&A PIPELINES --------------------
@st.cache_resource
def load_summarizer():
    """Load BART summarizer (CPU or GPU)."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_qa_model():
    """Load Flan-T5 for Q&A (CPU or GPU)."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model="google/flan-t5-large", device=device)

def generate_summary(transcription: str, summarizer) -> str:
    """
    Summarize the transcription by splitting it into manageable chunks
    and combining the partial summaries.
    """
    # If text is too short, skip chunking
    if len(transcription.split()) < 50:
        return "Transcription is too short for summarization."

    chunks = chunk_text(transcription, max_chunk_len=800)
    partial_summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(
                chunk,
                max_length=150,
                min_length=50,
                truncation=True,
                do_sample=False
            )[0]["summary_text"]
            partial_summaries.append(summary)
        except Exception as e:
            st.error(f"Summarization failed on chunk: {e}")
            partial_summaries.append("")

    # Combine all chunk summaries
    final_summary = " ".join(partial_summaries)
    return final_summary

def answer_question(transcription: str, question: str, qa_pipeline) -> str:
    """
    Use Flan-T5 to answer a question based on the transcription.
    Truncates context if necessary to keep input small.
    """
    # Keep the context to ~3000 chars to avoid token limit
    context = transcription[:3000].replace("\n", " ")
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    try:
        response = qa_pipeline(
            prompt,
            max_length=150,
            num_beams=4,
            do_sample=False
        )[0]["generated_text"]
        return response
    except Exception as e:
        st.error(f"Q&A failed: {e}")
        return "No answer generated."

# -------------------- 6. WHISPER TRANSCRIPTION --------------------
@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model (base)."""
    return whisper.load_model("base")

def transcribe_audio(audio_path: str, whisper_model) -> str:
    """
    Transcribe audio using Whisper.
    Returns the transcription as a single string.
    """
    result = whisper_model.transcribe(audio_path)
    return " ".join(seg["text"] for seg in result["segments"])

def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    """
    Download audio from a YouTube URL, returning (audio_path, title).
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
        title = info.get("title", "Unknown Title")
        return audio_path, title

# -------------------- 7. STREAMLIT APP --------------------
def main():
    st.title("RAG Driven Video Summarization with Context Aware Chatbot")

    # Load all required models
    whisper_model = load_whisper_model()
    summarizer = load_summarizer()
    qa_pipeline = load_qa_model()

    # Manage app state
    if "cache" not in st.session_state:
        st.session_state.cache = {"hash": None, "transcription": None, "title": None}

    # Tabs: Input vs. Analysis
    tab_input, tab_analysis = st.tabs(["Input", "Analysis"])

    # =============== INPUT TAB ===============
    with tab_input:
        source_option = st.radio("Select Input Source", ["YouTube", "File"], horizontal=True)
        content_hash = None

        if source_option == "YouTube":
            youtube_url = st.text_input("Enter YouTube URL")
            if youtube_url:
                content_hash = md5(youtube_url.encode()).hexdigest()
        else:
            uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])
            if uploaded_file:
                content_hash = md5(uploaded_file.getvalue()).hexdigest()

        if content_hash and st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        if source_option == "YouTube":
                            audio_path, title = download_audio(youtube_url, tmp_dir)
                        else:
                            # Save uploaded file
                            title = uploaded_file.name
                            audio_path = os.path.join(tmp_dir, sanitize_filename(title))
                            with open(audio_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                        # Transcribe
                        transcription = transcribe_audio(audio_path, whisper_model)

                    # Save results in session state
                    st.session_state.cache = {
                        "hash": content_hash,
                        "transcription": transcription,
                        "title": title
                    }
                    st.success("Processing Complete!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    # =============== ANALYSIS TAB ===============
    with tab_analysis:
        if st.session_state.cache.get("hash"):
            transcription = st.session_state.cache["transcription"]
            title = st.session_state.cache["title"]

            st.subheader(f"Title: {title}")
            st.text_area("Transcript", transcription, height=200)

            # Summarize
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    summary_text = generate_summary(transcription, summarizer)
                    st.subheader("Summary")
                    st.write(summary_text)

            # Q&A
            question = st.text_input("Ask a question about the video content")
            if question:
                if st.button("Get Answer"):
                    with st.spinner("Analyzing question..."):
                        answer = answer_question(transcription, question, qa_pipeline)
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")

        else:
            st.warning("No content processed yet. Go to 'Input' tab to add content.")

if __name__ == "__main__":
    main()
