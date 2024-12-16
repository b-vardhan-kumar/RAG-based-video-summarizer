import os
import yt_dlp
import subprocess
import whisper

def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    """
    Downloads a YouTube video and saves it in the specified folder. Extracts audio and saves it in another folder.
    """
    # Ensure the video and audio output folders exist
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    # Define the output template to save videos in the video folder
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download the best video and audio streams
        'merge_output_format': 'mp4',  # Merge video and audio into MP4
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),  # Save video with title as filename
    }

    # Download the video using yt-dlp
    print("Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)  # Download and retrieve metadata
        video_title = info.get('title', 'video')  # Get the video title from metadata

    video_file = os.path.join(video_folder, f"{video_title}.mp4")
    audio_file = os.path.join(audio_folder, f"{video_title}.mp3")

    # Extract audio from the downloaded MP4 file
    extract_audio(video_file, audio_file)

    return audio_file  # Return the path to the extracted audio file

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a given video file using FFmpeg.
    """
    print("Extracting audio...")
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, check=True)
    print(f"Audio extracted to: {audio_path}")

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribes audio to text using Whisper.
    """
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    print("Transcription completed.")
    return result['text']

# Example usage
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=6sim9aF3g2c"  # Replace with your YouTube URL
    
    # Step 1: Download video and extract audio
    audio_file = download_video(youtube_url)

    # Step 2: Transcribe audio to text
    transcript = transcribe_audio(audio_file)

    # Step 3: Save the transcript to a text file
    transcript_file = "transcript.txt"
    with open(transcript_file, "w") as file:
        file.write(transcript)
    print(f"Transcript saved to: {transcript_file}")