import os
import yt_dlp
import subprocess

def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    # Ensure the video and audio output folders exist
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    # Define the output template to save videos in the video folder
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download the best video and audio streams
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),  # Save with title as filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        print(f"Video downloaded to folder: {video_folder}")

    # Find the downloaded video (it may be in .mkv format or any other format)
    video_file = os.path.join(video_folder, f'{yt_dlp.YoutubeDL().extract_info(youtube_url, download=False)["title"]}.mkv')
    audio_file = os.path.join(audio_folder, f'{yt_dlp.YoutubeDL().extract_info(youtube_url, download=False)["title"]}.mp3')

    # Convert MKV to MP4 using FFmpeg
    convert_to_mp4(video_file)
    
    # Extract audio from the converted MP4
    extract_audio(video_file, audio_file)

def convert_to_mp4(video_path):
    # Convert MKV (or other formats) to MP4
    mp4_path = video_path.replace('.mkv', '.mp4')
    command = ["ffmpeg", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", mp4_path]
    subprocess.run(command)
    print(f"Video converted to: {mp4_path}")

def extract_audio(video_path, audio_path):
    # FFmpeg command to extract audio
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command)
    print(f"Audio extracted to: {audio_path}")

# Example usage
youtube_url = "https://www.youtube.com/watch?v=6sim9aF3g2c"
download_video(youtube_url)