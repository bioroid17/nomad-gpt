import streamlit as st
import subprocess
from pydub import AudioSegment
import math
import openai
import glob
import os

has_transcript = os.path.exists("./.cache/meetinggpt/transcripts/podcast.txt")

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.title("MeetingGPT")

st.markdown(
    """
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)


@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    if has_transcript:
        return
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]

    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunk_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]
        chunk.export(f"{chunk_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    video_path = f"./.cache/meetinggpt/videos/{video.name}"
    audio_path = f"./.cache/meetinggpt/audios/{video.name}".replace("mp4", "mp3")
    chunks_folder = "./.cache/meetinggpt/audios/chunks"
    transcript_path = f"./.cache/meetinggpt/transcripts/{video.name}".replace(
        "mp4", "txt"
    )
    with st.status("Loading video..."):
        video_content = video.read()
        with open(video_path, "wb") as f:
            f.write(video_content)
    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path, audio_path)
    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
    with st.status("Transcribing audio..."):
        transcribe_chunks(chunks_folder, transcript_path)
