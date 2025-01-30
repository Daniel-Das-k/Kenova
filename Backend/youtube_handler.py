import os
import yt_dlp
from dotenv import load_dotenv
import re

load_dotenv()

class YouTubeHandler:
    download_counter = 0  

    @staticmethod
    # def is_youtube_url(url):
    #     return bool(re.match(r"^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)", url))
    def is_youtube_url(url):
    # Updated regex to include YouTube Shorts URLs
        pattern = r"^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]+)"
        return bool(re.match(pattern, url))

            
    @classmethod
    def download_youtube_video(cls, url, output_dir="db", output_filename="downloaded_video"):
        print("Downloading YouTube video...")
        os.makedirs(output_dir, exist_ok=True)

        cls.download_counter += 1
        unique_filename = f"{output_filename}_{cls.download_counter}"

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': os.path.join(output_dir, f"{unique_filename}.%(ext)s"),
            'quiet': False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)
                absolute_path =(downloaded_file)  
                print(f"Download complete: {absolute_path}")
                return absolute_path
        except Exception as e:
            print(f"Failed to download YouTube video: {e}")
            return None
