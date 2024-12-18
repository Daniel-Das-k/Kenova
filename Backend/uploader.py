import time
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

class Uploader:
    def __init__(self, api_key=os.getenv("GOOGLE_API_KEY")):
        genai.configure(api_key=api_key)

    def upload_file(self, file_path, file_type):
        print(f"Uploading {file_type} file...")
        try:
            uploaded_file = genai.upload_file(file_path)
            print(f"Completed upload: {uploaded_file.uri}")

            if self.wait_for_processing(uploaded_file):
                file = genai.get_file(uploaded_file.uri.split("/")[-1])

                return file
            else:
                print("Error: File could not be processed.")
                return None
        except Exception as e:
            print(f"Failed to upload the {file_type} file: {e}")
            return None

    def wait_for_processing(self, uploaded_file):
        if not uploaded_file:
            print("Error: No file uploaded.")
            return False

        file_id = uploaded_file.uri.split("/")[-1]


        print("Waiting for file processing")
        while uploaded_file.state.name in ["PROCESSING", "PENDING"]:
            print('.', end='', flush=True)
            time.sleep(10)
            uploaded_file = genai.get_file(file_id)

        if uploaded_file.state.name != "ACTIVE":
            print("\nFile processing failed.")
            return False

        print("\nFile is ACTIVE.")
        return True