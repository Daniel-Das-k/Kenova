import boto3
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os
import mimetypes
import time


load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

class Uploader:
    def __init__(self, api_key=os.getenv("GOOGLE_API_KEY")):
        genai.configure(api_key=api_key)

    def get_file_stream_from_s3(self, s3_file_path):
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
   
        response = s3.get_object(Bucket="edusage-bucket", Key=s3_file_path)
        
        return io.BytesIO(response['Body'].read())
    
    def get_mime_type(self,file_name):
        mime_type, _ = mimetypes.guess_type(file_name)
        return mime_type or 'application/octet-stream'

    def upload_file_stream(self, file_stream, file_type,s3_file_path):

        print(f"Uploading {file_type} file...")
        mime = self.get_mime_type(s3_file_path)
        try:
            uploaded_file = genai.upload_file(file_stream,mime_type = mime)
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

    
# uploader = Uploader()

# s3_url = "https://s3.ap-south-1.amazonaws.com/uploads/iTNT+Hub+incubation+agreement.pdf"

# file_stream = uploader.get_file_stream_from_s3(s3_url)

# uploader.upload_file_stream(file_stream, "text")
