from file_handler import FileHandler
from youtube_handler import YouTubeHandler
from uploader import Uploader
import google.generativeai as genai
import os
from dotenv import load_dotenv
import ast
import re
import time

load_dotenv()

class ContentProcessor:
    def __init__(self, api_key=os.getenv("GOOGLE_API_KEY"), model_name="gemini-2.0-flash-exp", cursor=None, conn=None):
        self.file_handler = FileHandler()
        self.youtube_handler = YouTubeHandler()
        self.uploader = Uploader(api_key)
        self.model = genai.GenerativeModel(model_name)
        self.cursor = cursor
        self.conn = conn

    def upload_file(self, file_path):
        yt_path = None
        if self.youtube_handler.is_youtube_url(file_path):
            print("Detected YouTube video.")
            file_path = self.youtube_handler.download_youtube_video(file_path, output_dir="db")
            yt_path = file_path
            if not file_path:
                return None

        file_type = self.file_handler.determine_file_type(file_path)
        print(file_type)
        if not file_type:
            print("Error: Unsupported file type.")
            return None

        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}. Please check the path and try again.")
            return None

        uploaded_file = self.uploader.upload_file(file_path, file_type)
        upload_id = uploaded_file.uri.split("/")[-1]
        return uploaded_file, file_type, file_path,upload_id

    def fetch_history(self, file_path, file_type):
        if self.cursor is None:
            return []

        self.cursor.execute(
            "SELECT prompt, response FROM responses WHERE file_path = %s ORDER BY id DESC LIMIT 5",
            (file_path,)
        )
        return self.cursor.fetchall()

    def load_old_history(self, history):
        input_history = []
        for record in history[2:]:
            input_history.append({"User": record[0], "AI": record[1]})
        return input_history

    def clean_and_parse_output(self, llm_output):
        try:
            match = re.search(r'\[.*\]', llm_output, re.DOTALL)
            if match:
                json_like_content = match.group(0)
                return ast.literal_eval(json_like_content)
            else:
                print("Error: Valid JSON-like content not found in the output.")
                return None
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return None

    def summarize_content(self, input_history):
        prompt_template_summarize=f'''
            You are an assistant tasked with summarizing chat conversations while retaining specific details and context. I will provide you with a list of conversations between two roles: User(Human) and AI. Each entry contains a prompt from User and respective response from AI. Your task is to:
                - Summarize all the content spoken by the User into a single entry under the role "User," ensuring that all specific details and topics mentioned are accurately represented without generalization and summarized it within two to three lines.
                - Summarize all the content spoken by the AI into a single entry under the role "AI," ensuring the key responses are preserved with specific references to the User's input and summarized it within two to three lines.
                - Return **only** the summarized content in the form of a Python list of dictionaries, where:
                - Each dictionary contains two keys: "role" (User or AI) and "content" (summarized content for the respective role).
                - Do **not** include any extra text, such as "Output List," "json," or any additional prefixes, explanations, or formatting.
            The output must strictly start with a `[` and end with a `]`.
            Input List:
            {input_history}
            '''
        summarized_history = self.model.generate_content([prompt_template_summarize])
        summarized_history = self.clean_and_parse_output(summarized_history.text)
        return summarized_history

    def process_prompt(self, uploaded_file, prompt, file_type, full_file_path):
        if not uploaded_file:
            print("Error: No file provided.")
            return None
        try:
            get_file_start=time.time()
            # file_id = uploaded_file.uri.split("/")[-1]
            # uploaded_file = genai.get_file(file_id)
            print(f"Get file processing time: {time.time() - get_file_start:.4f} seconds")

        except Exception as e:
            print(f"Failed to fetch the updated file state: {e}")
            return None

        if uploaded_file.state.name != "ACTIVE":
            print("Error: File is not in ACTIVE state.")
            return None
        fetch_history_time=time.time()
        history = self.fetch_history(file_path=full_file_path, file_type=file_type)
        print(f"fetch history processing time: {time.time() - fetch_history_time:.4f} seconds")
        prompt_history = []
        if len(history)>3:

            self.load_old_history(history)
            summary=self.summarize_content(history)
            prompt_history.append(summary)

            for i in history[:2]:
                print(i)
                prompt_history.append({"User":i[0],"AI":i[1]})
            print(prompt_history)

        else:
            for i in history:
                prompt_history.append({"User":i[0],"AI":i[1]})
                print(prompt_history)

        prompt_template = self.get_prompt_template(file_type, prompt, prompt_history)

        try:
            response_time=time.time()
            response = self.model.generate_content([prompt_template, uploaded_file])
            print(f"response processing time: {time.time() - response_time:.4f} seconds")

            return response.text
        except Exception as e:
            print(f"Failed to make inference request: {e}")
            return None

    def get_prompt_template(self, file_type, prompt, prompt_history):
        if file_type == "video":
            return f"""
            I will provide a video file along with a question related to it. The video will primarily be in any  Indian language and is intended for educational and study purposes.
                Your task is to act as a knowledgeable and supportive teacher. When answering the question:
                - Analyze the visual content, such as images, scenes, objects, or any text visible in the video.
                - Analyze the audio content, such as speech, narration, or sounds in the video.
                Provide a detailed, accurate, and contextually relevant response within three lines. Make your explanation:
                - Clear and easy to understand for students.
                - Encouraging and insightful, offering additional context or knowledge where helpful.
                - Engaging, using examples from the video content to enhance learning.
                Additionally, consider the following conversation history between the User and the AI:
                {prompt_history}
                Now answer the following question considering the above context:
                Question: {prompt}
            """
        elif file_type == "image":
            return f"""
            I will provide an image along with a question related to it. The image may include visual elements such as objects, text, scenes, or symbols, and it is intended for educational and study purposes.
                Your task is to act as a knowledgeable and supportive teacher. When answering the question:
                - Analyze the image thoroughly, considering all visible details such as objects, colors, text, actions, or context.
                - Provide an answer that is:
                - Clear and simple for students to grasp.
                - Encouraging and explanatory, adding relevant details or insights when appropriate.
                - Engaging, using observations from the image to make the response meaningful for learning.
                - Your response should be within three lines in short and should convey the answer.
                Additionally, consider the following conversation history between the User and the AI:
    {prompt_history}
    Now answer the following question considering the above context:
    Question: {prompt}
            """
        elif file_type == "text":
            return f"""
            I will provide a text file along with a question related to its content. The text file may include written information, such as paragraphs, bullet points, or structured data, and it is intended for educational and study purposes.
                Your task is to act as a knowledgeable and supportive teacher. When answering the question:
                - Read and analyze the content of the text file thoroughly.
                - Provide an answer that is:
                - Accurate and relevant, addressing the question based solely on the text content.
                - Clear and detailed, breaking down complex ideas for better understanding.
                - Encouraging and insightful, offering logical reasoning and additional context to support the student’s learning.- 
                - Your response should be within three lines in short and should convey the answer.
                Additionally, consider the following conversation history between the User and the AI:
                {prompt_history}
                Now answer the following question considering the above context:
                Question: {prompt}
            """
