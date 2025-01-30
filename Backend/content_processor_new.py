
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import autogen
import PyPDF2
from aws import upload_file_to_s3, retrieve_s3_file_content
import hashlib
import time
from autogen import config_list_from_json
from file_handler import FileHandler
from youtube_handler import YouTubeHandler
from uploader import Uploader
import re
import ast

class ContentProcessorNew:
    def __init__(self, api_key, cursor=None, conn=None):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.cursor = cursor
        self.conn = conn
        self.file_handler = FileHandler()
        self.youtube_handler = YouTubeHandler()
        self.uploader = Uploader(api_key)
        self.llm_config = {
        "config_list": self._load_gemini_config(),
        "seed": 53,
        "temperature": 0.7,  # Adjust for creativity
        "timeout": 300,
    }
        
    def _load_gemini_config(self):
        
    # Load LLM configuration
        gemini_config_list = config_list_from_json(
            "OAI_CONFIG_LIST.json",
            filter_dict={"model": ["gemini-2.0-flash-exp"]},
        )
        return gemini_config_list
    

    def _generate_file_hash(self, file_path):
        """Generate a unique hash for the file content"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]  # Use first 12 characters for readability

    def _check_existing_summary(self, file_hash):
        """Check if summary already exists for this file"""
        if self.cursor:
            self.cursor.execute(
                "SELECT summary_path FROM lab_documents WHERE file_hash = %s",
                (file_hash,)
            )
            result = self.cursor.fetchone()
            return result[0] if result else None
        return None

    def process_pdf(self, input_pdf_path):
        """Process PDF and return paths to generated files"""
        try:
            # Generate file hash
            file_hash = self._generate_file_hash(input_pdf_path)
            
            # Check if summary exists
            existing_summary = self._check_existing_summary(file_hash)
            if existing_summary:
                
                return {
                    "summary_pdf_s3": existing_summary,
                    "file_hash": file_hash,
                    "new_summary": False
                }

            # Generate new summary
            # Upload original PDF to S3
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            s3_original_path = f"lab_docs/original_{file_hash}_{timestamp}.pdf"
            
            s3_pdf_url = upload_file_to_s3(input_pdf_path, "edusage-bucket", s3_original_path)
            
            # Generate summary
            summary = self._generate_summary(input_pdf_path)
            
            # Create and upload summary PDF
            summary_pdf_path = f"temp_summary_{file_hash}_{timestamp}.pdf"
            self._create_summary_pdf(summary, summary_pdf_path)
            
            
            s3_summary_path = f"lab_docs/summary_{file_hash}_{timestamp}.pdf"
            s3_summary_url = upload_file_to_s3(summary_pdf_path, "edusage-bucket", s3_summary_path)
            
            # Clean up local summary PDF
            if os.path.exists(summary_pdf_path):
                os.remove(summary_pdf_path)
            
            return {
                "original_pdf_s3": s3_pdf_url,
                "summary_pdf_s3": s3_summary_url,
                "summary_text": summary,
                "file_hash": file_hash,
                "new_summary": True
            }
        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            return None

    def _generate_summary(self, pdf_path):
        """Generate summary using Gemini"""
        uploaded_file = genai.upload_file(pdf_path)
        prompt = """Analyze this lab experiment/document and provide a comprehensive summary focusing on:
        1. Experiment objectives and theoretical background
        2. Key procedures and methodologies
        3. Important observations and results
        4. Critical analysis and conclusions
        5. Safety precautions and important notes
        Provide a clear, structured summary suitable for study purposes."""
        
        response = self.model.generate_content([prompt, uploaded_file])
        return response.text

    def _create_summary_pdf(self, content, output_path):
        """Create PDF from summary content"""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from io import BytesIO
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        y = height - 50
        for line in content.split('\n'):
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 12
            
        c.save()
        buffer.seek(0)
        
        with open(output_path, "wb") as output_file:
            output_file.write(buffer.read())

    def process_prompt(self, uploaded_file, prompt, file_type, full_file_path):
        """Process general prompts for AI assistant"""
        if not uploaded_file:
            print("Error: No file provided.")
            return None
        try:
            get_file_start = time.time()
            print(f"Get file processing time: {time.time() - get_file_start:.4f} seconds")

            if uploaded_file.state.name != "ACTIVE":
                print("Error: File is not in ACTIVE state.")
                return None

            fetch_history_time = time.time()
            history = self.fetch_history(file_path=full_file_path, file_type=file_type)
            print(f"fetch history processing time: {time.time() - fetch_history_time:.4f} seconds")
            
            prompt_history = []
            if len(history) > 3:
                # Load and summarize old history
                old_history = self.load_old_history(history)
                summary = self.summarize_content(old_history)
                prompt_history.append(summary)

                # Add recent history
                for i in history[:2]:
                    prompt_history.append({"User": i[0], "AI": i[1]})
            else:
                for i in history:
                    prompt_history.append({"User": i[0], "AI": i[1]})

            prompt_template = self.get_prompt_template(file_type, prompt, prompt_history)
            response_time = time.time()
            response = self.model.generate_content([prompt_template, uploaded_file])
            print(f"response processing time: {time.time() - response_time:.4f} seconds")

            return response.text
        except Exception as e:
            print(f"Failed to process prompt: {e}")
            return None

    def load_old_history(self, history):
        """Load and format old chat history"""
        input_history = []
        for record in history[2:]:
            input_history.append({"User": record[0], "AI": record[1]})
        return input_history

    def clean_and_parse_output(self, llm_output):
        """Clean and parse LLM output"""
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
        """Summarize chat history"""
        prompt_template_summarize = f'''
            You are an assistant tasked with summarizing chat conversations while retaining specific details and context. I will provide you with a list of conversations between two roles: User(Human) and AI. Each entry contains a prompt from User and respective response from AI. Your task is to:
                - Summarize all the content spoken by the User into a single entry under the role "User," ensuring that all specific details and topics mentioned are accurately represented without generalization and summarized it within two to three lines.
                - Summarize all the content spoken by the AI into a single entry under the role "AI," ensuring the key responses are preserved with specific references to the User's input and summarized it within two to three lines.
                - Return only the summarized content in the form of a Python list of dictionaries, where:
                - Each dictionary contains two keys: "role" (User or AI) and "content" (summarized content for the respective role).
                - Do not include any extra text, such as "Output List," "json," or any additional prefixes, explanations, or formatting.
            The output must strictly start with a [ and end with a ].
            Input List:
            {input_history}
            '''
        summarized_history = self.model.generate_content([prompt_template_summarize])
        return self.clean_and_parse_output(summarized_history.text)

    def get_prompt_template(self, file_type, prompt, prompt_history):
        """Get appropriate prompt template based on file type"""
        if file_type == "video":
            return f"""
            I will provide a video file along with a question related to it. The video will primarily be in any Indian language and is intended for educational and study purposes.
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
        else:  # text/pdf
            return f"""
            I will provide a text file along with a question related to its content. The text file may include written information, such as paragraphs, bullet points, or structured data, and it is intended for educational and study purposes.
                Your task is to act as a knowledgeable and supportive teacher. When answering the question:
                - Read and analyze the content of the text file thoroughly.
                - Provide an answer that is:
                - Accurate and relevant, addressing the question based solely on the text content.
                - Clear and detailed, breaking down complex ideas for better understanding.
                - Encouraging and insightful, offering logical reasoning and additional context to support the student's learning.
                - Your response should be within three lines in short and should convey the answer.
                Additionally, consider the following conversation history between the User and the AI:
                {prompt_history}
                Now answer the following question considering the above context:
                Question: {prompt}
            """

    def fetch_history(self, file_path, file_type):
        """Fetch chat history from database"""
        if self.cursor is None:
            return []

        self.cursor.execute(
            "SELECT prompt, response FROM responses WHERE file_path = %s ORDER BY id DESC LIMIT 5",
            (file_path,)
        )
        return self.cursor.fetchall()

    def upload_file(self, file_path, s3_file_path):
        """Upload file to S3 and Gemini"""
        
        if self.youtube_handler.is_youtube_url(file_path):
            print("Detected YouTube video.")
            dir_yt= re.sub(r'[^a-zA-Z0-9]', '_', file_path)
            s3_file_path = self.youtube_handler.download_youtube_video(file_path, output_filename=dir_yt)
            print("s3_file_path after downlaodiung video-",s3_file_path)
            file_path=upload_file_to_s3(s3_file_path, "edusage-bucket", s3_file_path)
            
            print("file_path after uploading it to the aws-",file_path)
            file_type="video"
            if not file_path:
                return None
        else: 
            file_type = self.file_handler.determine_file_type(file_path)
            s3_file_path=file_path
            file_path=upload_file_to_s3(s3_file_path, "edusage-bucket", s3_file_path)
            
            if not file_type:
                print("Error: Unsupported file type.")
                return None
        
        file_stream = self.uploader.get_file_stream_from_s3(s3_file_path)
        print("s3 path before upload file in genai-",s3_file_path)
        
        uploaded_file = self.uploader.upload_file_stream(file_stream, file_type, s3_file_path)
        upload_id = uploaded_file.uri.split("/")[-1]
        return uploaded_file, file_type, file_path, upload_id

    def generate_revision_notes(self, summary_s3_path, file_hash):
        """Generate structured revision notes"""
        try:
            # Check if revision notes already exist
            # if self.cursor:
            #     self.cursor.execute(
            #         "SELECT content FROM lab_notes WHERE file_hash = %s AND note_type = 'revision'",
            #         (file_hash,)
            #     )
            #     existing_notes = self.cursor.fetchone()
            #     if existing_notes:
            #         return json.loads(existing_notes[0])
            print("summary path--",summary_s3_path)
            # Generate new notes
            summ_get_path = re.search(r'(?<=amazonaws\.com\/).*',summary_s3_path).group(0)
            # Generate new notes
            summary_content = retrieve_s3_file_content("edusage-bucket", summ_get_path)
            if not summary_content:
                return None
            print(type(summary_content))
            if not summary_content:
                return None
            import pdfplumber
            from io import BytesIO
            
            text = ""
            with pdfplumber.open(BytesIO(summary_content)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    
            r_agents = self._setup_revision_agents()
            print("the revision agents",r_agents)
            result = self._run_revision_workflow(agents=r_agents,content=text)
            return result
        except Exception as e:
            print(f"Error in generate_revision_notes: {str(e)}")
            return None

    def generate_smart_notes(self, summary_s3_path, file_hash):
        """Generate enhanced smart notes"""
        try:
            print("file hash -",file_hash)                    
            summ_get_path = re.search(r'(?<=amazonaws\.com\/).*',summary_s3_path).group(0)
            # Generate new notes
            summary_content = retrieve_s3_file_content("edusage-bucket", summ_get_path)
            print(type(summary_content))
            if not summary_content:
                return None
            import pdfplumber
            from io import BytesIO
            
            text = ""
            with pdfplumber.open(BytesIO(summary_content)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""  # Handle pages with no text

            # Generate new notes
            agents = self._setup_smart_notes_agents()
            result = self._run_smart_notes_workflow(agents, text)
            print("result before returning ",result)
            return result
        except Exception as e:
            print(f"Error in generate_smart_notes: {str(e)}")
            return None

    def _setup_revision_agents(self):
        """Setup agents for revision notes"""
        initializer = autogen.UserProxyAgent(
            name="Initializer",
            human_input_mode="NEVER",  # Automate the process without human input
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda msg: "json" in msg["content"].lower(),  # Termination condition
        )

        # Agent 1: Concept Simplification & High-Yield Extraction Agent
        concept_simplification_agent = autogen.AssistantAgent(
            name="Concept_Simplification_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in simplifying complex concepts and extracting high-yield information. Break down each topic in the given lesson into easy-to-understand explanations. Focus on the most important definitions, formulas, and key points. Filter out less relevant details to keep the content concise for quick revision. Respond with simplified concepts and high-yield information for all topics in the lesson.""",
        )

        # Agent 2: Memory Booster & Mnemonics Agent
        memory_booster_agent = autogen.AssistantAgent(
            name="Memory_Booster_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in creating memory-enhancing tools. For each topic in the lesson:
        1. Convert complex concepts into easy-to-remember mnemonics, acronyms, or keywords.
        2. Highlight difficult-to-remember points using memory-enhancing techniques.
        3. Provide active recall prompts to reinforce learning.
        Ensure the mnemonics and memory aids are creative, practical, and easy to recall. Respond with memory tools for all topics in the lesson.""",
        )

        # Agent 3: Smart Comparison & Concept Linker Agent
        comparison_linker_agent = autogen.AssistantAgent(
            name="Comparison_Linker_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in comparing and linking concepts. For each topic in the lesson:
        1. Create side-by-side comparisons for confusing topics (e.g., mitosis vs. meiosis, classical vs. operant conditioning) using tables and color-coded highlights.
        2. Map related concepts across different chapters or subjects to show how they interconnect.
        3. Provide flowcharts or mind maps for better visualization.
        Ensure the comparisons and links are clear, concise, and visually intuitive. Respond with comparisons and concept maps for all topics in the lesson.""",
        )

        # Agent 4: Personalized Weak Spot & Last-Minute Refresher Agent
        weak_spot_refresher_agent = autogen.AssistantAgent(
            name="Weak_Spot_Refresher_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in identifying weak spots and providing last-minute revision tools. For each topic in the lesson:
        1. Track which sections the student spends the most time on and suggest focused revision for weak areas.
        2. Generate a one-page cheat sheet with the most crucial points before an exam.
        3. Provide 3 levels of revision:
        - Ultra-Short Summary (30-sec read)
        - Key Facts & Takeaways (2-min read)
        - Detailed Notes (for deeper understanding).
        Ensure the revision tools are concise, actionable, and tailored to the student's needs. Respond with personalized revision materials for all topics in the lesson.""",
        )

        # Agent 5: Revision Notes Structuring Agent
        revision_notes_agent = autogen.AssistantAgent(
            name="Revision_Notes_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in structuring revision notes for an entire lesson. Use the simplified concepts, memory tools, comparisons, and revision materials provided by the other agents to create a structured JSON format. The JSON should include:
        - "lesson_title": The title of the lesson.
        - "lesson_summary": A brief 2-3 line summary of the entire lesson.
        - "topics": A list of topics, each containing:
        - "topic_name": The name of the topic.
        - "summary": A brief 1-2 line summary of the topic.
        - "simplified_concept": The simplified explanation of the topic.
        - "key_points": A list of key points for the topic.
        - "memory_tools": A list of mnemonics, acronyms, or memory aids.
        - "comparisons": A list of side-by-side comparisons or concept links.
        - "revision_materials": A list of ultra-short summaries, key facts, and detailed notes.
        Ensure the JSON is well-structured, easy to read, and optimized for a clean and intuitive user experience. Respond only with the JSON structure.""",
        )
        print("finished setup for revise")
        return [initializer,concept_simplification_agent,memory_booster_agent,comparison_linker_agent,weak_spot_refresher_agent,revision_notes_agent]
    
    def _setup_smart_notes_agents(self):
        """Setup agents for smart notes"""
        initializer = autogen.UserProxyAgent(
            name="Initializer",
            human_input_mode="NEVER",  # Automate the process without human input
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda msg: "json" in msg["content"].lower(),  # Termination condition
        )

        concept_simplification_agent = autogen.AssistantAgent(
            name="Concept_Simplification_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in simplifying complex concepts. Break down each topic in the given lesson into easy-to-understand explanations. Use simple language and avoid jargon. Provide a clear and concise explanation for each topic. Respond with the simplified concepts for all topics in the lesson.""",
        )
        # Real-Time Examples Agent
        examples_agent = autogen.AssistantAgent(
            name="Examples_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in providing real-time examples to help understand concepts. For each topic in the lesson, provide relatable examples that make the concept easier to grasp. Ensure the examples are practical and relevant to real-life scenarios. Respond with examples for all topics in the lesson.""",
        )

        # Mnemonics Agent
        mnemonics_agent = autogen.AssistantAgent(
            name="Mnemonics_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in creating mnemonics for memorizing concepts. For each topic in the lesson, create memory aids (mnemonics) that help students remember key points. Ensure the mnemonics are creative and easy to recall. Respond with mnemonics for all topics in the lesson.""",
        )

        # Exam Tips Agent
        exam_tips_agent = autogen.AssistantAgent(
            name="Exam_Tips_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in providing exam-oriented tips and tricks that are highly specific to the content provided. For each topic in the lesson, analyze the simplified concepts, real-time examples, and mnemonics generated by the other agents, and provide tailored tips on:
        1. How to study effectively: Suggest study methods or resources that align with the specific concepts and examples.
        2. Common mistakes to avoid: Highlight errors students often make related to the specific topic or examples.
        3. Strategies for answering exam questions: Provide techniques for tackling questions that are likely to appear based on the topic's key points and examples.

        Ensure the tips are:
        - Directly tied to the content provided (simplified concepts, examples, and mnemonics).
        - Actionable and practical for students.
        - Specific to the topic, avoiding generic advice.

        Respond with exam tips for all topics in the lesson, ensuring each tip is relevant to the content provided.""",
        )

        # Smart Notes Structuring Agent
        smart_notes_agent = autogen.AssistantAgent(
            name="Smart_Notes_Agent",
            llm_config=self.llm_config,
            system_message="""You are an expert in structuring smart notes for an entire lesson. Use the simplified concepts, real-time examples, mnemonics, and exam tips provided by the other agents to create a structured JSON format. The JSON should include:
        - "lesson_title": The title of the lesson.
        - "lesson_summary": A brief 2-3 line summary of the entire lesson.
        - "topics": A list of topics, each containing:
        - "topic_name": The name of the topic.
        - "summary": A brief 1-2 line summary of the topic.
        - "simplified_concept": The simplified explanation of the topic.
        - "key_points": A list of key points for the topic.
        - "real_time_examples": A list of examples, each with a "description" and "application".
        - "mnemonics": A list of mnemonics, each with a "phrase" and "explanation".
        - "exam_tips": A list of tips for studying and answering exam questions.
        

        Ensure the JSON is well-structured, easy to read, and optimized for a clean and intuitive user experience. Respond only with the JSON structure.""",
        )
        
        return [initializer,concept_simplification_agent,examples_agent, mnemonics_agent,exam_tips_agent,smart_notes_agent]

   