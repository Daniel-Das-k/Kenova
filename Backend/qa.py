# import os
# import io
# from textwrap import wrap
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS 
# import chromadb
# import autogen
# import uuid
# from autogen import AssistantAgent
# from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# from autogen import config_list_from_json
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import time

# app = Flask(__name__)
# CORS(app)

# class CustomEmbeddingFunction:
#     def __init__(self, embedding_model):
#         self.embedding_model = embedding_model

#     def __call__(self, input: list) -> list:
#         return self.embedding_model.embed_documents(input)

# class QuestionPaperGenerator:
#     def __init__(self,file):
#         self.uuid = str(uuid.uuid4())
#         self.CHROMA_DB_PATH = f"./chroma_db_{self.uuid}"
#         self.CHROMA_COLLECTION = f"autogen_docs_{self.uuid}"
        
#         # Ensure database directory exists
#         try:
#             os.makedirs(self.CHROMA_DB_PATH, exist_ok=True)
#         except OSError as e:
#             print(f"Error creating ChromaDB directory: {e}")
#             raise
            
#         self.file = file
        
#         # Setup configurations
#         self.setup_config()
#         self.setup_database()
#         self.create_agents()
#         self.setup_group_chat()

#     def setup_config(self):
#         """Setup basic configurations for the agents."""
#         self.gemini_config_list = config_list_from_json(
#             "OAI_CONFIG_LIST.json",
#             filter_dict={"model": [os.getenv("MODEL")]},
#         )
        
#         self.llm_config = {
#             "config_list": self.gemini_config_list,
#             "cache_seed": 40,
#             "temperature": 0,
#             "timeout": 300,
#         }

#     def setup_database(self):
#         """Setup ChromaDB with unique collection and persistent storage"""
#         try:
#             # Initialize ChromaDB client with retry mechanism
#             max_retries = 3
#             retry_delay = 1  # seconds
            
#             for attempt in range(max_retries):
#                 try:
#                     self.chroma_client = chromadb.PersistentClient(path=self.CHROMA_DB_PATH)
#                     self.collection = self.chroma_client.get_or_create_collection(
#                         name=self.CHROMA_COLLECTION,
#                         metadata={"uuid": self.uuid}
#                     )
#                     break
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         raise
#                     print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
#                     time.sleep(retry_delay)
            
#             # Setup embeddings
#             self.embeddings = GoogleGenerativeAIEmbeddings(
#                 model=os.getenv("EMBEDDING"),
#                 google_api_key=os.getenv("GOOGLE_API_KEY"),
#             )
#             self.custom_embedding = CustomEmbeddingFunction(embedding_model=self.embeddings)
#             self.vector_db = Chroma(
#                 embedding_function=self.custom_embedding,
#                 collection_name=self.CHROMA_COLLECTION,
#                 persist_directory=self.CHROMA_DB_PATH
#             )
            
#         except Exception as e:
#             print(f"Error setting up ChromaDB: {str(e)}")
#             # Cleanup on failure
#             if hasattr(self, 'chroma_client'):
#                 try:
#                     self.chroma_client.delete_collection(self.CHROMA_COLLECTION)
#                 except:
#                     pass
#             raise

#     def create_agents(self):
#         text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])
 
#         self.ragproxyagent = RetrieveUserProxyAgent(
#             name="ragproxyagent",
#             human_input_mode="NEVER",
#             llm_config=self.llm_config,
#             code_execution_config=False,
#             retrieve_config={
#                 "model": self.gemini_config_list[0]["model"],
#                 "task": "qa",
#                 "update_context": True,
#                 "n_results": 3,
#                 "docs_path": [self.file],
#                 "chunk_token_size": 3000,
#                 "chunk_mode": "one_line",
#                 "client": self.chroma_client,
#                 "get_or_create": True,
#                 "overwrite": True,
#                 "vector_db": self.vector_db,
#                 "collection_name": self.CHROMA_COLLECTION,
#                 "embedding_function": self.custom_embedding,
#                 "custom_text_split_function": text_splitter.split_text,
#             },
#         )

#         self.json_formater = AssistantAgent(
#             name = "json_formater",
#             system_message = """You are a JSON formatting assistant. Your task is to take the structured question paper and convert it into a well-organized JSON format. 
#             The JSON should include the following structure:(Example) i should get the response only in this format and dont include any additional quotes or text.
#            {
#                 "title": "Exam Title",
#                 "instructions": [
#                     "Instruction 1",
#                     "Instruction 2",
#                     "Instruction 3",
#                     "Instruction 4",
#                     "..."
#                 ],
#                 "total_marks": 100,
#                 "sections": [
#                     {
#                     "section_title": "Short-Answer Questions",
#                     "total_marks": 20,
#                     "questions": [
#                         {
#                         "question_number": 1,
#                         "question_text": "What is the general form of a quadratic polynomial in x with real coefficients?",
#                         "marks": 2
#                         },
#                         ...
#                     ]
#                     },
#                     {
#                     "section_title": "Long-Answer Questions",
#                     "total_marks": 65,
#                     "questions": [
#                         {
#                         "question_number": 1,
#                         "question_text": "A student claims that the relationship between the zeroes and coefficients of a polynomial is only applicable to quadratic polynomials. Evaluate this claim, using examples from the text to support your reasoning.",
#                         "marks": 13
#                         },
#                         ...
#                     ]
#                     },
#                     {
#                     "section_title": "Case Study",
#                     "total_marks": 15,
#                     "background": "A group of students is preparing for a mathematics competition. They are reviewing concepts related to number theory, including prime factorization, HCF, LCM, and irrational numbers. They encounter a problem that requires them to apply these concepts in a practical scenario.",
#                     "problem_statement": "The students are tasked with organizing a school event. They need to arrange chairs in rows and columns such that each row has the same number of chairs, and each column has the same number of chairs. They also need to determine the minimum number of chairs required to accommodate a specific number of students. Additionally, they are exploring the properties of numbers and need to prove the irrationality of a given number.",
#                     "supporting_data": [
#                         "The total number of chairs available is 360.",
#                         "The number of students attending the event is 120.",
#                         "The students are also working on proving that √5 is an irrational number."
#                     ],
#                     "questions": [
#                         {
#                         "question_number": 1,
#                         "question_text": "Using the prime factorization method, find the HCF and LCM of 360 and 120. Explain how these values can help in arranging the chairs in rows and columns.",
#                         "marks": 5,
#                         },
#                       ...
#                     ]
#                     }
#                 ]
#                 }

#             Rules and Instructions for JSON Formatting:
            

#             Extract the exam title and include it as the value for "title".
#             Include all instructions as an array under "instructions".
#             The total marks for the exam should be added as the value for "total_marks".
#             For each section (e.g., Short-Answer Questions, Long-Answer Questions, Case Study), create a corresponding object in the "sections" array:
#             Use the section name as "section_title".
#             Add the total marks for the section as "total_marks".
#             Include all questions in the section under the "questions" array, with each question formatted as:
#             "question_number": The question number.
#             "question_text": The text of the question.
#             "marks": The marks allocated for the question.
#             Ensure all text is properly escaped to make the JSON valid.
#             Validate the JSON output to ensure it adheres to the structure.
#             Always maintain clear, consistent formatting. If there is any ambiguity in the question paper, use your best judgment to organize the content logically.
        
#             """,
#             llm_config=self.llm_config
#         )

#         self.final_paper  = AssistantAgent(
#             name="Question_format",
#             system_message="""Organize the following questions into a finalized question paper. The paper should include:
#             Instuctions: Provide Basic instuctions to the students to write the exam and the marks distribution.
#             1. *Short-Answer Questions (2 Marks)* focusing on Remember and Understand.(10 Questions)
#             2. *Long-Answer Questions (13 Marks)* focusing on Evaluate and Create.(5 Questions)
#             3. *Case Study (15 Marks)* focusing on Apply and Create.(1 Question)
#             Ensure the questions are categorized, formatted appropriately, and distributed according to Bloom's Taxonomy, with a logical structure.""",
#             llm_config=self.llm_config
#         )
        
#         self.short_answer = AssistantAgent(
#             name="2_marks",
#             is_termination_msg=self.termination_msg,
#             system_message="""Generate exactly 10 short-answer questions worth 2 marks each, focusing on Remember and Understand levels of Bloom's Taxonomy.
#             Each question should be directly related to the content in the document.
#             Format each question as:
#             {
#                 "question_number": <number>,
#                 "question_text": "<question>",
#                 "marks": 2
#             }
#             """,
#             llm_config=self.llm_config,
#         )
        
#         self.long_answer =AssistantAgent(
#             name="10_marks",
#             is_termination_msg=self.termination_msg,
#             system_message="""Generate exactly 5 long-answer questions worth 13 marks each, focusing on Evaluate and Create levels of Bloom's Taxonomy.
#             Each question should require detailed explanations, derivations, or critical analysis.
#             Format each question as:
#             {
#                 "question_number": <number>,
#                 "question_text": "<question>",
#                 "marks": 13
#             }
#             """,
#             llm_config=self.llm_config,
#         )
        
#         self.case_study = AssistantAgent(
#             name="case_study",
#             is_termination_msg=self.termination_msg,
#             system_message="""Generate exactly 1 case study worth 15 marks that includes:
#             {
#                 "section_title": "Case Study",
#                 "total_marks": 15,
#                 "background": "<brief scenario background>",
#                 "problem_statement": "<clear problem description>",
#                 "supporting_data": [
#                     "<data point 1>",
#                     "<data point 2>",
#                     "<data point 3>"
#                 ],
#                 "questions": [
#                     {
#                         "question_number": 1,
#                         "question_text": "<question>",
#                         "marks": <marks>
#                     }
#                 ]
#             }
#             The case study should be practical and encourage critical thinking.""",
#             llm_config=self.llm_config,
#         )

#         self.json_formater = AssistantAgent(
#             name="json_formater",
#             system_message="""Format the question paper into valid JSON. Do not add any additional text or quotes around the JSON.
#             The output should be exactly in this format:
#             {
#                 "title": string,
#                 "instructions": string[],
#                 "total_marks": number,
#                 "sections": [
#                     {
#                         "section_title": string,
#                         "total_marks": number,
#                         "questions": [
#                             {
#                                 "question_number": number,
#                                 "question_text": string,
#                                 "marks": number
#                             }
#                         ]
#                     },
#                     {
#                         "section_title": "Case Study",
#                         "total_marks": number,
#                         "background": string,
#                         "problem_statement": string,
#                         "supporting_data": string[],
#                         "questions": [
#                             {
#                                 "question_number": number,
#                                 "question_text": string,
#                                 "marks": number
#                             }
#                         ]
#                     }
#                 ]
#             }""",
#             llm_config=self.llm_config
#         )

#         self.final_paper = AssistantAgent(
#             name="Question_format",
#             system_message="""Organize the questions into a finalized question paper with exactly this structure:
#             {
#                 "title": "<subject> Examination",
#                 "instructions": [
#                     "All questions are compulsory",
#                     "<other relevant instructions>"
#                 ],
#                 "total_marks": 100,
#                 "sections": [
#                     {
#                         "section_title": "Short-Answer Questions",
#                         "total_marks": 20,
#                         "questions": [<questions from short_answer agent>]
#                     },
#                     {
#                         "section_title": "Long-Answer Questions",
#                         "total_marks": 65,
#                         "questions": [<questions from long_answer agent>]
#                     },
#                       {
#                 "section_title": "Case Study",
#                 "total_marks": 15,
#                 "background": "<brief scenario background>",
#                 "problem_statement": "<clear problem description>",
#                 "supporting_data": [
#                     "<data point 1>",
#                     "<data point 2>",
#                     "<data point 3>"
#                 ],
#                 "questions": [
#                     {
#                         "question_number": 1,
#                         "question_text": "<question>",
#                         "marks": <marks>
#                     }
#                 ]
#             }
#                 ]
#             }
#             """,
#             llm_config=self.llm_config
#         )

#     def setup_group_chat(self):
#         self.groupchat = autogen.GroupChat(
#             agents=[self.ragproxyagent, self.short_answer, self.long_answer, 
#                    self.case_study, self.final_paper,self.json_formater],
#             messages=[],
#             max_round=9,
#             speaker_selection_method="round_robin"
#         )
#         self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

#     @staticmethod
#     def termination_msg(x):
#         return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

#     def generate_paper(self):
#         self.manager.reset()
     
#         output = self.ragproxyagent.initiate_chat(
#             self.manager,
#             problem=self.get_problem(),
#             message=self.ragproxyagent.message_generator,
#             n_results=3,
#         )
        
#         final_output = None
#         for message in output.chat_history:
#             if 'content' in message:
#                 final_output = message['content']
        
#         return final_output

#     @staticmethod
#     def get_problem():
#         return """From the provided document, extract the most relevant topics and subtopics. 
#         For each topic, retrieve the corresponding explanations, definitions, or related content. 
#         Focus on identifying both *factual* and *conceptual* content that could span across 
#         different cognitive levels (e.g., factual recall, application, analysis).
#         Generate a question paper based on the provided document.

#         If Mathematical: Extract equations and concepts to create short-answer, long-answer, 
#         and problem-solving questions, ensuring balanced difficulty and a total of 100 marks.
#         If Non-Mathematical: Identify key topics, definitions, and concepts to create a mix of 
#         factual, application, and analysis questions, including diagrams and scenarios, 
#         with a total of 100 marks.
#         Ensure the format is clear and questions are categorized by type and difficulty."""

#     def __del__(self):
#         try:
#             if hasattr(self, 'chroma_client'):
#                 self.chroma_client.delete_collection(self.CHROMA_COLLECTION)
#             if os.path.exists(self.CHROMA_DB_PATH):
#                 import shutil
#                 shutil.rmtree(self.CHROMA_DB_PATH)
#         except:
#             pass

# # @app.route('/run', methods=['GET', 'POST'])
# # def run():
# #     try:
# #         file = request.files.get('file')
# #         if not file:
# #             return jsonify({"error": "No file uploaded"}), 400
            
# #         print("Running with file:", file.filename)

# #         generator = QuestionPaperGenerator(file=file.filename)

# #         final_output = generator.generate_paper()
        
# #         if not final_output:
# #             return jsonify({"error": "No output generated"}), 400
            
# #         pdf_path = os.path.join("db", "output.pdf")
# #         # generator.generate_pdf(final_output, pdf_path)
        
# #         # print(f"PDF saved to: {pdf_path}")
        
# #         return jsonify({
# #             "final_output": final_output
# #         })
        
# #     except Exception as e:
# #         print(f"Error: {str(e)}")
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, port=9090)
# # if __name__ == '__main__':
# #     app.run(debug=True, port=9090)
import os
import io
from textwrap import wrap
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import chromadb
import autogen
import uuid
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

from autogen import config_list_from_json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

app = Flask(__name__)
CORS(app)

class CustomEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input: list) -> list:
        return self.embedding_model.embed_documents(input)

class QuestionPaperGenerator:
    def __init__(self,file):
        self.uuid = str(uuid.uuid4())
        self.CHROMA_DB_PATH = f"./chroma_db_{self.uuid}"
        self.CHROMA_COLLECTION = f"autogen_docs_{self.uuid}"
        
        # Ensure database directory exists
        try:
            os.makedirs(self.CHROMA_DB_PATH, exist_ok=True)
        except OSError as e:
            print(f"Error creating ChromaDB directory: {e}")
            raise
            
        self.file = file
        
        # Setup configurations
        self.setup_config()
        self.setup_database()
        self.create_agents()
        self.setup_group_chat()

    def setup_config(self):
        """Setup basic configurations for the agents."""
        self.gemini_config_list = config_list_from_json(
            "OAI_CONFIG_LIST.json",
            filter_dict={"model": [os.getenv("MODEL")]},
        )
        
        self.llm_config = {
            "config_list": self.gemini_config_list,
            "cache_seed": 40,
            "temperature": 0,
            "timeout": 300,
        }

    def setup_database(self):
        """Setup ChromaDB with unique collection and persistent storage"""
        try:
            # Initialize ChromaDB client with retry mechanism
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    self.chroma_client = chromadb.PersistentClient(path=self.CHROMA_DB_PATH)
                    self.collection = self.chroma_client.get_or_create_collection(
                        name=self.CHROMA_COLLECTION,
                        metadata={"uuid": self.uuid}
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            # Setup embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=os.getenv("EMBEDDING"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            self.custom_embedding = CustomEmbeddingFunction(embedding_model=self.embeddings)
            self.vector_db = Chroma(
                embedding_function=self.custom_embedding,
                collection_name=self.CHROMA_COLLECTION,
                persist_directory=self.CHROMA_DB_PATH
            )
            
        except Exception as e:
            print(f"Error setting up ChromaDB: {str(e)}")
            # Cleanup on failure
            if hasattr(self, 'chroma_client'):
                try:
                    self.chroma_client.delete_collection(self.CHROMA_COLLECTION)
                except:
                    pass
            raise

    def create_agents(self):
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])
 
        self.ragproxyagent = RetrieveUserProxyAgent(
            name="ragproxyagent",
            human_input_mode="NEVER",
            llm_config=self.llm_config,
            code_execution_config=False,
            retrieve_config={
                "model": self.gemini_config_list[0]["model"],
                "task": "qa",
                "update_context": True,
                "n_results": 3,
                "docs_path": [self.file],
                "chunk_token_size": 3000,
                "chunk_mode": "one_line",
                "client": self.chroma_client,
                "get_or_create": True,
                "overwrite": True,
                "vector_db": self.vector_db,
                "collection_name": self.CHROMA_COLLECTION,
                "embedding_function": self.custom_embedding,
                "custom_text_split_function": text_splitter.split_text,
            },
        )

        self.json_formater = AssistantAgent(
            name = "json_formater",
            system_message = """You are a JSON formatting assistant. Your task is to take the structured question paper and convert it into a well-organized JSON format. 
            The JSON should include the following structure:(Example) i should get the response only in this format and dont include any additional quotes or text.
           {
                "title": "Exam Title",
                "instructions": [
                    "Instruction 1",
                    "Instruction 2",
                    "Instruction 3",
                    "Instruction 4",
                    "..."
                ],
                "total_marks": 100,
                "sections": [
                    {
                    "section_title": "Short-Answer Questions",
                    "total_marks": 20,
                    "questions": [
                        {
                        "question_number": 1,
                        "question_text": "What is the general form of a quadratic polynomial in x with real coefficients?",
                        "answer":"The general form of a quadratic polynomial in \( x \) with real coefficients is:  
                        ax^2 + bx + c 
                        where  a ,  b , and  c  are real numbers, and  a neq 0 . Here,  a  is the coefficient of  x^2 , \ b \ of \ x \, and \ c \ is the constant term."
                        "marks": 2
                        },
                        ...
                    ]
                    },
                    {
                    "section_title": "Long-Answer Questions",
                    "total_marks": 65,
                    "questions": [
                        {
                        "question_number": 1,
                        "question_text": "A student claims that the relationship between the zeroes and coefficients of a polynomial is only applicable to quadratic polynomials. Evaluate this claim, using examples from the text to support your reasoning.",
                        "answer":"1. The General Case: Relationship Between Zeroes and Coefficients
                            For any polynomial of degree .........",
                        "marks": 13
                        },
                        ...
                    ]
                    },
                    {
                    "section_title": "Case Study",
                    "total_marks": 15,
                    "background": "A group of students is preparing for a mathematics competition. They are reviewing concepts related to number theory, including prime factorization, HCF, LCM, and irrational numbers. They encounter a problem that requires them to apply these concepts in a practical scenario.",
                    "problem_statement": "The students are tasked with organizing a school event. They need to arrange chairs in rows and columns such that each row has the same number of chairs, and each column has the same number of chairs. They also need to determine the minimum number of chairs required to accommodate a specific number of students. Additionally, they are exploring the properties of numbers and need to prove the irrationality of a given number.",
                    "supporting_data": [
                        "The total number of chairs available is 360.",
                        "The number of students attending the event is 120.",
                        "The students are also working on proving that √5 is an irrational number."
                    ],
                    "questions": [
                        {
                        "question_number": 1,
                        "question_text": "Using the prime factorization method, find the HCF and LCM of 360 and 120. Explain how these values can help in arranging the chairs in rows and columns.",
                        "marks": 5,
                        },
                      ...
                    ]
                    }
                ]
                }

            Rules and Instructions for JSON Formatting:
            

            Extract the exam title and include it as the value for "title".
            Include all instructions as an array under "instructions".
            The total marks for the exam should be added as the value for "total_marks".
            For each section (e.g., Short-Answer Questions, Long-Answer Questions, Case Study), create a corresponding object in the "sections" array:
            Use the section name as "section_title".
            Add the total marks for the section as "total_marks".
            Include all questions in the section under the "questions" array, with each question formatted as:
            "question_number": The question number.
            "question_text": The text of the question.
            "marks": The marks allocated for the question.
            Ensure all text is properly escaped to make the JSON valid.
            Validate the JSON output to ensure it adheres to the structure.
            Always maintain clear, consistent formatting. If there is any ambiguity in the question paper, use your best judgment to organize the content logically.
        
            """,
            llm_config=self.llm_config
        )

        self.final_paper  = AssistantAgent(
            name="Question_format",
            system_message="""Organize the following questions into a finalized question paper. The paper should include:
            Instuctions: Provide Basic instuctions to the students to write the exam and the marks distribution.
<<<<<<< Updated upstream
            1. *Short-Answer Questions (2 Marks)* focusing on Remember and Understand.(10 Questions)
            2. *Long-Answer Questions (13 Marks)* focusing on Evaluate and Create.(5 Questions)
            3. *Case Study (15 Marks)* focusing on Apply and Create.(1 Question)
=======
            1. Short-Answer Questions (2 Marks) focusing on Remember and Understand.(10 Questions)
            2. Long-Answer Questions (13 Marks) focusing on Evaluate and Create.(5 Questions)
            3. Case Study (15 Marks) focusing on Apply and Create.(1 Question)
>>>>>>> Stashed changes
            The generated answer must be the perfect answer in the context of the question.
            
            Ensure the questions are categorized, formatted appropriately, and distributed according to Bloom's Taxonomy, with a logical structure.""",
            llm_config=self.llm_config
        )
        
        self.short_answer = AssistantAgent(
            name="2_marks",
            is_termination_msg=self.termination_msg,
            system_message="""Generate exactly 10 short-answer questions worth 2 marks each, focusing on Remember and Understand levels of Bloom's Taxonomy.
            Each question should be directly related to the content in the document.
            The answer should be more than 35 words less than 50 words, the generated answer must be the perfect answwer in the context of the question.
            The answer should be the only perfect answer that could be provided for that question.
            Format each question as:
            {
                "question_number": <number>,
                "question_text": "<question>",
                "answer": "<answer>",
                "marks": 2
                .....
            }
            """,
            llm_config=self.llm_config,
        )
        
        self.long_answer =AssistantAgent(
            name="10_marks",
            is_termination_msg=self.termination_msg,
            system_message="""Generate exactly 5 long-answer questions worth 13 marks each, focusing on Evaluate and Create levels of Bloom's Taxonomy.
            Each question should require detailed explanations, derivations, or critical analysis.
            The answer should be more than 900 words, less than 1000 words, the generated answer must be the perfect answwer in the context of the question.
            The answer shoould be the only perfect answer that could be provided.
            Format each question as:
            {
                "question_number": <number>,
                "question_text": "<question>",
                "answer": "<answer>",
                "marks": 13
                ......
            }
            """,
            llm_config=self.llm_config,
        )
        
        self.case_study = AssistantAgent(
            name="case_study",
            is_termination_msg=self.termination_msg,
            system_message="""Generate exactly 1 case study worth 15 marks that includes:
            The answer should be long, depending on the marks each question is worth.
            for Example:
                marks alloted = 15
                then, 1000 words answer should be generated
                length of the answe should be depending on the marks awarded to each question
                The answer shoould be the only perfect answer that could be provided.
            {
                "section_title": "Case Study",
                "total_marks": 15,
                "background": "<brief scenario background>",
                "problem_statement": "<clear problem description>",
                "supporting_data": [
                    "<data point 1>",
                    "<data point 2>",
                    "<data point 3>"
                ],
                "questions": [
                    {
                        "question_number": 1,
                        "question_text": "<question>",
                        "answer": <answer>
                        "marks": <marks>
                    }
                    .....
                ]
            }
            The case study should be practical and encourage critical thinking.""",
            llm_config=self.llm_config,
        )

        self.json_formater = AssistantAgent(
            name="json_formater",
            system_message="""Format the question paper into valid JSON. Do not add any additional text or quotes around the JSON.
            The output should be exactly in this format:
            {
                "title": string,
                "instructions": string[],
                "total_marks": number,
                "sections": [
                    {
                        "section_title": string,
                        "total_marks": number,
                        "questions": [
                            {
                                "question_number": number,
                                "question_text": string,
                                "answer": string,
                                "marks": number
                            }
                        ]
                    },
                    {
                        "section_title": "Case Study",
                        "total_marks": number,
                        "background": string,
                        "problem_statement": string,
                        "supporting_data": string[],
                        "questions": [
                            {
                                "question_number": number,
                                "question_text": string,
                                "answer": string,
                                "marks": number
                            }
                        ]
                    }
                ]
            }""",
            llm_config=self.llm_config
        )

        self.final_paper = AssistantAgent(
            name="Question_format",
            system_message="""Organize the questions into a finalized question paper with exactly this structure:
            {
                "title": "<subject> Examination",
                "instructions": [
                    "All questions are compulsory",
                    "<other relevant instructions>"
                ],
                "total_marks": 100,
                "sections": [
                    {
                        "section_title": "Short-Answer Questions",
                        "total_marks": 20,
                        "questions": [<questions from short_answer agent>]
                    },
                    {
                        "section_title": "Long-Answer Questions",
                        "total_marks": 65,
                        "questions": [<questions from long_answer agent>]
                    
                    },
                      {
                "section_title": "Case Study",
                "total_marks": 15,
                "background": "<brief scenario background>",
                "problem_statement": "<clear problem description>",
                "supporting_data": [
                    "<data point 1>",
                    "<data point 2>",
                    "<data point 3>"
                ],
                "questions": [
                    {
                        "question_number": 1,
                        "question_text": "<question>",
                        "marks": <marks>
                    }
                    ......
                ]
            }
                ]
            }
            """,
            llm_config=self.llm_config
        )

    def setup_group_chat(self):
        self.groupchat = autogen.GroupChat(
            agents=[self.ragproxyagent, self.short_answer, self.long_answer, 
                   self.case_study, self.final_paper,self.json_formater],
            messages=[],
            max_round=9,
            speaker_selection_method="round_robin"
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

    @staticmethod
    def termination_msg(x):
        return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

    def generate_paper(self):
        self.manager.reset()
     
        output = self.ragproxyagent.initiate_chat(
            self.manager,
            problem=self.get_problem(),
            message=self.ragproxyagent.message_generator,
            n_results=3,
        )
        
        final_output = None
        for message in output.chat_history:
            if 'content' in message:
                final_output = message['content']
        
        return final_output

    @staticmethod
    def get_problem():
        return """From the provided document, extract the most relevant topics and subtopics. 
        For each topic, retrieve the corresponding explanations, definitions, or related content. 
        Focus on identifying both factual and conceptual content that could span across 
        different cognitive levels (e.g., factual recall, application, analysis).
        Generate a question paper based on the provided document.

        If Mathematical: Extract equations and concepts to create short-answer, long-answer, 
        and problem-solving questions, ensuring balanced difficulty and a total of 100 marks.
        If Non-Mathematical: Identify key topics, definitions, and concepts to create a mix of 
        factual, application, and analysis questions, including diagrams and scenarios, 
        with a total of 100 marks.
        Ensure the format is clear and questions are categorized by type and difficulty."""

    def __del__(self):
        try:
            if hasattr(self, 'chroma_client'):
                self.chroma_client.delete_collection(self.CHROMA_COLLECTION)
            if os.path.exists(self.CHROMA_DB_PATH):
                import shutil
                shutil.rmtree(self.CHROMA_DB_PATH)
        except:
            pass

# @app.route('/run', methods=['GET', 'POST'])
# def run():
#     try:
#         file = request.files.get('file')
#         if not file:
#             return jsonify({"error": "No file uploaded"}), 400
            
#         print("Running with file:", file.filename)

#         generator = QuestionPaperGenerator(file=file.filename)

#         final_output = generator.generate_paper()
        
#         if not final_output:
#             return jsonify({"error": "No output generated"}), 400
            
#         pdf_path = os.path.join("db", "output.pdf")
#         # generator.generate_pdf(final_output, pdf_path)
        
#         # print(f"PDF saved to: {pdf_path}")
        
#         return jsonify({
#             "final_output": final_output
#         })
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if _name_ == '_main_':
#     app.run(debug=True, port=9090)
# if _name_ == '_main_':
#     app.run(debug=True, port=9090)