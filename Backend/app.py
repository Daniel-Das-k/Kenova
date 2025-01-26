from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from content_processor import ContentProcessor
import os
import psycopg2
import re
from dotenv import load_dotenv
import google.generativeai as genai
import openai
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from qa import QuestionPaperGenerator
from image_case import ImageCaseStudyGenerator
from aws import upload_file_to_s3, insert_file_path_to_rds,retrieve_s3_file_content
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

load_dotenv()

app = Flask(__name__)
CORS(app)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:DanielDas2004@edusage-database.cp6gyg0soaec.ap-south-1.rds.amazonaws.com:5432/edusage-database")

ALLOWED_EXTENSIONS = {'pdf'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

required_env_vars = ["MODEL", "GOOGLE_API_KEY", "EMBEDDING"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

processor = ContentProcessor(api_key=os.getenv("GOOGLE_API_KEY"), cursor=cursor, conn=conn)

cache = {}

def process_file(file_path):
    image_dir = "image/"
    print("Image directory:", image_dir)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if os.path.exists(image_dir) and os.path.isdir(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            image_path = os.path.join(image_dir, image_files[4])
            print(f"Using image file: {image_path}")
            image_generator = ImageCaseStudyGenerator(file_path, image_path)
            print("Generating Image QA")
            return image_generator.generate_paper(image_path)
    
    print("No images found, generating text-based questions")
    text_generator = QuestionPaperGenerator(file_path)
    print("Generating QA")
    return text_generator.generate_paper()


def process_file_request(file_path, s3_file_path, prompt):
    global cache
    try:
        if not file_path:
            return {"error": "file_path is required"}, 400

        print(f"Processing file: {file_path}")
        print(f"S3 File Path: {s3_file_path}")

        # Validate AWS credentials
        if not (os.getenv("AWS_ACCESS_KEY") and os.getenv("AWS_SECRET_KEY")):
            return {"error": "AWS credentials are not configured"}, 500

        if file_path in cache:
            cached_data = cache[file_path]
            uploaded_file = cached_data["uploaded_file"]
            file_type = cached_data["file_type"]
            full_file_path = cached_data["full_file_path"]
            print("Reusing previously processed file.")
        else:
            def is_youtube_url(url):
            
                return re.match(r"(www\\.)?(youtube\\.com|youtu\\.be)/.+", url)

            if is_youtube_url(file_path):
                file_type = "video"
                sanitized_url = re.sub(r'[^a-zA-Z0-9]', '_', file_path)
                half_path = r"AI"
                check_file_path = half_path + f"/db/downloaded_video_{sanitized_url}.mp4"
            else:
                file_type = processor.file_handler.determine_file_type(file_path)
                check_file_path = file_path

            cursor.execute("SELECT file_path, uploaded, upload_id FROM files WHERE file_path = %s", (check_file_path,))
            existing_file = cursor.fetchone()

            if existing_file:
                print("Reusing previously processed file.")
                full_file_path = existing_file[0]
                upload_id = existing_file[2]
                uploaded_file = genai.get_file(upload_id)
            else:
                print("Processing new file.")
                print(file_path,s3_file_path,"DEIOIIIIIIII")
                uploaded_file, file_type, full_file_path, upload_id = processor.upload_file(file_path,s3_file_path=s3_file_path)
                if not uploaded_file:
                    return {"error": "File upload/processing failed"}, 500

                cursor.execute(
                    "INSERT INTO files (file_path, file_type, uploaded, upload_id) VALUES (%s, %s, %s, %s)",
                    (file_path, file_type, True, upload_id)
                )
                conn.commit()

            cache[file_path] = {
                "uploaded_file": uploaded_file,
                "file_type": file_type,
                "full_file_path": full_file_path,
            }

        response = processor.process_prompt(uploaded_file, prompt, file_type, full_file_path)
        if response:
            cursor.execute(
                "INSERT INTO responses (file_path, prompt, response) VALUES (%s, %s, %s)",
                (file_path, prompt, response)
            )
            conn.commit()
            return {"response": response}, 200
        else:
            return {"error": "Content generation failed"}, 500

    except Exception as e:
        print(f"Unexpected error in process_file_request: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500


@app.route('/images/<path:filename>',methods=['GET'])
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/generate', methods=['POST'])
def generate_questions():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400
            
        print("Processing file:", file.filename)

        upload_dir = "/tmp/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        print("File saved to:", file_path)
        
        result = process_file(file_path)

        import shutil
        shutil.rmtree(upload_dir)
        
        return jsonify({
            "final_output": result
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "Please check your input files and try again"
        }), 500



@app.route('/process/<action>', methods=['POST'])
def process_action(action):
    try:
        file = request.files.get('file')
        print(file)
        question = request.form.get('message')
        print(question,"GGGGGGGGGg")
        url = request.form.get('url')

        if action not in ['qa', 'notes', 'flashcards', 'summary', 'keypoints', 'quiz']:
            return jsonify({"error": "Invalid action"}), 400

        if not file and not url:
            return jsonify({"error": "File is missing"}), 400

        if file:
            file_path = file.filename
            print(file_path)
            file_path = "uploads/"+ file_path 
            file.save(file_path)
            file_url = upload_file_to_s3(file_path, "edusage-bucket", file_path)
            print(file_url)
            file = retrieve_s3_file_content("edusage-bucket", file_path)

        prompts = {
            "qa": question,
            "notes": """
                         Analyze the provided document and generate detailed, well-organized smart notes. Ensure the notes are structured and informative, while capturing the essence of the document. The notes should include the following sections:
                         Overview:
                                 A high-level summary of the content, outlining the main topics and objectives. Provide a clear understanding of the purpose and scope of the material, offering context for the reader.
                         Key Concepts:
                                 A comprehensive list of the most important ideas, terms, or topics discussed in the document. For each concept, include a detailed explanation that clarifies its meaning, relevance, and relationship to other concepts in the document.
                         Critical Points:
                                 A summary of essential takeaways, key facts, and examples that are crucial for understanding the material. Emphasize important details that students must grasp, such as unique insights, data, or case studies.
                         Application:
                                 Provide detailed notes on how the concepts and critical points can be applied in real-world contexts or exams. Offer practical examples, potential use cases, or scenarios where these ideas might be relevant or tested.
                         Additional Insights:
                                 Include any additional important information, such as potential pitfalls, common misconceptions, or advanced topics that provide a deeper understanding of the subject.
                         the notes should contain all the topics listed above without leaving any headings above 
                         Ensure the notes are clear, concise, and easy to scan. Prioritize critical information while making sure that every section is well-explained and free of redundancy.
                         Strictyly dont provide the introductory paragraph as heres a break down and similar to that be oin context start your response from the overview

                         Example Output(Json Format):
                         {
                                        "Overview": "<overview of the document provided>",
                                        "Key Concepts": [
                                            "<key concept 1>",
                                            "<key concept 2>",
                                            "<key concept 3>"
                                        ],
                                        "Critical Points": [
                                            "<critical point 1>",
                                            "<critical point 2>",
                                            "<critical point 3>"
                                        ],
                                        "Application": [
                                            "<application 1>",
                                            "<application 2>",
                                            "<application 3>"
                                        ],
                                        "Additional Insights": [
                                            "<insight 1>",
                                            "<insight 2>",
                                            "<insight 3>"
                                        ]
                            }

                    """,

            "flashcards":  """
                                Analyze the entire content of the provided file and generate as many flashcards as possible, covering all key concepts, definitions, terms, important details, and examples. Be thorough in extracting and identifying all concepts from every section to ensure complete coverage of the material.
                                Each flashcard must follow this schema:
                                Key: The title or heading of the concept (e.g., a term, topic, question, or key idea).
                                Value: A concise yet clear explanation, description, or answer to the concept, including relevant examples or context if needed.
                                Ensure the flashcards capture every possible piece of information that could be useful for learning or revision. The output should be a dictionary where each key-value pair represents a flashcard. Structure the flashcards as:
                            {  
                                "Concept 1 Title": "Brief explanation or description of Concept 1",  
                                "Concept 2 Title": "Brief explanation or description of Concept 2",  
                                ...  
                            }  
                                Be exhaustive and ensure no important concept or detail is left out.
                                Avoid redundancy and ensure that each explanation is clear, relevant, and concise.
                                Your goal is to generate the maximum number of flashcards, making sure every topic, sub-topic, and important detail is covered.
                                                        
                            """,

            "summary": """
                            Analyze the entire content of the provided file and generate a summary that includes all headings and their corresponding content. Each heading should be followed by a detailed summary paragraph that captures the key points and essential ideas of that section. Additionally, for each section, provide a brief 150-word bullet-point summary to highlight the most important takeaways. Ensure the summary follows this structure:
                                Headings:
                                    Include every heading or sub-heading from the document to ensure full coverage.
                                For each heading, provide a summary paragraph (around 150 words) that concisely explains the content, focusing on the core ideas, concepts, and important details.
                                Bullet-Point Summary:
                                    Under each heading, list key points as brief bullet points (about 3-5 points).
                                    Each bullet point should capture the most important takeaway, fact, or concept from the section.
                                Overall Organization:
                                    The summary should be logically organized, following the structure of the original document.
                                    Ensure smooth transitions between sections, making the summary suitable for quick revision and easy comprehension by students.
                                Clarity and Simplicity:
                                    Complex ideas should be simplified for clarity without losing essential information.
                        Ensure the content is clear, concise, and free of redundancy.
                        """,

            "keypoints": """
                            Analyze the provided file and generate a comprehensive last-minute revision guide, ensuring the following structure:
                            Extract the most important concepts. Present the information in the following JSON format:

                            The key concept should be included under the "Concept" field.
                            List the critical topics related to the concept under the "response" field as an array.
                            Ensure the response is concise and formatted correctly as specified."
                            Example output:
                            {
                                "Concept": "Neural Networks",
                                "response": [
                                    "Forward propagation",
                                    "Backward propagation",
                                    "Hidden layers",
                                    "Input layer",
                                    "Output layer"
                            ],
                            {
                                "Concept": "Machine Learning",
                                "response": [
                                    "Supervised learning",
                                    "Unsupervised learning",
                                    "Reinforcement learning",
                                    "Overfitting and underfitting",
                                    "Feature scaling"
                                ]
                            },

                            ....
                            }


                            Example format(Json):
                           {
                                "Concept": <concept name>,  
                                "response":<responses>
                               
                            }

                        """    ,
            "quiz": """
               Analyze the provided file and generate at least 10 multiple-choice questions (MCQs) in strict JSON format. 
               Each question MUST follow this exact structure:
               {
                   "question": "A clear, concise question based on key concepts from the file",
                   "options": {
                       "A": "First option text",
                       "B": "Second option text", 
                       "C": "Third option text",
                       "D": "Fourth option text"
                   },
                   "correct_answer": "The letter of the correct option (A, B, C, or D)"
                   "explanation": "An explanation of the correct answer and how it relates to the question"
               }
               Guidelines:
               - Generate questions that cover different aspects of the document
               - Ensure questions are challenging but fair
               - Provide plausible distractors for incorrect options
               - If no clear content is available, return a JSON array with an error message object
               Example output:
               [
                   {
                       "question": "What is the primary purpose of machine learning?",
                       "options": {
                           "A": "To replace human programmers",
                           "B": "To learn and improve from data",
                           "C": "To create complex algorithms",
                           "D": "To generate random predictions"
                       },
                       "correct_answer": "B"
                       "explanation": "Machine learning uses data to learn patterns and make predictions without being explicitly programmed."
                   }
               ]
               """
        }
        if action not in prompts:
            return jsonify({"error": f"Invalid action '{action}'"}), 400

        prompt = prompts[action]
        if file:
            response, status_code = process_file_request(file_url,file_path, prompt)
        elif url:
            response, status_code = process_file_request(url, prompt)
        else:
            return jsonify({"error": "File or URL is missing"}), 400
        print(response)
        return jsonify(response), status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/quiz', methods=['POST'])
def process_quiz():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Upload to S3
            s3_file_path = f"uploads/{filename}"
            s3_url = upload_file_to_s3(filepath, "edusage-bucket", s3_file_path)
            
            if not s3_url:
                return jsonify({'error': 'Failed to upload file to S3'}), 500

            # Extract text and generate questions
            pdf_text = extract_text_from_pdf(filepath)
            questions = generate_mcq_questions(pdf_text)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({'response': questions})
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def generate_mcq_questions(text):
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise Exception("API key not found in environment variables")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = """Generate 10 multiple choice questions from the following text. 
        For each question:
        1. Include the question text
        2. Provide 4 options labeled A, B, C, D
        3. Specify the correct answer (A, B, C, or D)
        4. Include a brief explanation of why the answer is correct
        5. Assign marks (1-5 based on difficulty)
        Guidelines:
        - Generate questions that cover different aspects of the document
        - Ensure questions are challenging but fair
        - Provide plausible distractors for incorrect options
        - If no clear content is available, return a JSON array with an error message object
        Analyze the provided file and generate at least 10 multiple-choice questions (MCQs) in strict JSON format. 
        Each question MUST follow this exact structure:

        Format the output as a JSON array within markdown code blocks, like this:
        ```json

        {
            "question": "A clear, concise question based on key concepts from the file",
            "options": {
                "A": "First option text",
                "B": "Second option text", 
                "C": "Third option text",
                "D": "Fourth option text"
            },
            "correct_answer": "The letter of the correct option (A, B, C, or D)"
            "explanation": "An explanation of the correct answer and how it relates to the question"
            "marks":2
        }
        Example output:
        [
            {
                "question": "What is the primary purpose of machine learning?",
                "options": {
                    "A": "To replace human programmers",
                    "B": "To learn and improve from data",
                    "C": "To create complex algorithms",
                    "D": "To generate random predictions"
                },
                "correct_answer": "B"
                "explanation": "Machine learning uses data to learn patterns and make predictions without being explicitly programmed."
                "marks":2
            }
        ]
        ```
               """

        response = model.generate_content(prompt)
        print(response.text)
        return response.text

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        raise e

def extract_text_from_pdf(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    pdf_file.close()
    return text

def evaluate_answer(question, answer, context):
    try:
        context_document = Document(page_content=context)
        
        prompt_template = """
        Evaluate the student's answer to the question based on the provided context.

        If the answer is correct, award full marks and briefly explain why.
        If the answer is partially correct, award partial marks, explain what's missing or incorrect, and offer suggestions for improvement.
        If the answer is incorrect, give zero marks, provide the correct answer simply, and give constructive feedback.
        IMPORTANT: Always provide the marks. Marks are mandatory and should not be omitted under any circumstances.

        Be concise, friendly, and encouraging in your feedback.

        Context: {context}
        Question: {question}
        Student's Answer: {answer}

        Teacher's Feedback (including marks):
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "answer"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        response = chain.invoke({
            "input_documents": [context_document],
            "question": question,
            "answer": answer
        }, return_only_outputs=True)

        return response["output_text"]
    except Exception as e:
        print(f"Error in evaluate_answer: {str(e)}")
        raise e

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'answer' not in data or 'context' not in data:
            return jsonify({
                'status': 'error', 
                'message': 'Missing required fields: question, answer, and context'
            }), 400

        response = evaluate_answer(data['question'], data['answer'], data['context'])
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        print(f"Error in submit_answer: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
