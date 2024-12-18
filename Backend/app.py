from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from content_processor import ContentProcessor
import os
import psycopg2
import re
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/content_database")

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

processor = ContentProcessor(api_key=os.getenv("GOOGLE_API_KEY"), cursor=cursor, conn=conn)

cache = {}

def process_file_request(file_path, prompt):
    global cache
    try:
        if not file_path:
            return {"error": "file_path is required"}, 400

        if file_path in cache:
            cached_data = cache[file_path]
            uploaded_file = cached_data["uploaded_file"]
            file_type = cached_data["file_type"]
            full_file_path = cached_data["full_file_path"]
            print("Reusing previously processed file.")
        else:
            def is_youtube_url(url):
                return re.match(r"(https?://)?(www\\.)?(youtube\\.com|youtu\\.be)/.+", url)

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
                uploaded_file, file_type, full_file_path, upload_id = processor.upload_file(file_path)
                if not uploaded_file:
                    return {"error": "File upload/processing failed"}, 500

                cursor.execute(
                    "INSERT INTO files (file_path, file_type, uploaded, upload_id) VALUES (%s, %s, %s, %s)",
                    (full_file_path, file_type, True, upload_id)
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
                (full_file_path, prompt, response)
            )
            conn.commit()
            return {"response": response}, 200
        else:
            return {"error": "Content generation failed"}, 500

    except Exception as e:
        return {"error": str(e)}, 500



@app.route('/process/<action>', methods=['POST'])
def process_action(action):
    try:
        file = request.files.get('file')
        question = request.form.get('question')
        url = request.form.get('url')

        if action not in ['qa', 'notes', 'flashcards', 'summary', 'keypoints', 'quiz']:
            return jsonify({"error": "Invalid action"}), 400

        if not file and not url:
            return jsonify({"error": "File is missing"}), 400

        if file:
            file_path = file.filename
            file_path = "uploads/"+ file_path 
            file.save(file_path)

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
                            Headings and Two-Liner Summaries(dont include this(wordings) in your response):
                                For each heading or subheadingin the file provided, provide a two-line summary that captures the core essence of that section.
                                The two-liner should succinctly explain the topic, including key concepts, critical points, and important details, ensuring that no significant information is left out.
                            Example format:
                                Machine Learning:
                                Machine learning is a subset of artificial intelligence that enables a system to autonomously learn and improve using neural networks and deep learning, without being explicitly programmed, by feeding it large amounts of data.
                            Formulas Section:
                                If the content includes formulas, create a separate section titled 'Formulas'.
                                List all formulas from the file in a bulleted format, providing the formula name or context (if available).
                                Each formula should be accompanied by a brief description of its relevance or application in the material.
                            Organization:
                                The output should be well-organized and easy to scan. Ensure that each heading and subheading is followed by its two-liner summary.
                                The content should be structured clearly, making it easy for students to revise and grasp key concepts quickly before exams.
                            Clarity and Simplicity:
                                Ensure the content is clear, concise, and free of redundancy.
                            Ensure all topics are covered effectively, with no heading or formula overlooked.
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
            response, status_code = process_file_request(file_path, prompt)
        elif url:

            response, status_code = process_file_request(url, prompt)
        else:
            return jsonify({"error": "File or URL is missing"}), 400
        print(response)
        return jsonify(response), status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
  
