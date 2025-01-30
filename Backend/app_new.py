from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
from content_processor_new import ContentProcessorNew
import os
from dotenv import load_dotenv
import google.generativeai as genai
from collections import OrderedDict
import psycopg2
import json
from aws import upload_file_to_s3, retrieve_s3_file_content
from qa import QuestionPaperGenerator
from image_case import ImageCaseStudyGenerator
import re
from evaluate import AnswerEvaluator

load_dotenv()

app = Flask(__name__)
CORS(app)
evaluator = AnswerEvaluator()
# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:DanielDas2004@edusage-database.cp6gyg0soaec.ap-south-1.rds.amazonaws.com:5432/edusage-database")

# Cache for file processing
cache = {}
AWS_REGION = os.getenv("AWS_REGION")
# Initialize processor
processor = None
conn = None
cursor = None

def init_db():
    global processor, conn, cursor
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            -- Existing tables from old app.py
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                uploaded BOOLEAN NOT NULL,
                upload_id TEXT
            );

            CREATE TABLE IF NOT EXISTS responses (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- New tables for lab features
            CREATE TABLE IF NOT EXISTS lab_documents (
                id SERIAL PRIMARY KEY,
                file_hash TEXT UNIQUE NOT NULL,
                original_path TEXT NOT NULL,
                summary_path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS lab_notes (
                id SERIAL PRIMARY KEY,
                doc_id INTEGER REFERENCES lab_documents(id),
                file_hash TEXT NOT NULL,
                note_type TEXT NOT NULL,
                content JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_lab_notes_file_hash ON lab_notes(file_hash);
            CREATE INDEX IF NOT EXISTS idx_lab_documents_file_hash ON lab_documents(file_hash);
        """)
        conn.commit()
        
        # Initialize processor with database connection
        processor = ContentProcessorNew(
            api_key=os.getenv("GOOGLE_API_KEY"),
            cursor=cursor,
            conn=conn
        )
        
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        raise

init_db()

# Existing routes from old app.py
@app.route('/process/<action>', methods=['POST'])
def process_action(action):
    try:
        file = request.files.get('file')
        question = request.form.get('message')
        url = request.form.get('url')

        if action not in ['qa', 'notes', 'flashcards', 'summary', 'keypoints', 'quiz']:
            return jsonify({"error": "Invalid action"}), 400

        if not file and not url:
            return jsonify({"error": "File is missing"}), 400

        if file:
            file_path = file.filename
            file_path = "uploads/" + file_path 
            print("check man -",file_path)
            file.save(file_path)
            # file_url = upload_file_to_s3(file_path, "edusage-bucket", file_path)
            # file = retrieve_s3_file_content("edusage-bucket", file_path)
        
        prompts = {
            "qa": question,
            "notes": """
                    Analyze the provided document and generate detailed, well-organized smart notes...
                    """,
            "flashcards": """
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
                    Generate a comprehensive summary...
                    """,
            "keypoints": """
                    Extract and organize key points...
                    """,
            "quiz": """
                    Generate quiz questions...
                    """
        }

        prompt = prompts.get(action)
        if not prompt:
            return jsonify({"error": "Invalid action"}), 400
        if file :
            result=process_file_request(file_path=file_path,s3_file_path=file_path, prompt=prompt)
        elif url:
            result = process_file_request(file_path=url, s3_file_path=None, prompt=prompt)
        return jsonify(result[0]), result[1]

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_file_request(file_path, s3_file_path, prompt):
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
            if processor.youtube_handler.is_youtube_url(file_path):
                print("file_path of vid:",file_path)
                print("s3 path of vid :",s3_file_path)
                
                file_type = "video"
                sanitized_url = re.sub(r'[^a-zA-Z0-9]', '_', file_path)
                check_file_path=f"https://{file_path}.s3.{AWS_REGION}.amazonaws.com/{file_path}"
            else:
                print("file_path of file -",file_path)
                file_type = processor.file_handler.determine_file_type(file_path)
                check_file_path=f"https://{file_path}.s3.{AWS_REGION}.amazonaws.com/{file_path}"

            # Check for existing file in database
            cursor.execute("SELECT file_path, uploaded, upload_id FROM files WHERE file_path = %s", (check_file_path,))
            existing_file = cursor.fetchone()

            if existing_file:
                print("Reusing previously processed file.")
                full_file_path = existing_file[0]
                upload_id = existing_file[2]
                uploaded_file = genai.get_file(upload_id)
            else:
                
                print("Processing new file.")
                uploaded_file, file_type, full_file_path, upload_id = processor.upload_file(file_path, s3_file_path)
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
        print(f"Error in process_file_request: {str(e)}")
        return {"error": str(e)}, 500

@app.route('/lab/process', methods=['POST'])
def process_lab_document():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Save uploaded file temporarily
        temp_path = f"temp_lab_{file.filename}"
        file.save(temp_path)

        # Process the PDF
        result = processor.process_pdf(temp_path)
        
        if result:
            if result.get('new_summary', True):
                print('Storing if it is new -')
                # Store in database only if it's a new summary
                cursor.execute("""
                    INSERT INTO lab_documents (
                        file_hash,
                        original_path,
                        summary_path,
                        status
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (file_hash) DO UPDATE 
                    SET status = 'updated'
                    RETURNING id
                """, (
                    result['file_hash'],
                    result.get('original_pdf_s3', ''),
                    result['summary_pdf_s3'],
                    'processed'
                ))
                print("Reached after get")
                doc_id = cursor.fetchone()[0]
                conn.commit()
            else:
                print("got into else")
                # Get existing document ID
                cursor.execute(
                    "SELECT id FROM lab_documents WHERE file_hash = %s",
                    (result['file_hash'],)
                )
                
                doc_id = cursor.fetchone()[0]
                print("got doc_id")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            result['doc_id'] = doc_id
            
            return jsonify({
                "status": "success",
                "data": result
            })
        else:
            return jsonify({"error": "Failed to process document"}), 500

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

@app.route('/lab/revise', methods=['POST'])
def generate_lab_revision_notes():
    try:
        data = request.json
        summary_path = data.get('summary_path')
        file_hash = data.get('file_hash')
        doc_id = data.get('doc_id')
        print("File hash -",file_hash)
        print("Summary",summary_path)
        print("doc id",doc_id)
        
        if not all([summary_path, file_hash, doc_id]):
            return jsonify({"error": "Missing required parameters"}), 400
        cursor.execute(
                "SELECT content,doc_id FROM lab_notes WHERE file_hash = %s AND note_type = 'revise'",
                (file_hash,)
            )
        existing_notes = cursor.fetchone()
        if existing_notes:
            try:
                print("in exisitng")
                
                print(existing_notes)
                notes={}
                notes["content"]=existing_notes[0]
                print('---------------------------')
                print(notes["content"])
                note_id=existing_notes[1]
            
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
        else:                
            notes = processor.generate_revision_notes(summary_path, file_hash)
        
            if notes:
                # Store in database
                cursor.execute("""
                    INSERT INTO lab_notes (
                        doc_id,
                        file_hash,
                        note_type,
                        content
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_id, note_type) DO UPDATE 
                    SET content = EXCLUDED.content
                    RETURNING id
                """, (
                    doc_id,
                    file_hash,
                    'revise',
                    json.dumps(notes["content"])
                ))
                
                note_id = cursor.fetchone()[0]
                conn.commit()
                
                return jsonify({
                    "status": "success",
                    "data": {
                        "note_id": note_id,
                        "notes": notes["content"]
                    }
                })
            else:
                return jsonify({"error": "Failed to generate revision notes"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/lab/smart-notes', methods=['POST'])
def generate_lab_smart_notes():
    try:
        data = request.json
        summary_path = data.get('summary_path')
        # print("summary_path-",summary_path)
        file_hash = data.get('file_hash')
        doc_id = data.get('doc_id')
        
        if not all([summary_path, file_hash, doc_id]):
            return jsonify({"error": "Missing required parameters"}), 400
        print("-----no error before generate smart notes----")
        
        cursor.execute(
                "SELECT content,doc_id FROM lab_notes WHERE file_hash = %s AND note_type = 'smart'",
                (file_hash,)
            )
        existing_notes = cursor.fetchone()
        if existing_notes:
            try:
                print("in exisitng")
                
                print(existing_notes)
                notes={}
                notes["content"]=json.load(existing_notes[0])
                print("------------------")
                print(notes["content"])
                note_id=existing_notes[1]
            
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
        else:                
            notes = processor.generate_smart_notes(summary_path, file_hash)
            print("actual notes -", notes)
            print("-----------------")
            print("Content of notes-")
            print(type(notes["content"]))
             
            if notes:
                print("S got the notes")
                # Store in database
                try:
                    cursor.execute("""
                        INSERT INTO lab_notes (
                            doc_id,
                            file_hash,
                            note_type,
                            content
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (doc_id, note_type) DO UPDATE 
                        SET content = EXCLUDED.content
                        RETURNING id
                    """, (
                        doc_id,
                        file_hash,
                        'smart',
                        str(notes["content"])
                    ))
                    print("Query executed successfully.")
                    note_id = cursor.fetchone()[0]
                except Exception as db_error:
                    print("Database error:", db_error)
                    raise

                conn.commit()
            else:
                return jsonify({"error": "Failed to generate smart notes"}), 500
            
        print("notes id ",note_id)
        return jsonify({
                "status": "success",
                "data": {
                    "note_id": note_id,
                    "notes": notes["content"]
                }
            })
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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



@app.route('/evaluate', methods=['POST'])
def evaluate_answers():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            

        results = evaluator.batch_evaluate(data)
        
        # Return the evaluation results
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)