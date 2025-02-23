from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from educhain import Educhain, LLMConfig
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up API keys
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ API Key is missing. Please set it in the .env file.")

# Configure Groq
groq = ChatGroq(model="llama-3.1-8b-instant")
groq_config = LLMConfig(custom_model=groq)
client = Educhain(groq_config)
groq_client = Groq()

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    try:
        data = request.json
        topic = data.get("topic")
        num_days = data.get("num_days")
        difficulty = data.get("difficulty", "Medium")

        # Generate initial content with Groq
        prompt = f"""
        Create a detailed {num_days}-day study plan for {topic} at {difficulty} difficulty level.
        Include:
        1. Daily topics and subtopics
        2. Learning objectives
        3. Practice exercises
        4. Review points
        
        Format the response in Markdown with clear headers and bullet points.
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable educational assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )

        plan = chat_completion.choices[0].message.content
        return jsonify({"plan": plan})

    except Exception as e:
        print(f"Error generating plan: {str(e)}")
        return jsonify({"error": "Failed to generate study plan", "details": str(e)}), 500

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.json
        topic = data.get("topic")
        num_questions = data.get("num_questions", 5)

        # Generate quiz with Groq
        prompt = f"""
        Create {num_questions} multiple-choice questions about {topic}.
        For each question, provide:
        1. The question
        2. Four possible answers
        3. The correct answer index (0-3)
        
        Format as a list of questions with options.
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable educational assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )

        # Parse the response into structured questions
        raw_response = chat_completion.choices[0].message.content
        
        # Simple parsing of the response into questions
        questions = []
        current_lines = raw_response.split('\n')
        current_question = None
        options = []
        
        for line in current_lines:
            line = line.strip()
            if line.startswith(('Q:', 'Question:')):
                if current_question:
                    questions.append({
                        "question": current_question,
                        "options": options[:4],
                        "correctAnswer": 0  # Default to first option
                    })
                current_question = line.split(':', 1)[1].strip()
                options = []
            elif line.startswith(('A)', 'B)', 'C)', 'D)', 'a)', 'b)', 'c)', 'd)')):
                options.append(line[2:].strip())
                
        if current_question:
            questions.append({
                "question": current_question,
                "options": options[:4],
                "correctAnswer": 0
            })

        return jsonify({"questions": questions})

    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return jsonify({"error": "Failed to generate quiz", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
