from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from educhain import Educhain, LLMConfig
from groq import Groq
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ API Key is missing. Please set it in the .env file.")

# Configure Groq Model
groq = ChatGroq(model="llama-3.1-8b-instant")
groq_config = LLMConfig(custom_model=groq)
client = Educhain(groq_config)
groq_client = Groq()

def format_lesson_plan(plan_text):
    plan_text = re.sub(r"\*\*Topic:\*\*-(.*?)\n", r"\n### **Topic:** \1\n", plan_text)
    plan_text = re.sub(r"\*\*Subtopic:\*\*-(.*?)\n", r"\n#### **Subtopic:** \1\n", plan_text)
    plan_text = plan_text.replace("* ", "- ")
    return plan_text.strip()

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    data = request.json
    topic = data.get("topic")
    num_days = data.get("num_days")
    difficulty = data.get("difficulty")

    if not topic or not num_days or not difficulty:
        return jsonify({"error": "Missing required fields"}), 400

    raw_plan = client.content_engine.generate_lesson_plan(topic=topic)
    raw_plan_content = getattr(raw_plan, 'content', str(raw_plan))

    prompt = f"""
    You are a helpful assistant. Adapt the following lesson plan for {num_days} days at a {difficulty} difficulty level.

    Topic: {topic}
    
    Raw Lesson Plan:
    {raw_plan_content}

    Please structure the plan clearly and ensure it is well-distributed over {num_days} days.
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_completion_tokens=1024
    )

    refined_plan = chat_completion.choices[0].message.content
    formatted_plan = format_lesson_plan(refined_plan)
    return jsonify({"plan": formatted_plan})

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    data = request.json
    topic = data.get("topic")
    quiz_type = data.get("quiz_type")
    num_questions = data.get("num_questions")

    if not topic or not quiz_type or not num_questions:
        return jsonify({"error": "Missing required fields"}), 400

    questions = client.qna_engine.generate_questions(topic=topic, num=num_questions, question_type=quiz_type)

    quiz_questions = []
    for question in questions.questions:
        correct_option = question.answer
        options = question.options if hasattr(question, 'options') else []

        if correct_option in options:
            correct_index = options.index(correct_option)
        else:
            correct_index = 0  # Default to first option if something goes wrong

        quiz_questions.append({
            "question": question.question,
            "options": options,
            "correct_answer": correct_index,
            "explanation": question.explanation
        })

    return jsonify({"questions": quiz_questions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
