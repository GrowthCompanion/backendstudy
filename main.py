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
CORS(app)  # Enable CORS for all routes

# Set up API keys securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ API Key is missing. Please set it in the .env file.")

# Configure Groq Model
groq = ChatGroq(model="llama-3.3-70b-versatile")
groq_config = LLMConfig(custom_model=groq)

# Initialize Educhain client
client = Educhain(groq_config)

# Initialize Groq's API client
groq_client = Groq()

# Function to format lesson plans properly
def format_lesson_plan(plan_text):
    """Format lesson plans with proper Markdown."""
    plan_text = re.sub(r"\*\*Topic:\*\*-(.*?)\n", r"\n### **Topic:** \1\n", plan_text)  # Format Topic
    plan_text = re.sub(r"\*\*Subtopic:\*\*-(.*?)\n", r"\n#### **Subtopic:** \1\n", plan_text)  # Format Subtopic
    plan_text = plan_text.replace("* ", "- ")  # Convert bullet points to Markdown lists
    return plan_text

# Endpoint to generate a study plan
@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    data = request.json
    topic = data.get("topic")
    num_days = data.get("num_days")
    difficulty = data.get("difficulty")

    if not topic or not num_days or not difficulty:
        return jsonify({"error": "Missing required fields"}), 400

    # Step 1: Get raw lesson plan from Educhain
    raw_plan = client.content_engine.generate_lesson_plan(topic=topic)

    # Extract lesson plan text properly
    if hasattr(raw_plan, 'dict'):
        raw_plan_content = raw_plan.dict()
    elif hasattr(raw_plan, 'content'):
        raw_plan_content = raw_plan.content
    elif hasattr(raw_plan, 'plan'):
        raw_plan_content = raw_plan.plan
    else:
        raw_plan_content = str(raw_plan)

    # Step 2: Refine the plan with Groq
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
        max_completion_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Step 3: Format and return the lesson plan
    refined_plan = chat_completion.choices[0].message.content
    formatted_plan = format_lesson_plan(refined_plan)  # Apply formatting

    return jsonify({"plan": formatted_plan})

# Endpoint to generate a quiz
@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    data = request.json
    topic = data.get("topic")
    quiz_type = data.get("quiz_type")
    num_questions = data.get("num_questions")

    if not topic or not quiz_type or not num_questions:
        return jsonify({"error": "Missing required fields"}), 400

    # Generate quiz questions
    questions = client.qna_engine.generate_questions(topic=topic, num=num_questions, question_type=quiz_type)

    # Format quiz questions for response
    quiz_questions = []
    for question in questions.questions:
        quiz_questions.append({
            "question": question.question,
            "options": question.options if hasattr(question, 'options') else [],
            "correct_answer": question.answer,
            "explanation": question.explanation
        })

    return jsonify({"questions": quiz_questions})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
