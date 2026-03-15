ResumeBuilderAI Setup Guide

1. Install Python
Make sure Python 3.10 or newer is installed.

You can check with:
python --version

2. Clone the Repository

git clone https://github.com/YOUR_USERNAME/resume-ats-ai.git
cd resume-ats-ai


3. Install Ollama
Download and install Ollama from:
https://ollama.com

or 
curl -fsSL https://ollama.com/install.sh | sh


After installing, pull the required model:

ollama pull qwen2.5:7b-instruct


4. Create a Virtual Environment

python -m venv .venv


6. Activate the Virtual Environment

Linux / Mac:
source .venv/bin/activate

Windows:
.venv\Scripts\activate


7. Install Required Dependencies

pip install -r requirements.txt


8. Run the Server

uvicorn main:app --reload


9. Open the API

Server will run at:
http://127.0.0.1:8000

Interactive API documentation:
http://127.0.0.1:8000/docs


10. Available Endpoints

Upload a resume and convert it to the generator schema:
POST /parse_resume

Generate an ATS-optimized resume:
POST /generate_resume
