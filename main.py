import os
import threading
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import requests
import json
import re

import pdfplumber
import docx


# ==============================
# CONFIG
# ==============================

app = FastAPI(title="ResumeBuilderAI - Qwen Edition")

OLLAMA_API = os.getenv("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_PULL_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_PULL_TIMEOUT_SECONDS", "1800"))
OLLAMA_GENERATE_TIMEOUT = int(os.getenv("OLLAMA_GENERATE_TIMEOUT", "600"))

_MODEL_READY = False
_MODEL_LOCK = threading.Lock()


def _ollama_base_url() -> str:
    # Example: http://ollama:11434/api/generate -> http://ollama:11434
    return OLLAMA_API.split("/api/generate", 1)[0].rstrip("/")


def _is_model_present() -> bool:
    try:
        tags_url = f"{_ollama_base_url()}/api/tags"
        resp = requests.get(tags_url, timeout=10)
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        models = payload.get("models") or []
        names = {(m.get("name") or "").strip() for m in models if isinstance(m, dict)}
        return MODEL in names
    except Exception:
        return False


def _pull_model_if_missing() -> bool:
    pull_url = f"{_ollama_base_url()}/api/pull"
    print(
        f"[AI] Ensuring Ollama model '{MODEL}' is available via POST {pull_url} "
        f"(timeout={OLLAMA_PULL_TIMEOUT_SECONDS}s)..."
    )
    pull_resp = requests.post(
        pull_url,
        json={"name": MODEL},
        stream=True,
        timeout=OLLAMA_PULL_TIMEOUT_SECONDS,
    )
    pull_resp.raise_for_status()

    success = False
    last_status = None
    for line in pull_resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        status = obj.get("status")
        if isinstance(status, str) and status != last_status:
            print(f"[AI] ollama pull status: {status}")
            last_status = status
        if isinstance(status, str) and status.lower().startswith("success"):
            success = True
    return success


def ensure_model_available(force_pull: bool = False) -> None:
    global _MODEL_READY
    if _MODEL_READY and not force_pull:
        return

    with _MODEL_LOCK:
        if _MODEL_READY and not force_pull:
            return

        if not force_pull and _is_model_present():
            _MODEL_READY = True
            return

        retries = 2
        for attempt in range(1, retries + 1):
            try:
                pulled = _pull_model_if_missing()
                if pulled or _is_model_present():
                    _MODEL_READY = True
                    print(f"[AI] Ollama model '{MODEL}' is ready.")
                    return
            except Exception as e:
                print(f"[AI] Model pull attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(2)
                    continue
                raise

        raise RuntimeError(f"Ollama model '{MODEL}' not available after retries")


# ==============================
# DATA MODELS
# ==============================

class Experience(BaseModel):
    role: str
    company: str
    location: Optional[str] = None
    startDate: str
    endDate: str
    bullets: List[str] = Field(default_factory=list)


class Project(BaseModel):
    title: str
    link: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class Education(BaseModel):
    school: str
    degree: str
    field: str
    location: Optional[str] = None
    endDate: str


class SkillCategory(BaseModel):
    category: str
    skills: List[str]


class ResumeRequest(BaseModel):
    name: str
    email: str
    phone: str
    linkedin: str
    portfolio: str
    summary: str
    experiences: List[Experience]
    projects: List[Project]
    education: List[Education]
    skills: List[SkillCategory]
    job_description: str


class OptimizedResume(BaseModel):
    summary: str
    experiences: List[Experience]
    projects: List[Project]
    education: List[Education]
    skills: List[SkillCategory]
    ats_report: Dict[str, object]


# ==============================
# OLLAMA CALL
# ==============================

def call_ollama(prompt: str):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }

    # Fast path: ensure model once, then call generate.
    ensure_model_available()
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=OLLAMA_GENERATE_TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.HTTPError as e:
        # If model was unloaded/missing, force pull and retry once.
        if getattr(e.response, "status_code", None) == 404:
            ensure_model_available(force_pull=True)
            retry = requests.post(OLLAMA_API, json=payload, timeout=OLLAMA_GENERATE_TIMEOUT)
            retry.raise_for_status()
            return retry.json().get("response", "")
        raise


@app.on_event("startup")
def _warm_ollama_model() -> None:
    # Pre-warm on startup to avoid first-request 404 and long UI waits.
    try:
        ensure_model_available()
    except Exception as e:
        # Keep service booting; request path still retries.
        print(f"[AI] Startup model warm-up warning: {e}")


def safe_json_parse(text: str):

    try:
        return json.loads(text)

    except:

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("Model did not return valid JSON.")

        return json.loads(match.group(0))


# ==============================
# RESUME TEXT EXTRACTION
# ==============================

def extract_resume_text(file: UploadFile):

    filename = file.filename.lower()

    if filename.endswith(".pdf"):

        with pdfplumber.open(file.file) as pdf:

            text = ""

            for page in pdf.pages:
                text += page.extract_text() or ""

        return text


    elif filename.endswith(".docx"):

        doc = docx.Document(file.file)

        text = "\n".join([p.text for p in doc.paragraphs])

        return text

    else:
        raise HTTPException(
            status_code=400,
            detail="Only PDF or DOCX resumes supported"
        )


# ==============================
# RESUME SECTION SPLITTER
# ==============================

def split_resume_sections(text: str):

    sections = {
        "summary": "",
        "experience": "",
        "education": "",
        "projects": "",
        "skills": ""
    }

    patterns = {

        "summary": r"(summary|professional summary|profile|about me|career summary|career overview|objective|personal statement)",

        "experience": r"(experience|employment|work history|professional experience|career history|internships)",

        "education": r"(education|academic|academic background|qualifications)",

        "projects": r"(projects|portfolio|personal projects|research)",

        "skills": r"(skills|technical skills|core skills|competencies|technologies|tech stack)"
    }

    lines = text.split("\n")

    current = "summary"   # safe default

    for line in lines:

        lower = line.lower().strip()

        for key, pattern in patterns.items():
            if re.search(pattern, lower):
                current = key
                continue

        if current in sections:
            sections[current] += line + "\n"

    return sections
def extract_contact_info(text: str):

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

    phone_match = re.search(r"\+?\d[\d\s\-]{8,15}", text)

    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""

    return email, phone
# ==============================
# RESUME PARSER
# ==============================

def parse_resume_to_schema(resume_text: str):

    sections = split_resume_sections(resume_text)
    email, phone = extract_contact_info(resume_text)
    prompt = f"""
You are a resume parser.

Convert the resume sections into the EXACT JSON schema below.

If information missing:
- use empty string ""
- or empty lists []

Do NOT invent data.

Return STRICT JSON ONLY.

SCHEMA:

{{
  "name": "",
  "email": "",
  "phone": "",
  "linkedin": "",
  "portfolio": "",
  "summary": "",
  "experiences": [
    {{
      "role": "",
      "company": "",
      "location": "",
      "startDate": "",
      "endDate": "",
      "bullets": []
    }}
  ],
  "projects": [
    {{
      "title": "",
      "link": "",
      "bullets": []
    }}
  ],
  "education": [
    {{
      "school": "",
      "degree": "",
      "field": "",
      "location": "",
      "endDate": ""
    }}
  ],
  "skills": [
    {{
      "category": "",
      "skills": []
    }}
  ],
  "job_description": ""
}}

RESUME SECTIONS:
Detected email: {email}
Detected phone: {phone}

If these values exist, use them in the JSON.
SUMMARY:
{sections["summary"]}

EXPERIENCE:
{sections["experience"]}

EDUCATION:
{sections["education"]}

PROJECTS:
{sections["projects"]}

SKILLS:
{sections["skills"]}
"""

    result = call_ollama(prompt)

    parsed = safe_json_parse(result)

    return parsed


# ==============================
# PASS 1: EXTRACT JD SIGNALS
# ==============================

def extract_jd_signals(job_description: str):

    prompt = f"""
You are an ATS job analyzer.

Return STRICT JSON only:

{{
  "must_have_keywords": ["..."],
  "nice_to_have_keywords": ["..."],
  "tools": ["..."],
  "responsibilities": ["..."],
  "soft_skills": ["..."]
}}

Rules:
- Extract 10-20 must-have ATS keywords.
- Keep terms short (e.g., "SQL", "Power BI", "ETL").
- No commentary. JSON only.

JOB DESCRIPTION:
{job_description}
"""

    result = call_ollama(prompt)

    return safe_json_parse(result)


# ==============================
# PASS 2: OPTIMIZE RESUME
# ==============================

def optimize_resume(req: ResumeRequest, jd_data: dict):

    prompt = f"""
You are ResumeBuilderAI.

Rewrite the resume to maximize ATS match using the extracted job signals.

Return STRICT JSON only:

{{
  "summary": "...",
  "experiences": [...],
  "projects": [...],
  "education": [...],
  "skills": [...],
  "ats_report": {{
      "coverage_percent": 0,
      "keywords_covered": ["..."],
      "keywords_missing": ["..."],
      "notes": ["..."]
  }}
}}

Rules:
- Do not invent employers or degrees.
- Improve impact with quantified achievements.
- Naturally integrate must-have keywords.
- 3–6 bullets per experience.
- Professional, ATS-safe language.

JOB SIGNALS:
{json.dumps(jd_data, indent=2)}

CANDIDATE RESUME:
{req.model_dump_json(indent=2)}
"""

    result = call_ollama(prompt)

    parsed = safe_json_parse(result)

    return OptimizedResume(**parsed)


# ==============================
# PASS 3: COVERAGE CHECK
# ==============================

def compute_coverage(resume: OptimizedResume, jd_data: dict):

    must_have = [k.lower() for k in jd_data.get("must_have_keywords", [])]

    resume_text = json.dumps(resume.model_dump()).lower()

    covered = [k for k in must_have if k in resume_text]
    missing = [k for k in must_have if k not in resume_text]

    coverage_percent = round(
        (len(covered) / max(len(must_have), 1)) * 100,
        2
    )

    return coverage_percent, covered, missing


def improve_resume_if_needed(resume: OptimizedResume, jd_data: dict):

    coverage, covered, missing = compute_coverage(resume, jd_data)

    if coverage >= 85:
        return resume

    prompt = f"""
The resume below is missing important ATS keywords:

{missing}

Improve the resume by:
- Naturally integrating missing keywords
- Strengthening quantified achievements
- Keeping same structure
- Not inventing companies or degrees

Return STRICT JSON in same format as before.

CURRENT RESUME:
{resume.model_dump_json(indent=2)}

JOB SIGNALS:
{json.dumps(jd_data, indent=2)}
"""

    improved = safe_json_parse(call_ollama(prompt))

    return OptimizedResume(**improved)


# ==============================
# ATS KEYWORD SCORING
# ==============================

def compute_ats_score(resume: OptimizedResume, jd_data: dict):

    coverage, covered, missing = compute_coverage(resume, jd_data)

    return {
        "final_ats_score": coverage,
        "keywords_found": covered,
        "keywords_missing": missing
    }


# ==============================
# IMPROVEMENT LOOP
# ==============================

def optimize_until_threshold(resume: OptimizedResume, jd_data: dict, threshold=90):

    max_iterations = 3

    for i in range(max_iterations):

        ats = compute_ats_score(resume, jd_data)

        if ats["final_ats_score"] >= threshold:
            return resume, ats, i + 1

        missing = ats["keywords_missing"]

        prompt = f"""
The resume below has an ATS score of {ats["final_ats_score"]}.

It must reach at least {threshold}.

Missing keywords:
{missing}

Improve the resume by:
- Integrating missing keywords naturally
- Strengthening quantified achievements
- Maintaining structure
- Not inventing companies or degrees

Return STRICT JSON in the same schema.

CURRENT RESUME:
{resume.model_dump_json(indent=2)}

JOB SIGNALS:
{json.dumps(jd_data, indent=2)}
"""

        improved = safe_json_parse(call_ollama(prompt))

        resume = OptimizedResume(**improved)

    ats = compute_ats_score(resume, jd_data)

    return resume, ats, max_iterations


# ==============================
# RESUME PARSER ENDPOINT
# ==============================

@app.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...)):

    try:

        resume_text = extract_resume_text(file)

        parsed_resume = parse_resume_to_schema(resume_text)

        return parsed_resume

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ==============================
# GENERATOR ENDPOINT
# ==============================

@app.post("/generate_resume")
def generate_resume(req: ResumeRequest):

    try:

        jd_data = extract_jd_signals(req.job_description)

        optimized_resume = optimize_resume(req, jd_data)

        optimized_resume = improve_resume_if_needed(
            optimized_resume,
            jd_data
        )

        optimized_resume, ats_score, iterations = optimize_until_threshold(
            optimized_resume,
            jd_data
        )

        return {

            "candidate_info": {
                "name": req.name,
                "email": req.email,
                "phone": req.phone,
                "linkedin": req.linkedin,
                "portfolio": req.portfolio
            },

            "resume": optimized_resume.model_dump(),

            "jd_analysis": jd_data,

            "ats_final_result": {
                "final_ats_score": ats_score["final_ats_score"],
                "iterations_needed": iterations,
                "keywords_found": ats_score["keywords_found"],
                "keywords_missing": ats_score["keywords_missing"]
            }

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )