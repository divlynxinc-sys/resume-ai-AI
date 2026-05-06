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

OLLAMA_API = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"


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

    response = requests.post(OLLAMA_API, json=payload, timeout=180)
    response.raise_for_status()

    return response.json().get("response", "")


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
# ATS SCORING HELPERS
# ==============================

def normalize_text(text: str):
    return re.sub(r"\s+", " ", text.lower()).strip()


def keyword_match_score(resume_text: str, keywords: list):
    """
    Calculates keyword coverage percentage.

    Returns:
    - score
    - covered keywords
    - missing keywords
    """

    normalized_resume = normalize_text(resume_text)

    keywords = list(dict.fromkeys([
        k.strip().lower()
        for k in keywords
        if isinstance(k, str) and k.strip()
    ]))

    covered = []
    missing = []

    for keyword in keywords:
        keyword_norm = normalize_text(keyword)

        if keyword_norm in normalized_resume:
            covered.append(keyword)
        else:
            missing.append(keyword)

    score = round((len(covered) / max(len(keywords), 1)) * 100, 2)

    return score, covered, missing


def responsibility_alignment_score(resume_text: str, responsibilities: list):
    """
    Scores whether resume content reflects job responsibilities,
    not just isolated keywords.
    """

    normalized_resume = normalize_text(resume_text)

    responsibilities = [
        r.strip().lower()
        for r in responsibilities
        if isinstance(r, str) and r.strip()
    ]

    if not responsibilities:
        return 60

    matched = 0

    stop_words = {
        "with", "from", "that", "this", "will", "have",
        "your", "their", "using", "work", "team", "role",
        "able", "into", "over", "under", "within", "across",
        "while", "also", "such", "including", "based"
    }

    for responsibility in responsibilities:

        words = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", responsibility)
            if w not in stop_words
        ]

        if not words:
            continue

        overlap = sum(1 for w in words if w in normalized_resume)

        if overlap / max(len(words), 1) >= 0.35:
            matched += 1

    return round((matched / max(len(responsibilities), 1)) * 100, 2)


def quantified_achievement_score(resume: OptimizedResume):
    """
    Rewards bullets with measurable achievements, metrics,
    or impact-focused wording.
    """

    all_bullets = []

    for exp in resume.experiences:
        all_bullets.extend(exp.bullets)

    for project in resume.projects:
        all_bullets.extend(project.bullets)

    if not all_bullets:
        return 40

    quantified_count = 0

    impact_terms = [
        "increased",
        "reduced",
        "improved",
        "optimized",
        "decreased",
        "saved",
        "automated",
        "accelerated",
        "enhanced",
        "delivered",
        "achieved",
        "boosted",
        "streamlined",
        "cut",
        "grew",
        "resolved"
    ]

    for bullet in all_bullets:

        bullet_lower = bullet.lower()

        has_number = bool(re.search(r"\d+|%|percent", bullet_lower))
        has_impact = any(term in bullet_lower for term in impact_terms)

        if has_number or has_impact:
            quantified_count += 1

    score = round((quantified_count / max(len(all_bullets), 1)) * 100, 2)

    return min(score, 100)


def structure_score(resume: OptimizedResume):
    """
    Checks resume completeness.
    """

    score = 0

    if resume.summary and len(resume.summary.split()) >= 25:
        score += 20

    if resume.experiences:
        score += 25

    if resume.projects:
        score += 20

    if resume.education:
        score += 15

    if resume.skills:
        score += 20

    return score


def keyword_stuffing_penalty(resume: OptimizedResume, jd_data: dict):
    """
    Penalizes unnatural keyword repetition.
    """

    resume_text = normalize_text(json.dumps(resume.model_dump()))

    all_keywords = (
        jd_data.get("must_have_keywords", []) +
        jd_data.get("nice_to_have_keywords", []) +
        jd_data.get("tools", [])
    )

    penalty = 0

    for keyword in all_keywords:

        keyword_norm = normalize_text(keyword)

        if not keyword_norm:
            continue

        count = resume_text.count(keyword_norm)

        if count >= 5:
            penalty += 2

    return min(penalty, 5)


# ==============================
# PASS 3: COVERAGE CHECK
# ==============================

def compute_coverage(resume: OptimizedResume, jd_data: dict):

    must_have = jd_data.get("must_have_keywords", [])

    resume_text = json.dumps(resume.model_dump())

    keyword_score, covered, missing = keyword_match_score(
        resume_text,
        must_have
    )

    return keyword_score, covered, missing


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
# ATS KEYWORD + QUALITY SCORING
# ==============================

def compute_ats_score(resume: OptimizedResume, jd_data: dict):

    resume_text = json.dumps(resume.model_dump())

    must_score, must_covered, must_missing = keyword_match_score(
        resume_text,
        jd_data.get("must_have_keywords", [])
    )

    nice_tools_score, nice_tools_covered, nice_tools_missing = keyword_match_score(
        resume_text,
        jd_data.get("nice_to_have_keywords", []) + jd_data.get("tools", [])
    )

    responsibility_score = responsibility_alignment_score(
        resume_text,
        jd_data.get("responsibilities", [])
    )

    quantified_score = quantified_achievement_score(resume)

    structural_score = structure_score(resume)

    stuffing_penalty = keyword_stuffing_penalty(resume, jd_data)

    raw_score = (
        (must_score * 0.35) +
        (nice_tools_score * 0.15) +
        (responsibility_score * 0.20) +
        (quantified_score * 0.15) +
        (structural_score * 0.10) -
        stuffing_penalty
    )

    final_score = round(raw_score, 2)

    # Realistic cap so the score behaves more like online ATS checkers
    final_score = max(0, min(final_score, 89))

    return {
        "final_ats_score": final_score,

        # Same keys as before, so frontend remains compatible
        "keywords_found": must_covered + nice_tools_covered,
        "keywords_missing": must_missing + nice_tools_missing,

        # Extra details; frontend can ignore this if not used
        "score_breakdown": {
            "must_have_keyword_score": must_score,
            "nice_to_have_tools_score": nice_tools_score,
            "responsibility_alignment_score": responsibility_score,
            "quantified_achievement_score": quantified_score,
            "structure_score": structural_score,
            "keyword_stuffing_penalty": stuffing_penalty
        }
    }


# ==============================
# IMPROVEMENT LOOP
# ==============================

def optimize_until_threshold(resume: OptimizedResume, jd_data: dict, threshold=78):

    max_iterations = 3

    for i in range(max_iterations):

        ats = compute_ats_score(resume, jd_data)

        if ats["final_ats_score"] >= threshold:
            return resume, ats, i + 1

        missing = ats["keywords_missing"]
        breakdown = ats.get("score_breakdown", {})

        prompt = f"""
The resume below has an ATS score of {ats["final_ats_score"]}.

It must reach at least {threshold}.

Current score breakdown:
{json.dumps(breakdown, indent=2)}

Missing keywords:
{missing}

Improve the resume by:
- Integrating missing keywords naturally
- Improving responsibility alignment with the job description
- Strengthening quantified achievements
- Making bullets more specific and impact-focused
- Maintaining the same JSON structure
- Not inventing companies, degrees, job titles, dates, or fake experience

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
                "keywords_missing": ats_score["keywords_missing"],
                "score_breakdown": ats_score.get("score_breakdown", {})
            }

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )