from dotenv import load_dotenv
import os
import io
import json
import re
import textwrap
import tiktoken
import unicodedata
import streamlit as st
import PyPDF2
from groq import Groq
from fpdf import FPDF
import pandas as pd
import hashlib
import requests
from datetime import datetime
import base64

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

if not GITHUB_TOKEN:
    st.warning("GITHUB_TOKEN not found. GitHub API calls will be limited without authentication.")

client = Groq(api_key=GROQ_API_KEY)

MODEL_OPTIONS = [
    "allam-2-7b",
    "compound-beta",
    "compound-beta-mini",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "mistral-saba-24b",
    "qwen-qwq-32b",
    "qwen/qwen3-32b",
    "distil-whisper-large-v3-en",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "playai-tts",
    "playai-tts-arabic",
]

def sanitize_text(s: str) -> str:
    """Normalize Unicode to Latin-1–safe text for FPDF."""
    if not s:
        return ""
    nk = unicodedata.normalize("NFKD", s)
    clean_text = nk.encode("latin-1", "ignore").decode("latin-1")
    clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', clean_text)
    return clean_text

def count_tokens(text: str, model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def generate_deterministic_seed(job_desc: str, resume_text: str, analysis_type: str) -> int:
    """Generate a consistent seed based on input content for reproducible results."""
    combined_text = f"{job_desc}_{resume_text}_{analysis_type}"
    hash_object = hashlib.md5(combined_text.encode())
    return int(hash_object.hexdigest()[:8], 16) % 1000000

def get_deterministic_params(system_prompt: str, job_desc: str, model: str):
    """Get deterministic parameters for consistent results."""
    used = count_tokens(system_prompt + job_desc, model)
    max_tokens = max(512, 8192 - used - 200)
    
    temperature = 0.0000000000000001
    top_p = 0.0000000000000001
    
    return max_tokens, temperature, top_p

def extract_text_from_pdf(f) -> str:
    reader = PyPDF2.PdfReader(f)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text: str, max_chars: int = 3000):
    return textwrap.wrap(text, max_chars)

def parse_category_scores(text: str) -> dict:
    cats = {}
    for key in ["skills", "experience", "education", "keywords", "certifications"]:
        m = re.search(rf"{key}\s*[:\-]\s*(\d{{1,3}})", text, re.IGNORECASE)
        cats[key.title()] = int(m.group(1)) if m else 0
    return cats

def safe_get_string(value, default=""):
    """Safely get string value, handling None cases."""
    if value is None:
        return default
    return str(value)

def fetch_github_repositories_exclude_user(username: str) -> list:
    """Fetch all repositories from a GitHub user excluding user-named repos."""
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    
    try:
        url = f"https://api.github.com/users/{username}/repos"
        params = {
            'sort': 'updated',
            'direction': 'desc',
            'per_page': 100,
            'type': 'owner'
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        repos = response.json()
        
        filtered_repos = []
        for repo in repos:
            if not repo.get('fork', False):
                # Exclude repos with name same as username
                if safe_get_string(repo.get('name', '')).lower() != username.lower():
                    repo_data = {
                        'name': safe_get_string(repo.get('name', '')),
                        'description': safe_get_string(repo.get('description', '')),
                        'html_url': safe_get_string(repo.get('html_url', '')),
                        'language': safe_get_string(repo.get('language', '')),
                        'languages_url': safe_get_string(repo.get('languages_url', '')),
                        'stargazers_count': repo.get('stargazers_count', 0),
                        'forks_count': repo.get('forks_count', 0),
                        'created_at': safe_get_string(repo.get('created_at', '')),
                        'updated_at': safe_get_string(repo.get('updated_at', '')),
                        'topics': repo.get('topics', []) or [],
                        'size': repo.get('size', 0)
                    }
                    
                    try:
                        if repo_data['languages_url']:
                            lang_response = requests.get(repo_data['languages_url'], headers=headers)
                            if lang_response.status_code == 200:
                                languages_data = lang_response.json()
                                repo_data['languages'] = [safe_get_string(lang) for lang in languages_data.keys() if lang]
                            else:
                                repo_data['languages'] = [repo_data['language']] if repo_data['language'] else []
                        else:
                            repo_data['languages'] = [repo_data['language']] if repo_data['language'] else []
                    except:
                        repo_data['languages'] = [repo_data['language']] if repo_data['language'] else []
                    
                    filtered_repos.append(repo_data)
        
        return filtered_repos
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GitHub repositories: {str(e)}")
        return []

def extract_github_username(github_url: str) -> str:
    """Extract username from GitHub URL."""
    patterns = [
        r'github\.com/([^/]+)/?$',
        r'github\.com/([^/]+)/.*',
        r'^([^/]+)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, github_url.strip())
        if match:
            return match.group(1)
    
    return github_url.strip()

def extract_existing_projects_from_resume(resume_text: str) -> list:
    """Extract existing projects from resume for comparison."""
    existing_projects = []
    
    project_headers = [
        r'UNIVERSITY PROJECTS?',
        r'PROJECTS?',
        r'KEY PROJECTS?',
        r'RELEVANT PROJECTS?',
        r'TECHNICAL PROJECTS?',
        r'PERSONAL PROJECTS?',
        r'ACADEMIC PROJECTS?'
    ]
    
    for header in project_headers:
        pattern = rf'({header})\s*\n(.*?)(?=\n[A-Z][A-Z\s]+\n|\n\n[A-Z]|\Z)'
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            projects_content = match.group(2).strip()
            
            lines = projects_content.split('\n')
            current_project = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if not line.startswith('•') and not line.startswith('-') and len(line) > 10 and not line.startswith('Technologies:') and not line.startswith('GitHub:'):
                    if current_project:
                        existing_projects.append(current_project)
                    current_project = {
                        'title': line,
                        'description': '',
                        'source': 'resume'
                    }
                elif current_project and line:
                    current_project['description'] += line + '\n'
            
            if current_project:
                existing_projects.append(current_project)
            break
    
    return existing_projects

def compare_and_select_projects(repositories: list, existing_projects: list, job_description: str, model_choice: str, max_projects: int) -> list:
    """Compare GitHub repos with existing resume projects and select the best ones based on job relevance only."""
    
    # Create comparison data with safe string handling
    existing_titles = [safe_get_string(proj.get('title', '')).lower() for proj in existing_projects]
    existing_keywords = set()
    for proj in existing_projects:
        title = safe_get_string(proj.get('title', ''))
        description = safe_get_string(proj.get('description', ''))
        existing_keywords.update(title.lower().split())
        existing_keywords.update(description.lower().split())
    
    # Score repositories based on job relevance only (no stars consideration)
    scored_repos = []
    for repo in repositories:
        repo_name = safe_get_string(repo.get('name', ''))
        repo_name_lower = repo_name.lower().replace('-', ' ').replace('_', ' ')
        
        # Check if similar project already exists in resume
        similarity_penalty = 0
        for existing_title in existing_titles:
            if any(word in existing_title for word in repo_name_lower.split() if word):
                similarity_penalty += 5  # Increased penalty for duplicates
        
        # Job relevance score (keyword matching only)
        job_keywords = set(safe_get_string(job_description, '').lower().split())
        repo_keywords = set()
        
        # Safely add repository keywords
        repo_keywords.update(repo_name.lower().split())
        
        # Safely handle description
        repo_description = safe_get_string(repo.get('description', ''))
        if repo_description:
            repo_keywords.update(repo_description.lower().split())
        
        # Safely handle languages
        languages = repo.get('languages', []) or []
        for lang in languages:
            if lang:
                repo_keywords.add(safe_get_string(lang).lower())
        
        # Safely handle topics
        topics = repo.get('topics', []) or []
        for topic in topics:
            if topic:
                repo_keywords.add(safe_get_string(topic).lower())
        
        # Calculate job relevance score
        relevance_score = len(job_keywords.intersection(repo_keywords))
        
        # Add bonus for exact keyword matches in project name and description
        exact_matches = 0
        for keyword in job_keywords:
            if keyword in repo_name.lower() or keyword in repo_description.lower():
                exact_matches += 2
        
        # Final score calculation (ONLY job relevance, NO stars)
        final_score = relevance_score + exact_matches - similarity_penalty
        
        scored_repos.append((repo, final_score, relevance_score))
    
    # Sort by job relevance score only and select top projects
    scored_repos.sort(key=lambda x: x[1], reverse=True)
    selected_repos = [repo for repo, score, relevance in scored_repos[:max_projects]]
    
    return selected_repos

def generate_project_descriptions_for_download(selected_projects: list, job_description: str, model_choice: str) -> str:
    """Generate optimized project descriptions for download."""
    
    descriptions_text = "SELECTED GITHUB PROJECTS - OPTIMIZED FOR JOB APPLICATION\n"
    descriptions_text += "=" * 60 + "\n"
    descriptions_text += f"Selection Criteria: Job Relevance Only (No GitHub Stars Considered)\n"
    descriptions_text += "=" * 60 + "\n\n"
    
    for i, project in enumerate(selected_projects, 1):
        # Create a professional title with safe string handling
        project_name = safe_get_string(project.get('name', f'Project_{i}'))
        title = project_name.replace('-', ' ').replace('_', ' ').title()
        
        # Generate description using AI
        project_description = safe_get_string(project.get('description', 'No description'))
        languages = project.get('languages', []) or []
        topics = project.get('topics', []) or []
        
        description_prompt = f"""
        You are a professional resume writer. Create a compelling, ATS-optimized project description for this GitHub project that aligns with the job requirements.
        
        Job Description:
        {safe_get_string(job_description, '')}
        
        Project Details:
        - Name: {project_name}
        - Description: {project_description}
        - Languages: {', '.join([safe_get_string(lang) for lang in languages if lang])}
        - Topics: {', '.join([safe_get_string(topic) for topic in topics if topic])}
        - GitHub URL: {safe_get_string(project.get('html_url', ''))}
        
        Write a professional project description with 2-3 bullet points that:
        - Highlights the project's key features and impact
        - Uses keywords from the job description
        - Shows technical skills and achievements
        - Is ATS-optimized and professional
        
        Format your response as:
        TITLE: [Professional project title]
        DESCRIPTION: 
        • [First bullet point]
        • [Second bullet point]
        • [Third bullet point if needed]
        TECHNOLOGIES: [Comma-separated list of technologies]
        """
        
        try:
            mt, temp, tp = get_deterministic_params(description_prompt, job_description, model_choice)
            
            messages = [
                {"role": "system", "content": "You are a professional resume writer. Create compelling project descriptions that match job requirements."},
                {"role": "user", "content": description_prompt}
            ]
            
            response = make_api_call_with_reproducibility(
                client, model_choice, messages, mt, temp, tp
            )
            
            if response:
                content = response.choices[0].message.content.strip()
                
                # Parse the response manually
                title_match = re.search(r'TITLE:\s*(.+)', content, re.IGNORECASE)
                desc_match = re.search(r'DESCRIPTION:\s*(.*?)(?=TECHNOLOGIES:|$)', content, re.IGNORECASE | re.DOTALL)
                tech_match = re.search(r'TECHNOLOGIES:\s*(.+)', content, re.IGNORECASE)
                
                if title_match:
                    title = title_match.group(1).strip()
                
                if desc_match:
                    description = desc_match.group(1).strip()
                else:
                    description = f"• Developed {project_description if project_description != 'No description' else 'a comprehensive software project showcasing technical skills'}\n• Implemented using {', '.join([safe_get_string(lang) for lang in languages if lang]) or 'modern technologies'}\n• Demonstrates proficiency in software development and problem-solving"
                
                if tech_match:
                    technologies = tech_match.group(1).strip()
                else:
                    technologies = ', '.join([safe_get_string(lang) for lang in languages if lang]) or 'Python, JavaScript'
                
            else:
                description = f"• Developed {project_description if project_description != 'No description' else 'a comprehensive software project showcasing technical skills'}\n• Implemented using {', '.join([safe_get_string(lang) for lang in languages if lang]) or 'modern technologies'}\n• Demonstrates proficiency in software development and problem-solving"
                technologies = ', '.join([safe_get_string(lang) for lang in languages if lang]) or 'Python, JavaScript'
        
        except Exception as e:
            description = f"• Developed {project_description if project_description != 'No description' else 'a comprehensive software project showcasing technical skills'}\n• Implemented using {', '.join([safe_get_string(lang) for lang in languages if lang]) or 'modern technologies'}\n• Demonstrates proficiency in software development and problem-solving"
            technologies = ', '.join([safe_get_string(lang) for lang in languages if lang]) or 'Python, JavaScript'
        
        # Add to descriptions text
        descriptions_text += f"PROJECT {i}: {title}\n"
        descriptions_text += f"GitHub: {safe_get_string(project.get('html_url', ''))}\n"
        descriptions_text += f"Languages: {', '.join([safe_get_string(lang) for lang in languages if lang])}\n"
        descriptions_text += f"Topics: {', '.join([safe_get_string(topic) for topic in topics if topic])}\n\n"
        descriptions_text += f"{description}\n\n"
        descriptions_text += f"Technologies: {technologies}\n"
        descriptions_text += "-" * 50 + "\n\n"
    
    descriptions_text += f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    descriptions_text += "Selection Method: Job Relevance Matching Only\n"
    descriptions_text += "© 2025 ResumeMatch Pro - T V L BHARATHWAJ\n"
    
    return descriptions_text

def make_api_call_with_reproducibility(client, model_choice, messages, max_tokens, temperature, top_p):
    """Make API call with reproducibility parameters."""
    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return None

def generate_pdf(report: dict) -> bytes:
    """Generate PDF report and return as bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 14)
    pdf.cell(0, 10, sanitize_text("ResumeMatch Pro Analysis Report"), ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)

    try:
        pdf.multi_cell(0, 6, sanitize_text("Job Description:\n" + report["job_description"][:1000] + "..."))
        pdf.ln(4)

        if "profile_fit" in report:
            pdf.multi_cell(0, 6, sanitize_text("Profile Fit Evaluation:\n" + report["profile_fit"][:1000] + "..."))
            pdf.ln(4)

        if "keyword_match" in report:
            pdf.multi_cell(0, 6, sanitize_text("Keyword Match Results:\n" + report["keyword_match"][:1000] + "..."))
            pdf.ln(4)

        if "categories" in report:
            pdf.multi_cell(0, 6, sanitize_text("Category Scores:"))
            for cat, score in report["categories"].items():
                pdf.multi_cell(0, 6, sanitize_text(f"  {cat}: {score}%"))
            pdf.ln(2)
            pdf.multi_cell(
                0, 6,
                sanitize_text(f"Selection Percentage (average): {report['selection_percentage']}%")
            )
            pdf.ln(4)

        pdf.set_font("Arial", "", 8)
        pdf.cell(0, 10, sanitize_text("© 2025 T V L BHARATHWAJ"), ln=True, align="C")

    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "Resume Analysis Report", ln=True, align="C")
        pdf.ln(5)
        pdf.multi_cell(0, 6, "Report generation encountered an error. Please try again.")

    return bytes(pdf.output(dest="S"))

# Streamlit UI
st.set_page_config(page_title="ResumeMatch Pro", layout="wide")
st.markdown("<h1 style='text-align:center;'>🔍 ResumeMatch Pro</h1>", unsafe_allow_html=True)
st.markdown("---")

system_prompt = "You are an expert Technical HR Manager and ATS analyzer."

col1, col2 = st.columns([1, 2], gap="large")
with col1:
    st.header("Inputs")
    job_desc = st.text_area(
        "Job Description", height=200,
        help="Paste the complete job description here."
    )
    
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    st.subheader("GitHub Integration")
    github_url = st.text_input(
        "GitHub Profile URL or Username",
        placeholder="https://github.com/username or just username",
        help="Enter your GitHub profile URL or username to fetch projects"
    )
    
    max_projects = st.slider(
        "Number of Projects to Select",
        min_value=3,
        max_value=8,
        value=5,
        help="Number of best projects to select from GitHub based on job relevance only"
    )
    
    model_choice = st.selectbox(
        "Select Groq Model", MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("llama3-70b-8192")
    )
    
    resume_text = extract_text_from_pdf(resume_file) if resume_file else None
    if resume_file:
        st.success("Resume loaded successfully.")

with col2:
    st.header("Analysis & Results")
    report = {"job_description": job_desc}

    # GitHub Projects Analysis and Selection
    if st.button("🔍 Analyze & Select Best GitHub Projects (Job Relevance Only)") and job_desc and resume_text and github_url:
        username = extract_github_username(github_url)
        
        with st.spinner(f"Fetching repositories from GitHub user: {username} (excluding user-named repos)..."):
            repositories = fetch_github_repositories_exclude_user(username)
        
        if repositories:
            st.success(f"Found {len(repositories)} repositories (excluding user-named repo)")
            
            with st.spinner("Analyzing existing resume projects..."):
                existing_projects = extract_existing_projects_from_resume(resume_text)
            
            if existing_projects:
                st.info(f"Found {len(existing_projects)} existing projects in resume")
                with st.expander("📋 Current Resume Projects"):
                    for i, proj in enumerate(existing_projects, 1):
                        st.write(f"**{i}. {safe_get_string(proj.get('title', ''))}**")
                        st.write(f"{safe_get_string(proj.get('description', ''))[:200]}...")
                        st.markdown("---")
            else:
                st.info("No existing projects found in resume.")
            
            with st.spinner(f"Selecting best {max_projects} projects based on job relevance only..."):
                selected_projects = compare_and_select_projects(repositories, existing_projects, job_desc, model_choice, max_projects)
            
            if selected_projects:
                st.success(f"✅ Selected {len(selected_projects)} best projects based on job relevance (no GitHub stars considered)")
                
                st.subheader("Selected GitHub Projects")
                st.info("🎯 Projects selected based on job description relevance only")
                
                for i, project in enumerate(selected_projects, 1):
                    project_name = safe_get_string(project.get('name', f'Project_{i}'))
                    with st.expander(f"{i}. {project_name} (Selected for Job Relevance)"):
                        st.write(f"**Description:** {safe_get_string(project.get('description', 'No description'))}")
                        languages = project.get('languages', []) or []
                        st.write(f"**Languages:** {', '.join([safe_get_string(lang) for lang in languages if lang])}")
                        topics = project.get('topics', []) or []
                        st.write(f"**Topics:** {', '.join([safe_get_string(topic) for topic in topics if topic])}")
                        st.write(f"**URL:** {safe_get_string(project.get('html_url', ''))}")
                        st.write(f"**Last Updated:** {safe_get_string(project.get('updated_at', ''))[:10]}")
                        st.write(f"**GitHub Stars:** {project.get('stargazers_count', 0)} (not used in selection)")
                
                with st.spinner("Generating optimized project descriptions..."):
                    project_descriptions = generate_project_descriptions_for_download(selected_projects, job_desc, model_choice)
                
                st.subheader("✅ Project Descriptions Generated!")
                st.markdown("### Preview:")
                
                # Show preview of first project
                preview_lines = project_descriptions.split('\n')[:25]
                st.text('\n'.join(preview_lines) + '\n...')
                
                # Download button for project descriptions
                st.download_button(
                    label="📥 Download Project Descriptions (Job Relevance Based)",
                    data=project_descriptions.encode('utf-8'),
                    file_name=f"job_relevant_github_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Download the job-relevance optimized project descriptions for your resume"
                )
                
                # Show comparison summary
                st.subheader("📊 Selection Summary")
                col_summary1, col_summary2 = st.columns(2)
                
                with col_summary1:
                    st.metric("Existing Resume Projects", len(existing_projects))
                    st.metric("Available GitHub Repos", len(repositories))
                
                with col_summary2:
                    st.metric("Selected New Projects", len(selected_projects))
                    st.metric("Selection Criteria", "Job Relevance Only")
                
                st.info("🎯 Selection based purely on job description keyword matching and relevance, not GitHub popularity")
                
        else:
            st.error("No repositories found or error fetching from GitHub")

    # Original analysis features
    st.markdown("---")
    st.subheader("Resume Analysis Features")
    
    # 1. Enhanced Profile Fit with Reproducibility
    if st.button("Analyze Profile Fit") and job_desc and resume_text:
        seed = generate_deterministic_seed(job_desc, resume_text, "profile_fit")
        mt, temp, tp = get_deterministic_params(system_prompt, job_desc, model_choice)
        
        fit_prompt = (
            "You are a seasoned Technical HR Manager. "
            "Use deep domain knowledge to evaluate the candidate's profile against the job description. "
            "Provide your analysis in exactly this format:\n\n"
            "**FIT SCORE: [X]%**\n\n"
            "**TOP 3 STRENGTHS:**\n"
            "1. [Strength with specific example]\n"
            "2. [Strength with specific example]\n"
            "3. [Strength with specific example]\n\n"
            "**TOP 3 GAPS/RISKS:**\n"
            "1. [Gap with actionable suggestion]\n"
            "2. [Gap with actionable suggestion]\n"
            "3. [Gap with actionable suggestion]\n\n"
            "Be specific and reference exact details from the resume."
        )
        
        msgs = [
            {"role": "system", "content": fit_prompt},
            {"role": "user", "content": f"Job Description:\n{job_desc}\n\nResume Text:\n{resume_text}"}
        ]
        
        with st.spinner("Analyzing profile fit…"):
            r = make_api_call_with_reproducibility(
                client, model_choice, msgs, mt, temp, tp
            )
        
        if r:
            pf = r.choices[0].message.content
            report["profile_fit"] = pf
            st.subheader("Profile Fit Evaluation")
            st.markdown(pf)

    # 2. Enhanced Keyword Match with Reproducibility
    if st.button("Calculate Keyword Match") and job_desc and resume_text:
        seed = generate_deterministic_seed(job_desc, resume_text, "keyword_match")
        mt, temp, tp = get_deterministic_params(system_prompt, job_desc, model_choice)
        
        kw_prompt = (
            "You are an ATS optimization expert. "
            "Analyze keyword alignment between the resume and job description. "
            "Provide your analysis in exactly this format:\n\n"
            "**KEYWORD MATCH PERCENTAGE: [X]%**\n\n"
            "**10 CRITICAL MISSING KEYWORDS:**\n"
            "1. [keyword]\n2. [keyword]\n3. [keyword]\n4. [keyword]\n5. [keyword]\n"
            "6. [keyword]\n7. [keyword]\n8. [keyword]\n9. [keyword]\n10. [keyword]\n\n"
            "**INTEGRATION RECOMMENDATIONS:**\n"
            "- [Specific recommendation 1]\n"
            "- [Specific recommendation 2]\n"
            "- [Specific recommendation 3]\n"
        )
        
        msgs = [
            {"role": "system", "content": kw_prompt},
            {"role": "user", "content": f"Job Description:\n{job_desc}\n\nResume Text:\n{resume_text}"}
        ]
        
        with st.spinner("Calculating keyword match…"):
            r = make_api_call_with_reproducibility(
                client, model_choice, msgs, mt, temp, tp
            )
        
        if r:
            km = r.choices[0].message.content
            report["keyword_match"] = km
            st.subheader("Keyword Match Results")
            st.markdown(km)

    # 3. Selection Percentage & Category Breakdown with Reproducibility
    if st.button("Evaluate Selection Percentage") and job_desc and resume_text:
        seed = generate_deterministic_seed(job_desc, resume_text, "selection_percentage")
        mt, temp, tp = get_deterministic_params(system_prompt, job_desc, model_choice)
        
        sel_prompt = (
            "You are an ATS-savvy analyst. "
            "Score the candidate (0–100) in each category and return ONLY a valid JSON object. "
            "Use exactly this format:\n"
            "{\n"
            '  "skills": [score 0-100],\n'
            '  "experience": [score 0-100],\n'
            '  "education": [score 0-100],\n'
            '  "keywords": [score 0-100],\n'
            '  "certifications": [score 0-100]\n'
            "}\n"
            "Return only the JSON object, no additional text."
        )
        
        msgs = [
            {"role": "system", "content": sel_prompt},
            {"role": "user", "content": f"Job Description:\n{job_desc}\n\nResume Text:\n{resume_text}"}
        ]
        
        with st.spinner("Calculating selection percentages…"):
            r = make_api_call_with_reproducibility(
                client, model_choice, msgs, mt, temp, tp
            )
        
        if r:
            raw = r.choices[0].message.content
            try:
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    cats = {k.title(): data.get(k, 0) for k in
                            ["skills", "experience", "education", "keywords", "certifications"]}
                else:
                    raise json.JSONDecodeError("No JSON found", raw, 0)
            except json.JSONDecodeError:
                cats = parse_category_scores(raw)
            
            sel_pct = round(sum(cats.values()) / len(cats)) if cats else 0
            positive = [c for c, s in cats.items() if s >= sel_pct]
            negative = [c for c, s in cats.items() if s < sel_pct]
            
            report.update({
                "categories": cats,
                "selection_percentage": sel_pct,
                "positive_categories": positive,
                "negative_categories": negative
            })
            
            st.subheader("Category Scores & Selection Percentage")
            with st.expander("How Selection Percentage Is Computed"):
                st.markdown("""
                **Selection Percentage** = (skills + experience + education + keywords + certifications) / 5  
                • Each category is scored 0–100 by the AI based on relevance to the job description.  
                • Categories ≥ average are strength areas; categories < average are areas for improvement.  
                """)
            
            df = pd.DataFrame.from_dict(cats, orient="index", columns=["Score"])
            st.bar_chart(df, use_container_width=True)
            st.metric("Selection Percentage", f"{sel_pct}%")
            st.write("**Strength Categories:**", ", ".join(positive))
            st.write("**Improvement Categories:**", ", ".join(negative))

    # 4. Q&A with Reproducibility
    st.markdown("---")
    st.subheader("Ask Anything About the Resume")
    question = st.text_input("Your Question", help="Ask about skills, experience, or qualifications.")
    if st.button("Get Answer") and question and job_desc and resume_text:
        seed = generate_deterministic_seed(job_desc + question, resume_text, "qa")
        mt, temp, tp = get_deterministic_params(system_prompt, job_desc + question, model_choice)
        
        context = "\n\n".join(chunk_text(resume_text)[:2])
        msgs = [
            {"role": "system", "content": "You are a helpful HR assistant. Provide detailed, specific answers based on the resume content."},
            {"role": "user", "content": f"Job Description:\n{job_desc}\n\nResume Excerpt:\n{context}\n\nQuestion: {question}"}
        ]
        
        with st.spinner("Fetching answer…"):
            r = make_api_call_with_reproducibility(
                client, model_choice, msgs, mt, temp, tp
            )
        
        if r:
            qa = r.choices[0].message.content
            report["qa_answer"] = qa
            st.markdown(qa)

    # Download all results
    if any(k in report for k in ("profile_fit", "keyword_match", "categories", "qa_answer")):
        with st.spinner("Generating analysis report..."):
            try:
                pdf_bytes = generate_pdf(report)
                
                col_report1, col_report2 = st.columns(2)
                
                with col_report1:
                    st.download_button(
                        label="📄 Download Analysis Report (PDF)",
                        data=pdf_bytes,
                        file_name="resume_analysis_report.pdf",
                        mime="application/pdf"
                    )
                
                with col_report2:
                    st.download_button(
                        label="📊 Download Results (JSON)",
                        data=json.dumps(report, indent=2).encode('utf-8'),
                        file_name="resume_analysis_report.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"Error generating reports: {str(e)}")
                st.info("Please try downloading individual sections.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2025 T V L BHARATHWAJ</p>",
    unsafe_allow_html=True
)
