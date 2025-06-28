# ResumeMatch Pro

**Intelligent Resume Analysis and GitHub Project Selection**

ResumeMatch Pro is an AI-powered tool that helps job seekers optimize their resumes and select the most relevant GitHub projects based on job descriptions. It analyzes resume content for keyword matches, evaluates job fit, and integrates with GitHub to generate ATS-optimized project summaries—boosting your chances of landing technical roles.

---

## 🚀 Features

### 🔍 Resume Analysis
- **Profile Fit Evaluation**: AI-powered assessment of how well your profile matches job requirements.
- **Keyword Match Analysis**: Identifies missing keywords and provides ATS optimization recommendations.
- **Category Scoring**: Evaluates skills, experience, education, keywords, and certifications.
- **Selection Percentage**: Calculates overall job match percentage.
- **Interactive Q&A**: Ask specific questions about your resume content.

### 📂 GitHub Integration
- **Smart Project Selection**: Automatically fetches and analyzes GitHub repositories.
- **Job Relevance Matching**: Selects projects based on job description relevance (not GitHub stars).
- **Duplicate Detection**: Avoids selecting projects already mentioned in your resume.
- **Professional Descriptions**: Generates ATS-optimized project descriptions.
- **Bulk Export**: Download all selected project descriptions in a formatted document.

### 📊 Advanced Analytics
- **Reproducible Results**: Deterministic AI responses for consistent analysis.
- **Multi-Model Support**: Choose from 20+ Groq AI models.
- **Visual Reports**: Interactive charts and metrics.
- **Export Options**: PDF and JSON report downloads.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Groq API Key
- GitHub Token *(optional for higher API limits)*

### Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/resumematch-pro.git
cd resumematch-pro
```
Install dependencies:
```bash
pip install -r requirements.txt
```
## Environment Configuration

Create a .env file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
GITHUB_TOKEN=your_github_token_here
```
## Run the application
```bash
streamlit run app.py
```

## 📦 Dependencies
```bash
streamlit >= 1.28.0
python-dotenv >= 1.0.0
groq >= 0.4.0
PyPDF2 >= 3.0.0
fpdf2 >= 2.7.0
pandas >= 2.0.0
tiktoken >= 0.5.0
requests >= 2.31.0
```

## 🔧 Usage

### Basic Resume Analysis

##### 1. Upload your PDF resume.
##### 2. Paste the complete job description.
##### 3. Choose from available Groq models.
##### 4. Run analysis for detailed insights.

### GitHub Project Selection

##### 1. Enter your GitHub username or profile URL.
##### 2. Select number of projects to extract (3-8).
##### 3. Click "Analyze & Select Best GitHub Projects".
##### 4. Download professionally written project descriptions.

## ⚙️ Advanced Features

##### 1. Reproducible analysis with deterministic settings
##### 2. Token management and optimization
##### 3. Robust error handling
##### 4. Export in PDF and JSON formats

## API Configuration

### Groq Models Supported
- `allam-2-7b`
- `compound-beta`
- `compound-beta-mini`
- `deepseek-r1-distill-llama-70b`
- `gemma2-9b-it`
- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`
- `llama3-70b-8192`
- `llama3-8b-8192`
- `meta-llama/llama-4-maverick-17b-128e-instruct`
- `meta-llama/llama-4-scout-17b-16e-instruct`
- `mistral-saba-24b`
- `qwen-qwq-32b`
- `qwen/qwen3-32b`


### Rate Limits
- **With GitHub Token**: 5,000 requests/hour
- **Without Token**: 60 requests/hour

## Architecture

### Core Components

#### Text Processing
- PDF text extraction using PyPDF2
- Unicode normalization for PDF generation
- Token counting with tiktoken

#### AI Integration
- Deterministic prompt engineering
- Reproducible API calls with fixed parameters
- Multi-model support with automatic fallbacks

#### GitHub Analysis
- Repository fetching with language detection
- Keyword-based relevance scoring
- Duplicate project detection

#### Report Generation
- PDF creation with FPDF
- JSON structured data export
- Professional formatting and styling

## Configuration Options

### Analysis Parameters
- **Temperature**: `0.0000000000000001` (for reproducibility)
- **Top-p**: `0.0000000000000001` (for deterministic results)
- **Max Tokens**: Dynamic based on input length
- **Chunk Size**: 3000 characters for large documents

### GitHub Settings
- **Repository Filters**: Excludes forks and user-named repositories
- **Language Detection**: Automatic programming language identification
- **Topic Extraction**: GitHub topics and keywords analysis
- **Relevance Scoring**: Job description keyword matching

## Error Handling

The application includes comprehensive error handling for:
- Invalid PDF files
- API rate limiting
- Network connectivity issues
- Malformed GitHub URLs
- Missing environment variables
- Token limit exceeded scenarios

## Security Considerations

- Environment variables for sensitive API keys
- Input sanitization for PDF generation
- Rate limiting compliance
- No storage of user data
- Secure API communication

## Quick Start

##### 1. **Get API Keys**
   - Sign up for [Groq API](https://console.groq.com/)
   - Optionally get a [GitHub Token](https://github.com/settings/tokens)

##### 2. **Clone & Setup**

## Contributing

##### 1. Fork the repository
##### 2. Create a feature branch (`git checkout -b feature/amazing-feature`)
##### 3. Commit your changes (`git commit -m 'Add amazing feature'`)
##### 4. Push to the branch (`git push origin feature/amazing-feature`)
##### 5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features
- Update documentation for any changes


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**© 2025 T V L BHARATHWAJ**

*ResumeMatch Pro - Intelligent Resume Analysis and GitHub Project Selection*

---





