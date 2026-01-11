# Context-Aware ATS Resume Analyzer
**AI-Powered Resume Optimization Using Natural Language Processing**

## Overview
An intelligent resume analysis system leveraging AI and NLP techniques to evaluate resume-job description compatibility. Unlike traditional keyword-matching ATS systems, this application uses context-aware algorithms to understand document structure and skill relationships.

## AI & Machine Learning Techniques

### Natural Language Processing (NLP)
- **Tokenization & N-gram Analysis**: Extracts unigrams, bigrams, and trigrams using NLTK for comprehensive skill identification
- **Semantic Understanding**: Fuzzy string matching with Levenshtein distance algorithm captures skill variations and abbreviations
- **Stopword Filtering**: ML-based text preprocessing to remove linguistic noise

### Intelligent Parsing Algorithms
- **Section Recognition**: Pattern matching algorithms identify resume structure (Skills, Experience, Education)
- **Contextual Weighting**: AI-driven scoring that weights skills differently based on document location (2.0x for Skills section, 1.5x for Experience)
- **Acronym Expansion**: Knowledge-base driven synonym mapping (ML → Machine Learning, AWS → Amazon Web Services)

### Advanced Scoring Mechanisms
- **Skill Clustering**: Taxonomy-based analysis identifying complementary skill groups (e.g., Python Ecosystem, Cloud Platforms)
- **Non-linear Bonus System**: Rewards candidates with comprehensive technology stacks
- **Priority Ranking Algorithm**: Frequency-based analysis to identify most critical missing skills

## Key Features
- ✅ **AI-powered section-aware parsing** - Understands resume structure intelligently
- ✅ **Fuzzy matching algorithm** - Handles spelling variations and partial matches (75% similarity threshold)
- ✅ **Skill clustering & taxonomy** - Recognizes related skill groups using curated knowledge bases
- ✅ **Automated acronym expansion** - Pre-trained synonym mappings for industry terminology
- ✅ **Explainable AI** - Transparent scoring allows users to understand decision-making
- ✅ **OCR integration** - Fallback text extraction for scanned documents using Tesseract

## Technical Stack

### Core AI/NLP Libraries
```
flask              # Web framework
PyMuPDF           # PDF text extraction
pdf2image         # Image processing for OCR
pytesseract       # Optical Character Recognition (AI-based)
rapidfuzz         # Fuzzy string matching algorithms
nltk              # Natural Language Processing toolkit
reportlab         # Report generation
```

### AI Components
- **NLTK**: Tokenization, stopword filtering, linguistic analysis
- **RapidFuzz**: Levenshtein distance calculation for fuzzy matching
- **Tesseract OCR**: Neural network-based optical character recognition
- **Custom Rule Engine**: Knowledge-based AI system with curated taxonomies

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install AI/NLP system dependencies:**

**Tesseract OCR (Neural Network-based OCR):**
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Add to system PATH

**Poppler (PDF Processing):**
- Windows: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add `bin` folder to PATH

## Usage

1. **Start the AI-powered analysis server:**
```bash
python app.py
```

2. **Access the web interface:**
```
http://127.0.0.1:5000
```

3. **Analyze resume-job match:**
   - Upload PDF resume
   - Paste job description
   - Receive AI-powered compatibility score with detailed insights

## Algorithm Architecture

### Rule-Based AI with NLP Pipeline
```
Input (PDF) → Text Extraction → Preprocessing
     ↓
Acronym Expansion → Section Detection
     ↓
Skill Extraction (N-grams) → Contextual Weighting
     ↓
Fuzzy Matching → Cluster Analysis
     ↓
Final Score + Recommendations
```

### Intelligent Scoring Formula
```
Final Score = Weighted Match Score + Cluster Completion Bonus
```

Where:
- **Weighted Match Score**: Skills scored based on document section context
- **Cluster Completion Bonus**: Non-linear rewards for related skill groups
  - 75%+ cluster: +8 points
  - 50-75% cluster: +5 points
  - 25-50% cluster: +2 points

## AI Advantages Over Traditional ATS

| Traditional ATS | This AI System |
|----------------|----------------|
| Simple keyword counting | Context-aware structural analysis |
| No synonym recognition | Intelligent acronym expansion |
| Flat skill weighting | Section-based differential weighting |
| Isolated skill matching | Cluster-based complementary analysis |
| Black box scoring | Explainable AI with transparency |

## Project Structure
```
ATS_Optimized_Ramjot/
├── app.py                 # Main AI algorithm implementation
├── requirements.txt       # AI/ML dependencies
├── templates/            # User interface
│   ├── index.html        # Upload interface
│   └── result.html       # AI analysis results
└── README.md             # Documentation
```

## AI Research & Development
This system represents a **rule-based AI approach** to resume analysis, chosen for:
- **Transparency**: Explainable decision-making vs. black-box ML
- **Efficiency**: Real-time processing without GPU requirements  
- **Adaptability**: Easy updates to knowledge bases as technology evolves
- **Determinism**: Consistent, reproducible results

Future enhancements could incorporate:
- Deep learning embeddings (Word2Vec, BERT) for semantic similarity
- Temporal analysis with employment date parsing
- Multi-language support with translation APIs
- Active learning from user feedback

## Author
**Ramjot Dhanjal**  
rd0304k@gre.ac.uk  
School of Computing and Mathematical Sciences  
University of Greenwich

---

*Developed as part of COMP1827 Introduction to Artificial Intelligence coursework, demonstrating practical application of NLP, rule-based AI, and knowledge representation techniques.*