"""
Ramjot Dhanjal's optimised version of the ATS Checker - Flask app

Here im gonna list the key Optimizations i have done:
1. Section-aware parsing (Skills, Experience, Education sections weighted differently)
2. Skill clustering and taxonomy (related skills grouped for bonus scoring)
3. Acronym/synonym expansion (ML->Machine Learning, AWS->Amazon Web Services)
4. Position-based weighting (skills in headers/bullets score higher)
5. Missing skill priority ranking (most important missing skills highlighted)

The dependencies are listed in requirements.txt.
"""
import os
import uuid
import threading
import re
from pathlib import Path
from collections import defaultdict

from flask import Flask, render_template, request, redirect, url_for, send_file, flash

# External libs
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from rapidfuzz import fuzz
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Firstly we need the NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / 'tmp_uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

# Here im gonna put my acronym and synonym mappings and this will help make sure there are match variations
SKILL_SYNONYMS = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'dl': 'deep learning',
    'nlp': 'natural language processing',
    'cv': 'computer vision',
    'aws': 'amazon web services',
    'gcp': 'google cloud platform',
    'k8s': 'kubernetes',
    'js': 'javascript',
    'ts': 'typescript',
    'tf': 'tensorflow',
    'api': 'application programming interface',
    'rest': 'representational state transfer',
    'sql': 'structured query language',
    'nosql': 'non-relational database',
    'ci/cd': 'continuous integration continuous deployment',
    'devops': 'development operations',
    'ui': 'user interface',
    'ux': 'user experience',
    'etl': 'extract transform load',
    'kpi': 'key performance indicator',
    'crm': 'customer relationship management',
    'seo': 'search engine optimization',
    'sem': 'search engine marketing'
}

# Here below im going to define my skill clusters, these are basically the related skills grouped together

SKILL_CLUSTERS = {
    'Python Ecosystem': ['python', 'pandas', 'numpy', 'django', 'flask', 'scikit-learn'],
    'Machine Learning': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn'],
    'Cloud Platforms': ['aws', 'azure', 'gcp', 'cloud', 'ec2', 's3', 'lambda'],
    'Frontend Stack': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'typescript'],
    'Backend Stack': ['node.js', 'express', 'django', 'flask', 'spring boot', 'api'],
    'Data Visualization': ['tableau', 'power bi', 'matplotlib', 'seaborn', 'plotly', 'data visualization'],
    'DevOps Tools': ['docker', 'kubernetes', 'jenkins', 'gitlab', 'terraform', 'ansible'],
    'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle']
}

# ============================================================
# So basically after some testing i realised that the embedded database i had was not broad enough to cover all industries
# So here i have greatley expanded the skill clusters to account for these industries.
# EXPANDED CROSS-INDUSTRY SKILL CLUSTERS
# These broaden the ATS beyond computer-science roles.
# Paste this directly under your current SKILL_CLUSTERS.
# ============================================================

SKILL_CLUSTERS.update({

    # === Soft Skills & Workplace Competencies ===
    'Soft Skills': [
        'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking',
        'time management', 'adaptability', 'creativity', 'negotiation', 'collaboration',
        'emotional intelligence', 'decision making', 'conflict resolution'
    ],

    'Business & Management': [
        'project management', 'stakeholder management', 'agile', 'scrum', 'kanban',
        'risk management', 'resource planning', 'business strategy', 'operations management'
    ],

    # === Marketing & Communications ===
    'Marketing': [
        'seo', 'sem', 'ppc', 'content creation', 'copywriting', 'social media',
        'google analytics', 'email marketing', 'brand strategy', 'campaign management'
    ],

    'Creative & Design': [
        'adobe photoshop', 'adobe illustrator', 'adobe premiere', 'figma', 'canva',
        'graphic design', 'video editing', 'motion graphics'
    ],

    # === Sales & Customer-Facing Roles ===
    'Sales': [
        'lead generation', 'crm', 'cold calling', 'account management', 'salesforce',
        'pipeline management', 'b2b sales', 'negotiation', 'deal closing'
    ],

    'Customer Service': [
        'customer support', 'ticketing systems', 'complaint resolution',
        'client relationship management', 'live chat support'
    ],

    # === Finance & Accounting ===
    'Finance & Accounting': [
        'budgeting', 'forecasting', 'financial analysis', 'accounting',
        'bookkeeping', 'excel', 'financial modelling', 'auditing', 'payroll'
    ],

    # === HR & People Operations ===
    'Human Resources': [
        'talent acquisition', 'interviewing', 'onboarding', 'employee relations',
        'training', 'hr systems', 'performance management'
    ],

    # === Legal & Compliance ===
    'Legal & Compliance': [
        'contract review', 'policy drafting', 'regulatory compliance', 'risk assessment',
        'legal research', 'documentation', 'case management'
    ],

    # === Healthcare & Medical ===
    'Healthcare & Nursing': [
        'patient care', 'clinical skills', 'treatment planning', 'assessment',
        'documentation', 'vital signs monitoring', 'safeguarding'
    ],

    # === Education & Teaching ===
    'Education': [
        'lesson planning', 'curriculum development', 'classroom management',
        'assessment', 'instruction', 'teaching', 'special education'
    ],

    # === Administration & Office Operations ===
    'Operations & Admin': [
        'scheduling', 'office management', 'data entry', 'filing', 'document control',
        'inventory management', 'record keeping'
    ],

    # === Engineering & Manufacturing ===
    'Engineering': [
        'cad', 'autocad', 'solidworks', 'mechanical design', 'electrical design',
        'quality assurance', 'manufacturing processes', 'lean manufacturing'
    ],

    'Manufacturing & Production': [
        'quality control', 'assembly', 'machine operation', 'safety compliance',
        'production planning', 'six sigma'
    ],

    # === Logistics, Supply Chain, Transport ===
    'Logistics & Supply Chain': [
        'supply chain management', 'procurement', 'warehouse management',
        'inventory control', 'dispatch', 'fleet management'
    ],

    'Transport & Driving': [
        'route planning', 'vehicle inspection', 'gps navigation',
        'delivery operations', 'safety compliance'
    ],

    # === Hospitality & Retail ===
    'Hospitality': [
        'customer service', 'food safety', 'cash handling', 'inventory',
        'event coordination', 'point of sale'
    ],

    'Retail': [
        'merchandising', 'stock management', 'cashier operations',
        'customer engagement', 'product knowledge'
    ],

    # === Construction & Trades ===
    'Construction & Skilled Trades': [
        'blueprint reading', 'carpentry', 'plumbing', 'electrical work',
        'site safety', 'equipment operation', 'project coordination'
    ],

    # === Security & HSE ===
    'Security & Safety': [
        'risk assessment', 'patrolling', 'incident reporting',
        'first aid', 'emergency response', 'cctv monitoring'
    ],

    # === Hospitality & Events ===
    'Events & Coordination': [
        'event planning', 'vendor management', 'scheduling', 'client communication',
        'coordination', 'ticketing systems'
    ]
})



# This is the original skill sets from base non optimised version
SKILL_SETS = {
    'Data Analyst': [
        'python', 'pandas', 'numpy', 'sql', 'excel', 'tableau', 'power bi', 'statistics',
        'data visualization', 'matplotlib', 'seaborn', 'data cleaning', 'etl', 'data wrangling',
        'reporting', 'insight generation', 'data storytelling'
    ],
    'Data Scientist': [
        'python', 'r', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'machine learning', 'deep learning', 'feature engineering', 'nlp', 'computer vision',
        'predictive modeling', 'data preprocessing', 'statistics', 'mathematics', 'experiment design'
    ],
    'AI Engineer': [
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
        'nlp', 'computer vision', 'transformers', 'llm', 'openai api', 'huggingface',
        'mlops', 'model deployment', 'api integration', 'docker', 'fastapi'
    ],
    'Software Engineer': [
        'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'react', 'node.js',
        'django', 'flask', 'spring boot', 'git', 'rest api', 'microservices', 'testing',
        'unit testing', 'oop', 'design patterns', 'agile', 'scrum'
    ],
    'Full Stack Developer': [
        'html', 'css', 'javascript', 'typescript', 'react', 'vue.js', 'angular',
        'node.js', 'express', 'django', 'flask', 'mongodb', 'postgresql', 'mysql',
        'rest api', 'graphql', 'docker', 'git', 'ci/cd'
    ]
}


def clean_text(text: str) -> str:
    """So this is just basic text cleaning"""
    return ' '.join(text.lower().split())


def expand_acronyms(text: str) -> str:
    """
    SO heres my first main optimsation and its to allow the ATS checker to expand common acronyms to full forms
    basically this will help match skills even when written differently.
    """
    expanded = text.lower()
    for acronym, full_form in SKILL_SYNONYMS.items():
        # Im using word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(acronym) + r'\b'
        expanded = re.sub(pattern, full_form, expanded)
    return expanded


def extract_text_from_resume(path: str, ocr_threshold: int = 50) -> str:
    """
    So this part is the same extraction bit from the original
    its just optimised a little bit to return raw + cleaned versions.

    """
    text = ''
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print('PyMuPDF error:', e)
        text = ''

    if len(text.strip()) >= ocr_threshold:
        return text  # Here i need to return raw text for the section parsing

    # Heres the OCR fallback
    print('Using OCR for', path)
    images = convert_from_path(path, dpi=300)
    ocr_text = ''
    for img in images:
        ocr_text += pytesseract.image_to_string(img)

    return ocr_text


def parse_resume_sections(text: str) -> dict:
    """
    OKAY so this part is new and is part of my optimised version. It is going to parse the resume into sections
    Different sections will have different weights for skill matching
    """
    sections = {
        'skills': '',
        'experience': '',
        'education': '',
        'other': ''
    }

    # Here im going to list the common section headers including case-insensitive patterns
    patterns = {
        'skills': r'(skills?|technical skills?|core competencies|expertise|proficiencies)',
        'experience': r'(experience|work history|employment|professional experience)',
        'education': r'(education|academic|qualifications|degrees?)'
    }

    text_lower = text.lower()

    # Now i need to find the section positions
    section_positions = []
    for section_name, pattern in patterns.items():
        matches = list(re.finditer(pattern, text_lower))
        for match in matches:
            section_positions.append((match.start(), section_name))

    # We need to make sure to sort by position
    section_positions.sort()

    if not section_positions:
        # If there arent any sections found, im going to treat everything as 'other'
        sections['other'] = text
        return sections

    # Now im gonna extract the text for each section
    for i, (start_pos, section_name) in enumerate(section_positions):
        if i < len(section_positions) - 1:
            end_pos = section_positions[i + 1][0]
            sections[section_name] += text[start_pos:end_pos]
        else:
            sections[section_name] += text[start_pos:]

    return sections


def extract_skills_enhanced(text: str, sections: dict = None, apply_weighting: bool = True) -> dict:
    """
    heres another part that i have optimised, basically my program will extract skills with context awareness and will
    return dict with skills and their weights based on their location
    """
    # Firstly i need to expand acronyms first
    expanded_text = expand_acronyms(text)

    stop_words = set(stopwords.words('english'))
    tokens = [t for t in word_tokenize(expanded_text) if t.isalpha() and len(t) > 2]
    tokens = [t for t in tokens if t not in stop_words]

    # And also get unigrams and bigrams
    uni = set(tokens)
    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add(tokens[i] + ' ' + tokens[i + 1])

    # Also try trigrams for the multi-word skills
    trigrams = set()
    for i in range(len(tokens) - 2):
        trigrams.add(tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2])

    all_candidates = uni | bigrams | trigrams

    # Now here im going to make sure the program will weight skills based on where they appear
    skill_weights = defaultdict(float)

    if sections and apply_weighting:
        # Here ill put the section-based weighting
        weights_map = {
            'skills': 2.0,  # Skills section, which is the most important
            'experience': 1.5,  # Experience section put down as quite important
            'education': 1.0,  # Education will be standard weight
            'other': 0.8  # Other sections wil be lower weight
        }

        for section_name, section_text in sections.items():
            section_expanded = expand_acronyms(section_text)
            section_lower = section_expanded.lower()

            for candidate in all_candidates:
                if candidate in section_lower:
                    skill_weights[candidate] += weights_map[section_name]
    else:
        # This one will be for the no weighting, and all the skills will be equal
        for candidate in all_candidates:
            skill_weights[candidate] = 1.0

    return skill_weights


def calculate_cluster_bonus(resume_skills: set) -> float:
    """
    So here is another one of my optimisations. This program will calculate bonus score based on skill cluster completion
    and having related skills together is valuable
    """
    bonus = 0.0

    for cluster_name, cluster_skills in SKILL_CLUSTERS.items():
        cluster_set = set(cluster_skills)
        matched = len(resume_skills.intersection(cluster_set))
        total = len(cluster_set)

        if matched > 0:
            completion_pct = matched / total
            # Bonus increases the non-linearly with completion basically
            if completion_pct >= 0.75:
                bonus += 8  # Heres the high bonus for a nearly complete cluster
            elif completion_pct >= 0.5:
                bonus += 5
            elif completion_pct >= 0.25:
                bonus += 2

    return min(bonus, 20)  # Im gonna put a cap bonus at 20 points


def compare_skills_advanced(resume_skill_weights: dict, jd_text: str, fuzz_threshold: int = 85) -> dict:
    """
    Heres another part of my optimisations where this program will actually compare skills with weighting and fuzzy matching
    """
    # Here i need to extract JD skills
    jd_expanded = expand_acronyms(jd_text)
    jd_skills_dict = extract_skills_enhanced(jd_expanded, sections=None, apply_weighting=False)
    jd_skills = set(jd_skills_dict.keys())
    resume_skills = set(resume_skill_weights.keys())

    print(f"Number of resume skills: {len(resume_skills)}")
    print(f"First 20 resume skills: {list(resume_skills)[:20]}")
    print(f"Number of JD skills: {len(jd_skills)}")
    print(f"First 20 JD skills: {list(jd_skills)[:20]}")

    matched = set()
    match_scores = {}

    # --- Prevent false positives from fuzzy matching ---
    clean_resume_multiword = {s for s in resume_skills if " " in s}

    def should_skip_fuzzy(jd_skill):
        # Skip multi-word JD phrases if resume has no multi-word phrases
        if " " in jd_skill and not clean_resume_multiword:
            return True
        # Skip extremely mismatched word lengths (e.g., 'law' vs 'westlaw')
        if len(jd_skill) > 10:
            return False
        return False

    for jd_skill in jd_skills:

        # Skip fuzzy matching for this JD skill if needed
        if should_skip_fuzzy(jd_skill):
            # Only allow exact matching for multi-word JD skills
            if jd_skill in resume_skills:
                matched.add(jd_skill)
                match_scores[jd_skill] = resume_skill_weights.get(jd_skill, 1.0)
            continue

        # First we have to try the exact match first
        if jd_skill in resume_skills:
            matched.add(jd_skill)
            match_scores[jd_skill] = resume_skill_weights[jd_skill]
            continue

        # Then just do some fuzzy matching
        best_ratio = 0
        best_match = None

        for r_skill in resume_skills:
            # Use partial_ratio but penalize huge length differences
            ratio = fuzz.partial_ratio(jd_skill, r_skill)

            # Penalize if lengths are very different (prevents "law" matching "westlaw")
            len_diff = abs(len(jd_skill) - len(r_skill))
            max_len = max(len(jd_skill), len(r_skill))

            if max_len > 0:
                len_penalty = (len_diff / max_len) * 20  # Up to 20% penalty
                ratio = ratio - len_penalty

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = r_skill

        if best_ratio >= fuzz_threshold:
            print(f"FUZZY MATCH: '{jd_skill}' matched with '{best_match}' (ratio: {best_ratio})")
            matched.add(jd_skill)
            match_scores[jd_skill] = resume_skill_weights.get(best_match, 1.0)

    missing = jd_skills - matched

    # Go back and calculate the weighted score
    if len(jd_skills) > 0:
        # Weight the matched skills
        weighted_matched = sum(match_scores.values())
        max_possible = len(jd_skills) * 2.0  # Max weight is 2.0
        score = int((weighted_matched / max_possible) * 100)
        score = min(score, 100)  # Cap at 100
    else:
        score = 0

    return {
        'matched': sorted(list(matched)),
        'missing': sorted(list(missing)),
        'score': score,
        'match_scores': match_scores
    }




def rank_missing_skills(missing_skills: set, jd_text: str) -> list:
    """
    Here is yet another optimisation and it allows my program to rank missing skills by importance
    The skills mentioned more frequently in JD are more important
    """
    jd_lower = jd_text.lower()
    skill_counts = {}

    for skill in missing_skills:
        # This will count the occurrences in JD
        count = jd_lower.count(skill)
        skill_counts[skill] = count

    # I need to sort by count in descending order
    ranked = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)

    return [skill for skill, count in ranked]


def best_fit_role(resume_skills: set) -> tuple:
    """
    This part is just going to find the best matching role
    """
    best_role = None
    best_pct = 0

    for role, skills in SKILL_SETS.items():
        skills_set = set(skills)
        # Again were gonna use fuzzy matching
        inter = sum(1 for s in skills_set if any(fuzz.partial_ratio(s, r) >= 80 for r in resume_skills))
        pct = int((inter / max(1, len(skills_set))) * 100)

        if pct > best_pct:
            best_pct = pct
            best_role = role

    return best_role, best_pct


def generate_report(resume_text: str, jd_text: str, jd_domain: str = 'AI/CS') -> dict:
    """
    Generates an enhanced ATS report with domain-aware scoring.
    jd_domain: Tag the domain of the Job Description (e.g., 'AI/CS', 'Law').
    """
    # 1. Parse resume into sections
    sections = parse_resume_sections(resume_text)

    # 2. Extract skills with weighting
    resume_skill_weights = extract_skills_enhanced(resume_text, sections, apply_weighting=True)
    resume_skills = set(resume_skill_weights.keys())

    # 3. Compare with JD
    comparison = compare_skills_advanced(resume_skill_weights, jd_text)

    # 4. Calculate cluster bonus
    cluster_bonus = calculate_cluster_bonus(resume_skills)

    # 5. Base and preliminary final score
    base_score = comparison['score']
    preliminary_score = min(base_score + int(cluster_bonus), 100)

    # 6. Detect resume domain
    DOMAIN_MAP = {
        'AI/CS': ['Machine Learning', 'Deep Learning', 'Python Ecosystem', 'Cloud Platforms',
                  'Data Visualization', 'DevOps Tools', 'Databases'],
        'Law': ['Legal & Compliance']
        # Add more domains as needed
    }

    def detect_resume_domain(resume_skills: set) -> str:
        cluster_counts = {domain: 0 for domain in DOMAIN_MAP}
        for domain, clusters in DOMAIN_MAP.items():
            for cluster in clusters:
                if cluster in SKILL_CLUSTERS and resume_skills.intersection(set(SKILL_CLUSTERS[cluster])):
                    cluster_counts[domain] += 1
        primary_domain = max(cluster_counts, key=lambda k: cluster_counts[k])
        if cluster_counts[primary_domain] == 0:
            return 'Other'
        return primary_domain

    resume_domain = detect_resume_domain(resume_skills)
    domain_multiplier = 1.0 if resume_domain == jd_domain else 0.8
    final_score = int(preliminary_score * domain_multiplier)

    # 7. Best-fit role
    role, role_pct = best_fit_role(resume_skills)

    # 8. Rank missing skills
    ranked_missing = rank_missing_skills(set(comparison['missing']), jd_text)

    # 9. Generate improvement tips
    tips = []
    if final_score < 50:
        tips.append('Add more job-specific keywords, especially from the top missing skills.')
    elif final_score < 75:
        tips.append('Good keyword match. Consider emphasizing your skills in a dedicated section.')
    else:
        tips.append('Excellent keyword coverage. Focus on quantifiable achievements now.')

    if cluster_bonus < 10:
        tips.append('Consider developing complementary skills within key technology stacks.')

    if role_pct < 50:
        tips.append('Your skills may be better suited for a different role. Consider tailoring your resume.')

    return {
        'ats_score': final_score,
        'base_score': base_score,
        'cluster_bonus': int(cluster_bonus),
        'matched_skills': comparison['matched'],
        'missing_skills': comparison['missing'],
        'missing_ranked': ranked_missing[:10],
        'best_role': role,
        'best_role_pct': role_pct,
        'tips': tips,
        'sections_found': [k for k, v in sections.items() if v.strip()],
        'resume_domain': resume_domain,
        'jd_domain': jd_domain
    }




def save_pdf_report(report: dict, filename: str):
    """
    Heres another one of my optimisations where i have enhanced the PDF report with new metrics
    """
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    x = 50
    y = height - 50

    c.setFont('Helvetica-Bold', 16)
    c.drawString(x, y, 'Enhanced ATS Resume Report')
    y -= 30

    c.setFont('Helvetica', 12)
    c.drawString(x, y, f"Final ATS Score: {report['ats_score']}%")
    y -= 16
    c.drawString(x, y, f"Base Match Score: {report['base_score']}%")
    y -= 16
    c.drawString(x, y, f"Cluster Bonus: +{report['cluster_bonus']} points")
    y -= 20
    c.drawString(x, y, f"Best-fit Role: {report['best_role']} ({report['best_role_pct']}%)")
    y -= 30

    c.drawString(x, y, 'Matched Skills:')
    y -= 16
    for s in report['matched_skills']:
        c.drawString(x + 10, y, f"✓ {s}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    y -= 10
    c.drawString(x, y, 'Priority Missing Skills (Top 10):')
    y -= 16
    for i, s in enumerate(report['missing_ranked'][:10], 1):
        c.drawString(x + 10, y, f"{i}. {s}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    y -= 10
    c.drawString(x, y, 'Improvement Tips:')
    y -= 16
    for t in report['tips']:
        c.drawString(x + 10, y, f"• {t}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    c.save()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files or request.files['resume'].filename == '':
        flash('Please upload a PDF resume file.', 'danger')
        return redirect(url_for('index'))

    resume_file = request.files['resume']
    jd_text = request.form.get('job_description', '')

    if not jd_text.strip():
        flash('Please paste the job description text.', 'danger')
        return redirect(url_for('index'))

    # Save file
    uid = uuid.uuid4().hex
    filename = f"{uid}.pdf"
    path = UPLOAD_DIR / filename
    resume_file.save(str(path))

    # Extract text
    resume_text = extract_text_from_resume(str(path))

    # DEBUG: Print resume text to check
    print("=" * 50)
    print("RESUME TEXT EXTRACTED:")
    print(resume_text[:500])  # Print first 500 characters
    print("=" * 50)
    print("JD TEXT:")
    print(jd_text[:500])  # Print first 500 characters
    print("=" * 50)
    # Generate enhanced report
    report = generate_report(resume_text, jd_text)

    # Create PDF report
    pdf_name = f"report_{uid}.pdf"
    pdf_path = UPLOAD_DIR / pdf_name
    save_pdf_report(report, str(pdf_path))

    # Cleanup after 120 seconds
    def cleanup_files(*files):
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

    t = threading.Timer(120.0, cleanup_files, args=(str(path), str(pdf_path)))
    t.start()

    return render_template('result.html',
                           report=report,
                           resume_text=resume_text,
                           jd_text=jd_text,
                           pdf_name=pdf_name)


@app.route('/download/<pdf_name>')
def download(pdf_name):
    p = UPLOAD_DIR / pdf_name
    if not p.exists():
        flash('Report not found or expired.', 'warning')
        return redirect(url_for('index'))
    return send_file(str(p), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)