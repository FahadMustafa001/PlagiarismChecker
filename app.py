# app.py (ApexAudit AI - End-Level Professional)
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import sqlite3
import re
import requests
import json
import io
import threading
import time
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np

# Advanced Core AI Libraries
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF Export
from fpdf import FPDF

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Download NLTK requirements quietly
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secure_extreme_key_change_this_in_production")
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['DATABASE_PATH'] = os.getenv("DATABASE_PATH", "plagiarism_checker.db")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global Model Initialization (Loads once when server starts)
print("Loading Advanced NLP Semantic Model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# SSE Progress Tracking (thread-safe)
scan_progress = {}
progress_lock = threading.Lock()

def set_progress(sid, step, message, percent):
    with progress_lock:
        scan_progress[sid] = {"step": step, "message": message, "percent": percent}

def get_progress(sid):
    with progress_lock:
        return scan_progress.get(sid, {"step": 0, "message": "Waiting...", "percent": 0})

# ---------------------- Database Setup & Migration ----------------------
def get_db_connection():
    return sqlite3.connect(app.config['DATABASE_PATH'])

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    
    cur.execute('''CREATE TABLE IF NOT EXISTS assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    filename TEXT,
                    plagiarism REAL,
                    ai_generated_score REAL,
                    content TEXT,
                    scan_report TEXT,
                    model_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id))''')
                    
    # Dynamic Database Schema Migrations
    cur.execute("PRAGMA table_info(assignments)")
    columns = [col[1] for col in cur.fetchall()]
    
    migrations = [
        ("ai_generated_score", "REAL DEFAULT 0.0"),
        ("content", "TEXT DEFAULT ''"),
        ("scan_report", "TEXT DEFAULT '{}'"),
        ("model_used", "TEXT DEFAULT 'Semantic'"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    ]
    
    for col_name, col_type in migrations:
        if col_name not in columns:
            try:
                cur.execute(f"ALTER TABLE assignments ADD COLUMN {col_name} {col_type}")
                print(f"Database Migration: Added column '{col_name}'.")
            except Exception as e:
                print(f"Migration Error '{col_name}': {e}")
                
    conn.commit()
    conn.close()

# ---------------------- Processing Utilities ----------------------
def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif file_path.endswith(".docx"):
        import docx
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text.strip()

def clean_text_for_ai(text):
    return re.sub(r'\s+', ' ', text)

def fetch_page_content(url, headers):
    """Helper to safely fetch text content from a target webpage."""
    try:
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "iframe", "header"]):
                element.extract()
            text_content = soup.get_text(separator=' ')
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            if len(text_content) > 100:
                return text_content
    except Exception:
        pass
    return None

def auto_web_search(text_chunks):
    """
    Automated Web Plagiarism Search.
    Uses DDG HTML, DDG Lite, and Bing fallbacks.
    Returns matched results combining full page scrapes (via parallel ThreadPoolExecutor) 
    with snippet fallback references to guarantee plagiarism matches even when target 
    domains block python requests with Cloudflare or bot protection.
    """
    scraped_results = []
    
    # Enhanced browser headers to avoid detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    
    # Take top 3 longest sentences (more than 6 words) for search queries
    queries = [c for c in text_chunks if len(c.split()) > 6][:3]
    if not queries:
        return []
        
    visited_links = set()
    candidates = [] # List of dicts: {"raw_href": href, "snippet": search_snippet_text}
    
    for query in queries:
        try:
            # 1. Try DuckDuckGo HTML
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            res = requests.get(search_url, headers=headers, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                
                # Check for standard DDG result items
                results = soup.find_all('div', class_='result')
                for r in results:
                    a = r.find('a', class_='result__a')
                    snippet_a = r.find('a', class_='result__snippet')
                    if a:
                        href = a.get('href', '')
                        snippet = snippet_a.get_text().strip() if snippet_a else ""
                        if href:
                            candidates.append({"raw_href": href, "snippet": snippet})
                            
                # Fallback to plain links if custom classes not found
                if not candidates:
                    for a in soup.find_all('a'):
                        href = a.get('href', '')
                        if href and ('/l/?uddg=' in href or (href.startswith('http') and 'duckduckgo.com' not in href)):
                            candidates.append({"raw_href": href, "snippet": ""})
            
            # 2. Try DuckDuckGo Lite if HTML yielded no candidates
            if not candidates:
                lite_url = "https://lite.duckduckgo.com/lite/"
                res_lite = requests.post(lite_url, data={"q": query}, headers=headers, timeout=5)
                if res_lite.status_code == 200:
                    soup_lite = BeautifulSoup(res_lite.text, 'html.parser')
                    for table in soup_lite.find_all('table', class_='result-table'):
                        a = table.find('a', class_='result-link')
                        snippet_td = table.find('td', class_='result-snippet')
                        if a:
                            href = a.get('href', '')
                            snippet = snippet_td.get_text().strip() if snippet_td else ""
                            if href:
                                candidates.append({"raw_href": href, "snippet": snippet})
                                
            # 3. Try Bing Search as a third robust fallback
            if not candidates:
                bing_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}"
                res_bing = requests.get(bing_url, headers=headers, timeout=5)
                if res_bing.status_code == 200:
                    soup_bing = BeautifulSoup(res_bing.text, 'html.parser')
                    for li in soup_bing.find_all('li', class_='b_algo'):
                        a = li.find('a')
                        snippet_p = li.find('p')
                        if a:
                            href = a.get('href', '')
                            snippet = snippet_p.get_text().strip() if snippet_p else ""
                            if href and href.startswith('http') and 'bing.com' not in href and 'microsoft.com' not in href:
                                candidates.append({"raw_href": href, "snippet": snippet})
        except Exception as e:
            print(f"Web Search Query Error for '{query}': {e}")
            
    # Process and clean candidates
    valid_candidates = [] # list of {"url": cleaned_url, "snippet": snippet_text}
    for c in candidates:
        raw_href = c["raw_href"]
        target_url = raw_href
        
        # Decode DDG redirect format
        if "/l/?uddg=" in raw_href:
            try:
                target_url = raw_href.split("/l/?uddg=")[1].split("&")[0]
                target_url = requests.utils.unquote(target_url)
            except Exception:
                pass
                
        if target_url.startswith("//"):
            target_url = "https:" + target_url
            
        if not target_url.startswith("http") or "duckduckgo.com" in target_url:
            continue
            
        if target_url not in visited_links:
            visited_links.add(target_url)
            valid_candidates.append({"url": target_url, "snippet": c["snippet"]})
            
    # Concurrently fetch resolved targets (parallel execution)
    if valid_candidates:
        from concurrent.futures import ThreadPoolExecutor
        # Select top 4 distinct targets
        targets = valid_candidates[:4]
        
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = {executor.submit(fetch_page_content, t["url"], headers): t for t in targets}
            for future in futures:
                t = futures[future]
                url = t["url"]
                snippet = t["snippet"]
                try:
                    page_content = future.result()
                    # If page fetch was successful, use full page content
                    if page_content and len(page_content) > 100:
                        scraped_results.append({"text": page_content, "url": url})
                    # Else fall back to search engine's snippet! This guarantees dynamic plagiarism match works!
                    elif snippet:
                        scraped_results.append({"text": snippet, "url": url})
                except Exception:
                    # Fallback to snippet even on exception
                    if snippet:
                        scraped_results.append({"text": snippet, "url": url})
                        
    return scraped_results

# ---------------------- Core AI Engines ----------------------
def calculate_plagiarism_detailed(content, external_results, model_type="Semantic"):
    """Sentence-by-sentence plagiarism analysis."""
    if not external_results:
        return 0.0, [], []
        
    content_sentences = sent_tokenize(content)
    if not content_sentences:
        return 0.0, [], []
        
    ref_sentences = []
    ref_sources = []
    
    for item in external_results:
        url = item["url"]
        sentences = sent_tokenize(item["text"])
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean.split()) > 3:
                ref_sentences.append(s_clean)
                ref_sources.append(url)
                
    if not ref_sentences:
        return 0.0, [], []

    detailed_sentences = []
    plagiarized_count = 0
    source_match_counts = {}
    
    threshold = 0.75 if model_type == "Semantic" else 0.70
    
    if model_type == "Semantic":
        content_embeddings = semantic_model.encode(content_sentences, convert_to_tensor=True)
        reference_embeddings = semantic_model.encode(ref_sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(content_embeddings, reference_embeddings).cpu().numpy()
        
        for i, sentence in enumerate(content_sentences):
            best_match_idx = np.argmax(cosine_scores[i])
            best_score = float(cosine_scores[i][best_match_idx])
            
            is_plagiarized = best_score > threshold
            matched_url = ref_sources[best_match_idx] if is_plagiarized else None
            matched_text = ref_sentences[best_match_idx] if is_plagiarized else None
            
            if is_plagiarized:
                plagiarized_count += 1
                if matched_url not in source_match_counts:
                    source_match_counts[matched_url] = []
                source_match_counts[matched_url].append(best_score)
                
            detailed_sentences.append({
                "text": sentence,
                "plagiarism_score": round(best_score * 100, 2),
                "is_plagiarized": is_plagiarized,
                "matched_url": matched_url,
                "matched_text": matched_text
            })
    else:
        # Lexical (TF-IDF)
        vectorizer = TfidfVectorizer(stop_words='english')
        all_texts = content_sentences + ref_sentences
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            content_tfidf = tfidf_matrix[:len(content_sentences)]
            ref_tfidf = tfidf_matrix[len(content_sentences):]
            cosine_scores = cosine_similarity(content_tfidf, ref_tfidf)
            
            for i, sentence in enumerate(content_sentences):
                best_match_idx = np.argmax(cosine_scores[i])
                best_score = float(cosine_scores[i][best_match_idx])
                
                is_plagiarized = best_score > threshold
                matched_url = ref_sources[best_match_idx] if is_plagiarized else None
                matched_text = ref_sentences[best_match_idx] if is_plagiarized else None
                
                if is_plagiarized:
                    plagiarized_count += 1
                    if matched_url not in source_match_counts:
                        source_match_counts[matched_url] = []
                    source_match_counts[matched_url].append(best_score)
                    
                detailed_sentences.append({
                    "text": sentence,
                    "plagiarism_score": round(best_score * 100, 2),
                    "is_plagiarized": is_plagiarized,
                    "matched_url": matched_url,
                    "matched_text": matched_text
                })
        except Exception as e:
            print(f"TF-IDF failed: {e}")
            for sentence in content_sentences:
                detailed_sentences.append({
                    "text": sentence,
                    "plagiarism_score": 0.0,
                    "is_plagiarized": False,
                    "matched_url": None,
                    "matched_text": None
                })
                
    overall_score = round((plagiarized_count / len(content_sentences)) * 100, 2) if content_sentences else 0.0
    
    matched_sources = []
    for url, scores in source_match_counts.items():
        avg_score = float(np.mean(scores))
        matched_sources.append({
            "url": url,
            "match_count": len(scores),
            "avg_similarity": round(avg_score * 100, 2)
        })
    matched_sources = sorted(matched_sources, key=lambda x: x["avg_similarity"], reverse=True)
    
    return overall_score, detailed_sentences, matched_sources

def detect_ai_generation_detailed(text):
    """
    Enhanced AI detection using a weighted ensemble of 5 signals:
    1. Burstiness (sentence length standard deviation)
    2. Transition word patterns
    3. Vocabulary Richness (Type-Token Ratio)
    4. Sentence Opener Diversity
    5. Average Word Length uniformity
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return 0.0, [0.0] * len(sentences)
    
    words = text.lower().split()
    total_words = len(words)
    
    words_per_sentence = [len(s.split()) for s in sentences]
    avg_words = np.mean(words_per_sentence)
    std_words = np.std(words_per_sentence)
    
    # --- Signal 1: Burstiness (sentence length variance) ---
    burstiness = std_words
    if burstiness < 3.0:
        burstiness_score = 85.0 + (3.0 - burstiness) * 4
    elif burstiness < 6.0:
        burstiness_score = 50.0 + (6.0 - burstiness) * 11
    else:
        burstiness_score = max(5.0, 100.0 - (burstiness * 8))
    burstiness_score = min(burstiness_score, 99.0)
    
    # --- Signal 2: Transition word density ---
    ai_transition_patterns = re.compile(
        r'\b(moreover|furthermore|consequently|subsequently|indeed|in addition|'
        r'nonetheless|nevertheless|therefore|thus|hence|on the other hand|'
        r'in conclusion|to sum up|it is worth noting|it is important to note|'
        r'in this context|as a result|for instance|for example|additionally|'
        r'specifically|notably|significantly|ultimately|essentially)(?:\b|,)',
        re.IGNORECASE
    )
    transition_count = len(ai_transition_patterns.findall(text))
    transition_density = (transition_count / len(sentences)) * 100
    transition_score = min(transition_density * 18, 99.0)
    
    # --- Signal 3: Vocabulary Richness (Type-Token Ratio) ---
    if total_words > 0:
        unique_words = len(set(words))
        ttr = unique_words / total_words
        # AI text: TTR typically 0.40-0.55; Human: 0.55-0.80+
        if ttr < 0.45:
            vocab_score = 90.0
        elif ttr < 0.55:
            vocab_score = 70.0 + (0.55 - ttr) * 200
        elif ttr < 0.65:
            vocab_score = 40.0 + (0.65 - ttr) * 300
        else:
            vocab_score = max(5.0, 40.0 - (ttr - 0.65) * 200)
    else:
        vocab_score = 50.0
    vocab_score = min(max(vocab_score, 0.0), 99.0)
    
    # --- Signal 4: Sentence Opener Diversity ---
    openers = [s.strip().split()[0].lower() for s in sentences if s.strip()]
    if openers:
        unique_openers = len(set(openers))
        opener_diversity = unique_openers / len(openers)
        # Low diversity = AI-like (starts many sentences with "The", "It", "This")
        if opener_diversity < 0.4:
            opener_score = 85.0
        elif opener_diversity < 0.6:
            opener_score = 55.0 + (0.6 - opener_diversity) * 150
        else:
            opener_score = max(5.0, 55.0 - (opener_diversity - 0.6) * 125)
    else:
        opener_score = 50.0
    opener_score = min(max(opener_score, 0.0), 99.0)
    
    # --- Signal 5: Word Length Uniformity ---
    word_lengths = [len(w) for w in words if w.isalpha()]
    if word_lengths:
        avg_word_len = np.mean(word_lengths)
        std_word_len = np.std(word_lengths)
        # AI tends toward medium words (avg 4.5-5.5) with low variance
        len_uniformity = max(0, 100 - abs(avg_word_len - 5.0) * 20 - std_word_len * 10)
    else:
        len_uniformity = 50.0
    len_uniformity = min(max(len_uniformity, 0.0), 99.0)
    
    # --- Weighted Ensemble ---
    overall_score = (
        burstiness_score * 0.30 +
        transition_score * 0.20 +
        vocab_score * 0.20 +
        opener_score * 0.15 +
        len_uniformity * 0.15
    )
    overall_score = min(max(round(overall_score, 2), 0.0), 99.9)
    
    # --- Per-sentence scores ---
    sentence_scores = []
    for i, s in enumerate(sentences):
        w_count = words_per_sentence[i]
        deviation = abs(w_count - avg_words)
        length_uniformity_score = max(0, 100 - (deviation * 15))
        
        has_transitions = 1 if ai_transition_patterns.search(s) else 0
        transition_bonus = 20 if has_transitions else 0
        
        s_words = s.lower().split()
        s_unique = len(set(s_words))
        s_ttr = s_unique / max(len(s_words), 1)
        s_vocab_penalty = max(0, (0.6 - s_ttr) * 50) if s_ttr < 0.6 else 0
        
        s_ai_score = (length_uniformity_score * 0.5) + transition_bonus + (s_vocab_penalty * 0.3)
        s_ai_score = s_ai_score * (overall_score / 100.0)
        s_ai_score = min(max(s_ai_score, 0.0), 99.0)
        
        sentence_scores.append(round(s_ai_score, 2))
        
    return overall_score, sentence_scores

def extract_json_array_from_text(text):
    """Helper to locate and extract JSON array brackets from conversational LLM output."""
    text = text.strip()
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx+1]
    return text

def analyze_text_with_llm(content, external_results, api_provider, api_key):
    """Calls Groq or OpenRouter for sentence-by-sentence analysis."""
    content_sentences = sent_tokenize(content)
    if not content_sentences:
        return 0.0, 0.0, [], [], "Unknown"

    web_sources_str = ""
    for idx, item in enumerate(external_results):
        web_sources_str += f"[SOURCE {idx+1}] URL: {item['url']}\nContent Snippet: {item['text'][:1000]}\n\n"
    if not web_sources_str:
        web_sources_str = "No online search matches found.\n"

    doc_str = ""
    for idx, s in enumerate(content_sentences):
        doc_str += f"Sentence {idx+1}: {s}\n"

    system_prompt = (
        "You are an advanced academic integrity auditor. You evaluate documents for plagiarism and AI generation.\n"
        "Analyze the document sentence-by-sentence and return a JSON list of results.\n"
        "For each sentence, output exactly:\n"
        "- 'text': the exact sentence string.\n"
        "- 'plagiarism_score': 0-100 score indicating if it was paraphrased or copied from the Web Sources.\n"
        "- 'is_plagiarized': boolean (true if plagiarism_score > 70).\n"
        "- 'matched_url': the URL of the matching source (or null).\n"
        "- 'matched_text': the matching text snippet (or null).\n"
        "- 'ai_score': 0-100 probability of AI generation based on stylometrics.\n"
        "- 'explanation': brief explanation (max 15 words).\n\n"
        "Return ONLY a raw JSON array. Do NOT wrap it in ```json blocks."
    )

    user_prompt = f"Web Sources:\n{web_sources_str}\nDocument to Analyze:\n{doc_str}"

    headers = {"Content-Type": "application/json"}

    if api_provider == "Groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        llm_model = "llama-3.1-8b-instant"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = "http://127.0.0.1:5000"
        headers["X-Title"] = "ApexAudit AI"
        llm_model = "meta-llama/llama-3-8b-instruct:free"

    data = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }

    try:
        # Increased timeout to 40 seconds to prevent early drops for longer text scans
        res = requests.post(url, headers=headers, json=data, timeout=40)
        if res.status_code == 200:
            res_json = res.json()
            completion_text = res_json['choices'][0]['message']['content'].strip()
            
            # Robust JSON extraction to filter out introductory LLM responses
            completion_text = extract_json_array_from_text(completion_text)
            
            sentences_report = json.loads(completion_text)
            if isinstance(sentences_report, dict):
                for key, val in sentences_report.items():
                    if isinstance(val, list):
                        sentences_report = val
                        break
            
            if not isinstance(sentences_report, list):
                raise ValueError("LLM response did not format as a JSON array.")

            plagiarized_count = sum(1 for s in sentences_report if s.get("is_plagiarized", False) or s.get("plagiarism_score", 0) > 70)
            overall_plag = round((plagiarized_count / len(content_sentences)) * 100, 2)
            
            total_ai = sum(s.get("ai_score", 0) for s in sentences_report)
            overall_ai = round(total_ai / len(content_sentences), 2)

            source_match_counts = {}
            for s in sentences_report:
                m_url = s.get("matched_url")
                plag_s = s.get("plagiarism_score", 0)
                if m_url and plag_s > 70:
                    if m_url not in source_match_counts:
                        source_match_counts[m_url] = []
                    source_match_counts[m_url].append(plag_s)
                    
            matched_sources = []
            for url_key, scores in source_match_counts.items():
                matched_sources.append({
                    "url": url_key,
                    "match_count": len(scores),
                    "avg_similarity": round(sum(scores) / len(scores), 2)
                })
            matched_sources = sorted(matched_sources, key=lambda x: x["avg_similarity"], reverse=True)

            return overall_plag, overall_ai, sentences_report, matched_sources, f"{api_provider} LLM ({llm_model})"
        else:
            raise Exception(f"API returned status code {res.status_code}: {res.text}")
    except Exception as e:
        print(f"LLM API Error ({api_provider}): {e}. Falling back to local engines...")
        overall_plag, detailed_sentences, matched_sources = calculate_plagiarism_detailed(
            content, external_results, "Semantic"
        )
        overall_ai, ai_sentence_scores = detect_ai_generation_detailed(content)
        
        report_sentences = []
        for idx, sentence in enumerate(content_sentences):
            plag_info = detailed_sentences[idx] if idx < len(detailed_sentences) else {
                "plagiarism_score": 0.0, "is_plagiarized": False, "matched_url": None, "matched_text": None
            }
            report_sentences.append({
                "text": sentence,
                "plagiarism_score": plag_info["plagiarism_score"],
                "is_plagiarized": plag_info["is_plagiarized"],
                "matched_url": plag_info["matched_url"],
                "matched_text": plag_info["matched_text"],
                "ai_score": ai_sentence_scores[idx] if idx < len(ai_sentence_scores) else 0.0,
                "explanation": "Local S-BERT and burstiness evaluation fallback."
            })
            
        return overall_plag, overall_ai, report_sentences, matched_sources, "Local S-BERT / Hybrid"

# ---------------------- PDF Report Generator ----------------------
def generate_pdf_report(filename, plagiarism, ai_score, model_used, scan_report, created_at=None):
    """Generates a professional PDF report for a scan."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Title Header
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 10)
    pdf.cell(0, 10, "ApexAudit AI - Integrity Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.set_x(15)
    pdf.cell(0, 6, f"Generated: {created_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    pdf.ln(10)
    
    # Document Info
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Document Summary", ln=True)
    pdf.set_draw_color(59, 130, 246)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "", 10)
    info_items = [
        ("Filename", filename),
        ("Analysis Model", model_used),
        ("Plagiarism Score", f"{plagiarism}%"),
        ("AI Generation Score", f"{ai_score}%"),
    ]
    for label, value in info_items:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 7, f"{label}:", 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(value), ln=True)
    
    # Risk Assessment
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    plag_risk = "LOW RISK" if plagiarism < 15 else ("MODERATE" if plagiarism <= 40 else "HIGH RISK")
    ai_risk = "HUMAN" if ai_score < 30 else ("MIXED" if ai_score <= 70 else "AI GENERATED")
    pdf.cell(55, 7, "Plagiarism Risk:", 0, 0)
    pdf.set_font("Helvetica", "B", 10)
    if plagiarism < 15:
        pdf.set_text_color(16, 185, 129)
    elif plagiarism <= 40:
        pdf.set_text_color(245, 158, 11)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(0, 7, plag_risk, ln=True)
    
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(55, 7, "AI Detection:", 0, 0)
    if ai_score < 30:
        pdf.set_text_color(16, 185, 129)
    elif ai_score <= 70:
        pdf.set_text_color(245, 158, 11)
    else:
        pdf.set_text_color(168, 85, 247)
    pdf.cell(0, 7, ai_risk, ln=True)
    
    pdf.set_text_color(30, 30, 30)
    
    # Sentence Analysis Table
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Sentence-Level Analysis", ln=True)
    pdf.set_draw_color(59, 130, 246)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    sentences = scan_report.get("sentences", [])
    if sentences:
        # Table Header
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(8, 7, "#", 1, 0, 'C', True)
        pdf.cell(100, 7, "Sentence", 1, 0, 'C', True)
        pdf.cell(22, 7, "Plag %", 1, 0, 'C', True)
        pdf.cell(22, 7, "AI %", 1, 0, 'C', True)
        pdf.cell(38, 7, "Status", 1, 1, 'C', True)
        
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 7)
        
        for idx, s in enumerate(sentences[:50]):  # Cap at 50 rows
            text = s.get("text", "")[:80]
            plag_s = s.get("plagiarism_score", 0)
            ai_s = s.get("ai_score", 0)
            is_plag = s.get("is_plagiarized", False)
            is_ai = ai_s > 60
            
            status = "Clean"
            if is_plag and is_ai:
                status = "PLAG + AI"
                pdf.set_fill_color(254, 226, 226)
            elif is_plag:
                status = "PLAGIARIZED"
                pdf.set_fill_color(254, 243, 199)
            elif is_ai:
                status = "AI FLAGGED"
                pdf.set_fill_color(237, 233, 254)
            else:
                pdf.set_fill_color(255, 255, 255)
            
            pdf.cell(8, 6, str(idx+1), 1, 0, 'C', True)
            pdf.cell(100, 6, text, 1, 0, 'L', True)
            pdf.cell(22, 6, f"{plag_s}%", 1, 0, 'C', True)
            pdf.cell(22, 6, f"{ai_s}%", 1, 0, 'C', True)
            pdf.cell(38, 6, status, 1, 1, 'C', True)
    
    # Matched Sources
    matched_sources = scan_report.get("matched_sources", [])
    if matched_sources:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Matched Sources", ln=True)
        pdf.set_draw_color(59, 130, 246)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "", 9)
        for src in matched_sources:
            pdf.set_text_color(59, 130, 246)
            pdf.cell(0, 6, f"  {src['url'][:70]}", ln=True)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, f"    Match Count: {src['match_count']} sentences | Avg Similarity: {src['avg_similarity']}%", ln=True)
            pdf.ln(2)
    
    # Footer
    pdf.ln(8)
    pdf.set_text_color(148, 163, 184)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "This report was automatically generated by ApexAudit AI. Results are for academic guidance only.", ln=True, align='C')
    
    return pdf.output()

# ---------------------- App Routes ----------------------
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('upload_assignment'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            hashed_password = generate_password_hash(password)
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('upload_assignment'))
        else:
            flash("Invalid credentials!", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/scan-progress')
def scan_progress_sse():
    """SSE endpoint for real-time scan progress."""
    sid = session.get('user_id', 'anonymous')
    def generate():
        last_msg = ""
        for _ in range(200):  # Max ~60 seconds
            prog = get_progress(str(sid))
            msg = json.dumps(prog)
            if msg != last_msg:
                yield f"data: {msg}\n\n"
                last_msg = msg
            if prog.get("percent", 0) >= 100:
                break
            time.sleep(0.3)
        yield f"data: {json.dumps({'step': 99, 'message': 'Complete!', 'percent': 100})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['GET', 'POST'])
def upload_assignment():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        sid = str(session['user_id'])
        
        # Auto-resolve cloud LLM credentials from environment configurations
        api_provider = "Groq"
        resolved_key = os.getenv("GROQ_API_KEY")
        
        if not resolved_key:
            # Fallback to OpenRouter if Groq is not configured
            api_provider = "OpenRouter"
            resolved_key = os.getenv("OPENROUTER_API_KEY")
            
        # Fallback silently to local CPU matching if no cloud keys are found
        if not resolved_key:
            api_provider = "local"
            
        model_type = "Semantic"  # Default local fallback matching model
        enable_web_search = request.form.get('enable_web_search', 'on')  # Checked by default
        enable_ai_detect = "on"  # AI detection enabled by default
        
        set_progress(sid, 1, "Extracting document content...", 10)
        
        raw_content = ""
        filename = ""
        
        pasted_text = request.form.get('pasted_text', '').strip()
        
        if pasted_text:
            raw_content = pasted_text
            filename = f"PastedText_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        else:
            if 'file' not in request.files:
                flash("No file part", "danger")
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash("No selected file", "danger")
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                raw_content = extract_text(file_path)

        if not raw_content:
            flash("No text could be extracted.", "danger")
            return redirect(request.url)

        clean_content = clean_text_for_ai(raw_content)
        chunks = sent_tokenize(clean_content)
        
        if not chunks:
            flash("Text contains no valid sentences.", "danger")
            return redirect(request.url)

        set_progress(sid, 2, "Tokenized. Initiating web search...", 25)

        # 1. Live Web Search
        external_reference_texts = []
        if enable_web_search == 'on':
            external_reference_texts = auto_web_search(chunks)

        set_progress(sid, 3, "Web search complete. Running analysis engine...", 50)

        # Dispatch matching engine
        if api_provider in ['Groq', 'OpenRouter']:
            set_progress(sid, 4, f"Contacting {api_provider} cloud LLM...", 65)
            plagiarism_score, ai_score, report_sentences, matched_sources, model_used = analyze_text_with_llm(
                clean_content, external_reference_texts, api_provider, resolved_key
            )
        else:
            model_used = model_type
            set_progress(sid, 4, "Running embedding comparisons...", 65)
            plagiarism_score, detailed_sentences, matched_sources = calculate_plagiarism_detailed(
                clean_content, external_reference_texts, model_type
            )

            set_progress(sid, 5, "Computing AI detection metrics...", 80)
            ai_score = 0.0
            ai_sentence_scores = [0.0] * len(chunks)
            if enable_ai_detect == 'on':
                ai_score, ai_sentence_scores = detect_ai_generation_detailed(clean_content)

            report_sentences = []
            for idx, sentence in enumerate(chunks):
                plag_info = {
                    "plagiarism_score": 0.0, "is_plagiarized": False,
                    "matched_url": None, "matched_text": None
                }
                if idx < len(detailed_sentences):
                    plag_info = {
                        "plagiarism_score": detailed_sentences[idx]["plagiarism_score"],
                        "is_plagiarized": detailed_sentences[idx]["is_plagiarized"],
                        "matched_url": detailed_sentences[idx]["matched_url"],
                        "matched_text": detailed_sentences[idx]["matched_text"]
                    }
                    
                report_sentences.append({
                    "text": sentence,
                    "plagiarism_score": plag_info["plagiarism_score"],
                    "is_plagiarized": plag_info["is_plagiarized"],
                    "matched_url": plag_info["matched_url"],
                    "matched_text": plag_info["matched_text"],
                    "ai_score": ai_sentence_scores[idx] if idx < len(ai_sentence_scores) else 0.0
                })

        set_progress(sid, 6, "Saving report to database...", 92)

        scan_report = {
            "model_used": model_used,
            "sentences": report_sentences,
            "matched_sources": matched_sources
        }

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""INSERT INTO assignments 
                       (user_id, filename, plagiarism, ai_generated_score, content, scan_report, model_used) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (session['user_id'], filename, plagiarism_score, ai_score, clean_content, json.dumps(scan_report), model_used))
        conn.commit()
        new_id = cur.lastrowid
        conn.close()

        set_progress(sid, 7, "Complete!", 100)

        flash("Diagnostic check complete!", "success")
        return redirect(url_for('view_result', assignment_id=new_id))

    return render_template('upload.html')

@app.route('/assignments')
def view_assignments():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, plagiarism, ai_generated_score, model_used, created_at FROM assignments WHERE user_id=? ORDER BY created_at DESC", (session['user_id'],))
    rows = cur.fetchall()
    conn.close()
    return render_template('assignments.html', rows=rows)

@app.route('/result/<int:assignment_id>')
def view_result(assignment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT filename, plagiarism, ai_generated_score, content, scan_report, model_used FROM assignments WHERE id=? AND user_id=?", (assignment_id, session['user_id']))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        flash("Report not found!", "danger")
        return redirect(url_for('view_assignments'))
        
    filename, plagiarism, ai_score, content, scan_report_json, model_used = row
    
    # Calculate stylometric metrics for 5-Signal Ensemble display
    try:
        sentences = sent_tokenize(content)
        words = re.findall(r'\b\w+\b', content.lower())
        total_words = len(words)
        
        # 1. Burstiness (Std Dev of sentence lengths)
        words_per_sentence = [len(s.split()) for s in sentences]
        burstiness = round(float(np.std(words_per_sentence)), 1) if sentences else 0.0
        
        # 2. Transition Density (Rate of typical AI transitions)
        ai_transition_patterns = re.compile(
            r'\b(moreover|furthermore|consequently|subsequently|indeed|in addition|'
            r'nonetheless|nevertheless|therefore|thus|hence|on the other hand|'
            r'in conclusion|to sum up|it is worth noting|it is important to note|'
            r'in this context|as a result|for instance|for example|additionally|'
            r'specifically|notably|significantly|ultimately|essentially)(?:\b|,)',
            re.IGNORECASE
        )
        transition_count = len(ai_transition_patterns.findall(content))
        transition_density = round((transition_count / max(len(sentences), 1)) * 10, 1)
        
        # 3. Vocabulary Richness (Type-Token Ratio)
        ttr = round((len(set(words)) / max(total_words, 1)) * 100, 1) if total_words else 0.0
        
        # 4. Opener Diversity
        openers = [s.strip().split()[0].lower() for s in sentences if s.strip()]
        opener_diversity = round((len(set(openers)) / max(len(openers), 1)) * 100, 1) if openers else 0.0
        
        # 5. Word Length Uniformity
        word_lengths = [len(w) for w in words if w.isalpha()]
        avg_word_len = round(float(np.mean(word_lengths)), 1) if word_lengths else 0.0
    except Exception:
        burstiness = 0.0
        transition_density = 0.0
        ttr = 0.0
        opener_diversity = 0.0
        avg_word_len = 0.0
        
    try:
        scan_report = json.loads(scan_report_json)
    except Exception:
        scan_report = {"model_used": model_used, "sentences": [], "matched_sources": []}
        
    return render_template('result.html', 
                           filename=filename, 
                           plagiarism_percentage=plagiarism, 
                           ai_score=ai_score, 
                           content=content, 
                           scan_report=scan_report, 
                           model_used=model_used,
                           assignment_id=assignment_id,
                           burstiness=burstiness,
                           transition_density=transition_density,
                           ttr=ttr,
                           opener_diversity=opener_diversity,
                           avg_word_len=avg_word_len)

@app.route('/export/<int:assignment_id>')
def export_pdf(assignment_id):
    """Export scan result as a professional PDF report."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT filename, plagiarism, ai_generated_score, content, scan_report, model_used, created_at FROM assignments WHERE id=? AND user_id=?", (assignment_id, session['user_id']))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        flash("Report not found!", "danger")
        return redirect(url_for('view_assignments'))
    
    filename, plagiarism, ai_score, content, scan_report_json, model_used, created_at = row
    
    try:
        scan_report = json.loads(scan_report_json)
    except Exception:
        scan_report = {"sentences": [], "matched_sources": []}
    
    pdf_bytes = generate_pdf_report(filename, plagiarism, ai_score, model_used, scan_report, created_at)
    
    pdf_filename = f"ApexAudit_Report_{filename.rsplit('.', 1)[0]}.pdf"
    
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=pdf_filename
    )

@app.route('/compare/<int:assignment_id>')
def compare_view(assignment_id):
    """Side-by-side comparison of student text vs matched sources."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT filename, plagiarism, content, scan_report, model_used FROM assignments WHERE id=? AND user_id=?", (assignment_id, session['user_id']))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        flash("Report not found!", "danger")
        return redirect(url_for('view_assignments'))
    
    filename, plagiarism, content, scan_report_json, model_used = row
    
    try:
        scan_report = json.loads(scan_report_json)
    except Exception:
        scan_report = {"sentences": [], "matched_sources": []}
    
    # Build comparison pairs
    comparison_pairs = []
    for s in scan_report.get("sentences", []):
        if s.get("is_plagiarized") and s.get("matched_text"):
            comparison_pairs.append({
                "student_text": s["text"],
                "source_text": s["matched_text"],
                "source_url": s.get("matched_url", "Unknown"),
                "similarity": s.get("plagiarism_score", 0)
            })
    
    return render_template('compare.html',
                           filename=filename,
                           plagiarism=plagiarism,
                           comparison_pairs=comparison_pairs,
                           assignment_id=assignment_id)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/delete/<int:assignment_id>', methods=['POST'])
def delete_assignment(assignment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM assignments WHERE id=? AND user_id=?", (assignment_id, session['user_id']))
    conn.commit()
    conn.close()
    flash("Diagnostic report removed.", "success")
    return redirect(url_for('view_assignments'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)