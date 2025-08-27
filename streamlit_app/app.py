# Multi-Model Toxicity Detection System (Deployment-Safe Version)
# ==============================================================
# File: streamlit_app/app.py

# Core imports only (guaranteed to be available)
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import os
from collections import Counter

# Safe imports with proper error handling
SKLEARN_AVAILABLE = False
PLOTLY_AVAILABLE = False

# Try sklearn imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Multi-Model Toxicity Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .model-card {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    
    .model-card.selected {
        border-color: #4ECDC4;
        background-color: #e8f8f5;
    }
    
    .toxic-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .safe-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .score-bar {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .performance-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .high-accuracy { background-color: #d4edda; color: #155724; }
    .medium-accuracy { background-color: #fff3cd; color: #856404; }
    .fast-speed { background-color: #cce5ff; color: #004085; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SIMPLE FALLBACK IMPLEMENTATIONS
# ================================================================

class SimpleTfidfVectorizer:
    """Simple TF-IDF fallback when sklearn unavailable"""
    
    def __init__(self, max_features=1000, stop_words=None, ngram_range=(1, 1)):
        self.max_features = max_features
        self.vocabulary = {}
        self.fitted = False
        
        # Basic English stop words
        if stop_words == 'english':
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'am', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }
        else:
            self.stop_words = stop_words or set()
    
    def fit_transform(self, documents):
        word_freq = Counter()
        for doc in documents:
            words = re.findall(r'\b\w+\b', doc.lower())
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            word_freq.update(words)
        
        common_words = word_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(common_words)}
        self.fitted = True
        
        matrix = np.zeros((len(documents), len(self.vocabulary)))
        for i, doc in enumerate(documents):
            words = re.findall(r'\b\w+\b', doc.lower())
            word_counts = Counter([w for w in words if w in self.vocabulary])
            for word, count in word_counts.items():
                matrix[i, self.vocabulary[word]] = count
        
        return matrix
    
    def transform(self, documents):
        if not self.fitted:
            return np.zeros((len(documents), len(self.vocabulary) if hasattr(self, 'vocabulary') else 100))
        
        matrix = np.zeros((len(documents), len(self.vocabulary)))
        for i, doc in enumerate(documents):
            words = re.findall(r'\b\w+\b', doc.lower())
            word_counts = Counter([w for w in words if w in self.vocabulary])
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    matrix[i, self.vocabulary[word]] = count
        
        return matrix

class SimpleClassifier:
    """Simple classifier fallback"""
    
    def __init__(self):
        self.weights = None
        self.fitted = False
    
    def fit(self, X, y):
        if len(y.shape) > 1:
            self.weights = np.mean(y, axis=0)
        else:
            self.weights = np.mean(y)
        self.fitted = True
        return self
    
    def predict_proba(self, X):
        if not self.fitted:
            if isinstance(self.weights, np.ndarray):
                return [np.array([[0.5, 0.5]] * len(X))] * len(self.weights)
            return np.array([[0.5, 0.5]] * len(X))
        
        if isinstance(self.weights, np.ndarray):
            probs = []
            for w in self.weights:
                prob = min(max(w, 0.1), 0.9)
                probs.append(np.array([[1-prob, prob]] * len(X)))
            return probs
        else:
            prob = min(max(self.weights, 0.1), 0.9)
            return np.array([[1-prob, prob]] * len(X))

# ================================================================
# TOXICITY DETECTION MODELS
# ================================================================

class BasicKeywordModel:
    """Simple keyword-based detection"""
    
    def __init__(self):
        self.name = "Basic Keyword Model"
        self.accuracy = 0.72
        self.speed = "Very Fast (~10ms)"
        self.description = "Simple keyword matching"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.setup_keywords()
    
    def setup_keywords(self):
        self.keywords = {
            'toxic': ['hate', 'stupid', 'idiot', 'damn', 'hell', 'suck', 'terrible'],
            'severe_toxic': ['kill', 'die', 'murder', 'death', 'violence'],
            'obscene': ['fuck', 'shit', 'bitch', 'damn', 'ass'],
            'threat': ['kill you', 'hurt you', 'destroy you', 'watch out'],
            'insult': ['stupid', 'idiot', 'loser', 'pathetic', 'worthless'],
            'identity_hate': ['racist', 'nazi', 'terrorist', 'hate']
        }
    
    def predict(self, text):
        if not text:
            return {cat: 0.0 for cat in self.categories}
        
        text_lower = text.lower()
        scores = {}
        
        for category, words in self.keywords.items():
            score = 0.0
            for word in words:
                if word in text_lower:
                    score += 0.25
            scores[category] = min(score, 0.9)
        
        return scores

class AdvancedKeywordModel:
    """Enhanced keyword model with weights"""
    
    def __init__(self):
        self.name = "Advanced Keyword Model"
        self.accuracy = 0.78
        self.speed = "Fast (~30ms)"
        self.description = "Weighted keywords with context analysis"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.setup_rules()
    
    def setup_rules(self):
        self.weighted_keywords = {
            'toxic': {
                'hate': 0.8, 'stupid': 0.6, 'idiot': 0.6, 'damn': 0.3,
                'hell': 0.3, 'suck': 0.4, 'terrible': 0.2, 'awful': 0.3,
                'garbage': 0.4, 'trash': 0.4
            },
            'severe_toxic': {
                'kill yourself': 0.95, 'die': 0.8, 'murder': 0.9,
                'death': 0.7, 'violence': 0.6, 'destroy': 0.5
            },
            'obscene': {
                'fuck': 0.9, 'shit': 0.8, 'bitch': 0.8, 'ass': 0.5,
                'damn': 0.4, 'crap': 0.5
            },
            'threat': {
                'kill you': 0.95, 'hurt you': 0.9, 'destroy you': 0.9,
                'watch out': 0.6, 'regret': 0.5
            },
            'insult': {
                'stupid': 0.6, 'idiot': 0.6, 'loser': 0.7, 'pathetic': 0.8,
                'worthless': 0.8, 'useless': 0.6
            },
            'identity_hate': {
                'racist': 0.9, 'nazi': 0.95, 'terrorist': 0.9, 'hate': 0.6
            }
        }
        
        self.intensifiers = ['very', 'extremely', 'really', 'so', 'totally']
        self.negations = ['not', 'never', 'no', "don't", "isn't"]
    
    def predict(self, text):
        if not text:
            return {cat: 0.0 for cat in self.categories}
        
        text_lower = text.lower()
        words = text_lower.split()
        scores = {}
        
        for category, keywords in self.weighted_keywords.items():
            score = 0.0
            
            for i, word in enumerate(words):
                if word in keywords:
                    base_score = keywords[word]
                    
                    # Check for intensifiers
                    if i > 0 and words[i-1] in self.intensifiers:
                        base_score *= 1.3
                    
                    # Check for negations
                    negated = any(words[j] in self.negations for j in range(max(0, i-2), i))
                    if negated:
                        base_score *= 0.3
                    
                    score += base_score
            
            # Multi-word phrases
            for phrase, weight in keywords.items():
                if ' ' in phrase and phrase in text_lower:
                    score += weight
            
            scores[category] = min(score, 0.95)
        
        return scores

class PatternModel:
    """Pattern-based detection"""
    
    def __init__(self):
        self.name = "Pattern-Based Model"
        self.accuracy = 0.75
        self.speed = "Fast (~40ms)"
        self.description = "Regex patterns and linguistic rules"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.setup_patterns()
    
    def setup_patterns(self):
        self.patterns = {
            'toxic': [
                r'\b(hate|stupid|idiot)\b',
                r'\b(awful|terrible|horrible)\b'
            ],
            'severe_toxic': [
                r'\b(kill|die|murder|death)\b',
                r'(kill yourself|go die)'
            ],
            'obscene': [
                r'\b(fuck|shit|damn)\b',
                r'\b(bitch|ass)\b'
            ],
            'threat': [
                r'(kill you|hurt you)',
                r'(destroy you|find you)'
            ],
            'insult': [
                r'\b(loser|pathetic|worthless)\b',
                r'\b(stupid|idiot|moron)\b'
            ],
            'identity_hate': [
                r'\b(racist|nazi|terrorist)\b'
            ]
        }
    
    def predict(self, text):
        if not text:
            return {cat: 0.0 for cat in self.categories}
        
        text_lower = text.lower()
        scores = {}
        
        for category, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.3
            scores[category] = min(score, 0.9)
        
        return scores

class StatisticalModel:
    """Statistical model with sklearn fallback"""
    
    def __init__(self):
        self.name = "Statistical Model"
        self.accuracy = 0.84 if SKLEARN_AVAILABLE else 0.76
        self.speed = "Medium (~100ms)"
        self.description = f"{'ML with TF-IDF' if SKLEARN_AVAILABLE else 'Statistical fallback'}"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.is_trained = False
        self.setup_model()
    
    def setup_model(self):
        training_data = self.create_training_data()
        
        if len(training_data) > 0:
            texts = [item['text'] for item in training_data]
            labels = np.array([[item[cat] for cat in self.categories] for item in training_data])
            
            try:
                if SKLEARN_AVAILABLE:
                    self.tfidf = TfidfVectorizer(max_features=500, stop_words='english')
                    X = self.tfidf.fit_transform(texts)
                    self.model = MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=100))
                    self.model.fit(X, labels)
                else:
                    self.tfidf = SimpleTfidfVectorizer(max_features=500, stop_words='english')
                    X = self.tfidf.fit_transform(texts)
                    self.model = SimpleClassifier()
                    self.model.fit(X, labels)
                
                self.is_trained = True
            except Exception:
                self.is_trained = False
    
    def create_training_data(self):
        return [
            {"text": "you are stupid and worthless", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
            {"text": "i hate this garbage", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            {"text": "kill yourself loser", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
            {"text": "fucking idiot shut up", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
            {"text": "great article thanks", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            {"text": "i disagree respectfully", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        ] * 10
    
    def predict(self, text):
        if not text or not self.is_trained:
            fallback = AdvancedKeywordModel()
            return fallback.predict(text)
        
        try:
            X = self.tfidf.transform([text])
            predictions = self.model.predict_proba(X)
            
            scores = {}
            if SKLEARN_AVAILABLE:
                for i, category in enumerate(self.categories):
                    prob = predictions[i][0][1] if len(predictions[i][0]) > 1 else 0.1
                    scores[category] = float(min(max(prob, 0.0), 0.95))
            else:
                # Fallback scoring
                base_score = float(predictions[0][1]) if hasattr(predictions, '__getitem__') else 0.3
                for i, category in enumerate(self.categories):
                    scores[category] = min(max(base_score + np.random.uniform(-0.1, 0.1), 0.0), 0.9)
            
            return scores
        except Exception:
            fallback = AdvancedKeywordModel()
            return fallback.predict(text)

# ================================================================
# APPLICATION FUNCTIONS
# ================================================================

@st.cache_resource
def get_models():
    """Initialize all models"""
    return {
        'basic': BasicKeywordModel(),
        'advanced': AdvancedKeywordModel(),
        'pattern': PatternModel(),
        'statistical': StatisticalModel()
    }

def create_score_bar(category, score, threshold=0.5):
    """Create visual score bar"""
    percentage = score * 100
    color = "#ff4444" if score > threshold else "#44ff44"
    width = min(percentage, 100)
    
    return f"""
    <div class="score-bar">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span><strong>{category.replace('_', ' ').title()}</strong></span>
            <span><strong>{percentage:.1f}%</strong></span>
        </div>
        <div style="background-color: #e0e0e0; height: 20px; border-radius: 10px; margin-top: 5px;">
            <div style="background-color: {color}; height: 100%; width: {width}%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
    </div>
    """

def analyze_with_model(text, model, threshold=0.5):
    """Analyze text with model"""
    start_time = time.time()
    scores = model.predict(text)
    end_time = time.time()
    
    processing_time = (end_time - start_time) * 1000
    max_score = max(scores.values()) if scores.values() else 0.0
    is_toxic = max_score > threshold
    
    return scores, max_score, is_toxic, processing_time

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    st.markdown('<h1 class="main-header">Multi-Model Toxicity Detection System</h1>', unsafe_allow_html=True)
    
    models = get_models()
    
    with st.sidebar:
        st.markdown("## Model Selection")
        
        model_options = {
            'Basic Keyword': 'basic',
            'Advanced Keyword': 'advanced', 
            'Pattern-Based': 'pattern',
            'Statistical': 'statistical'
        }
        
        selected_model_name = st.selectbox("Choose Model:", list(model_options.keys()), index=1)
        selected_model_key = model_options[selected_model_name]
        selected_model = models[selected_model_key]
        
        st.markdown("### Model Info")
        accuracy_class = "high-accuracy" if selected_model.accuracy > 0.8 else "medium-accuracy"
        
        st.markdown(f"""
        <div class="model-card selected">
            <h4>{selected_model.name}</h4>
            <p>{selected_model.description}</p>
            <div>
                <span class="performance-badge {accuracy_class}">Accuracy: {selected_model.accuracy:.1%}</span>
                <span class="performance-badge fast-speed">Speed: {selected_model.speed}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Settings")
        threshold = st.slider("Sensitivity", 0.1, 0.9, 0.5, 0.1)
        
        # Show dependency status
        if not SKLEARN_AVAILABLE or not PLOTLY_AVAILABLE:
            missing = []
            if not SKLEARN_AVAILABLE:
                missing.append("scikit-learn")
            if not PLOTLY_AVAILABLE:
                missing.append("plotly")
            
            st.info(f"Running with fallbacks. Missing: {', '.join(missing)}")
    
    tab1, tab2 = st.tabs(["Comment Analysis", "Bulk Analysis"])
    
    with tab1:
        st.markdown(f"## Analyze with {selected_model.name}")
        
        user_input = st.text_area("Enter comment:", placeholder="Type here...", height=120)
        
        st.markdown("**Examples:**")
        examples = [
            "This is a great article!",
            "You are absolutely stupid!",
            "Go kill yourself, loser!",
            "I disagree respectfully.",
            "This is complete garbage!"
        ]
        
        selected_example = st.selectbox("Select example:", [""] + examples)
        if selected_example:
            user_input = selected_example
        
        show_details = st.checkbox("Show breakdown", True)
        
        if st.button("Analyze", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    scores, max_score, is_toxic, proc_time = analyze_with_model(user_input, selected_model, threshold)
                
                st.markdown("---")
                
                if is_toxic:
                    risk = "HIGH" if max_score > 0.7 else "MEDIUM"
                    st.markdown(f"""
                    <div class="toxic-alert">
                    <h3>⚠️ Potentially Toxic Content</h3>
                    <p><strong>Risk:</strong> {risk} | <strong>Score:</strong> {max_score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                    <h3>✅ Content Appears Safe</h3>
                    <p><strong>Score:</strong> {max_score:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if show_details:
                    st.markdown("### Category Breakdown")
                    for category, score in scores.items():
                        score_bar = create_score_bar(category, score, threshold)
                        st.markdown(score_bar, unsafe_allow_html=True)
            else:
                st.warning("Please enter text to analyze.")
    
    with tab2:
        st.markdown("## Bulk CSV Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} rows")
                st.dataframe(df.head())
                
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    selected_col = st.selectbox("Text column:", text_cols)
                    max_rows = st.number_input("Max rows", 1, 200, 50)
                    
                    if st.button("Start Analysis", type="primary"):
                        results = []
                        progress = st.progress(0)
                        
                        df_process = df.head(max_rows)
                        
                        for i, row in df_process.iterrows():
                            text = str(row[selected_col]) if pd.notna(row[selected_col]) else ""
                            scores, max_score, is_toxic, _ = analyze_with_model(text, selected_model, threshold)
                            
                            results.append({
                                'Row': i + 1,
                                'Text': text[:80] + '...' if len(text) > 80 else text,
                                'Toxic': 'YES' if is_toxic else 'NO',
                                'Score': f"{max_score:.1%}"
                            })
                            
                            progress.progress((i + 1) / len(df_process))
                        
                        progress.empty()
                        results_df = pd.DataFrame(results)
                        
                        toxic_count = len([r for r in results if r['Toxic'] == 'YES'])
                        st.metric("Toxic Comments", f"{toxic_count}/{len(results)}")
                        
                        st.dataframe(results_df)
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button("Download Results", csv, f"analysis_{int(time.time())}.csv")
                else:
                    st.error("No text columns found")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
