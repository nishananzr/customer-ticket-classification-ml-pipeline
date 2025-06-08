import gradio as gr
import joblib
import pandas as pd
import re
import nltk
from textblob import TextBlob
from scipy.sparse import hstack
import json

# --- 1. NLTK Resource Downloads ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# --- 2. Load Models and Artifacts ---
issue_classifier = joblib.load('You are absolutely right. My apologies. Let's stick with the `gr.Interface` template you like and fix the underlying issue. The problem isn't the template itself, but how the data is being passed and returned.

The core issue is that when an error happens inside your Python function, `gr.Interface` doesn't know how to handle it gracefully and just shows "Error" in the output fields. We need to make your main function more robust.

The most likely cause is an unexpected input to one of the scikit-learn or NLTK functions, especially if the input text is very short or empty after preprocessing.

### The Corrected Code (Using `gr.Interface`)

This version keeps the `gr.Interface` structure but adds error handling (`try...except`) inside your main processing function. This way, if anything goes wrong, it will return a helpful message instead of crashing.

**models/issue_classifier.joblib')
urgency_classifier = joblib.load('models/urgency_classifier.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')
product_list = joblib.load('models/product_list.joblib')

# --- 3. Define Helper Functions ---
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower().strip()
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    Replace the entire content of your `app.py` file with this corrected code:**

```python
import gradio as gr
import joblib
import pandas as pd
import re
import nltk
from textblob import TextBlob
from scipy.sparse import hstack
import json

# --- NLTK Resource Downloads ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    return " ".join(processed_tokens)

def extract_entities_advanced(text):
    extracted_entities = {"products": [], "dates": [], "complaint_phrases": []}
    grammar = r"NP: {<JJ.*>*<NN.*>+}"
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    chunk_parser = nltk.Regnltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# --- Load Models and Artifacts ---
issue_classifier = joblib.load('models/issue_classifier.joblib')
urgency_classifier = joblib.load('models/urgency_classifier.jobexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    found_products = set()
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        potential_product = " ".join(word for word, tag in subtree.leaves())
        for known_product in product_list:
            if known_product.lower() in potential_product.lower():
                found_products.add(known_product)
    extracted_entities["products"] = list(found_products)
    date_pattern = r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|(?:Jan|Feblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')
product_list = joblib.load('models/product_list.joblib')

# --- Initialize NLTK Components ---
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# --- Preprocessing and Extraction Functions ---
def preprocess_text(text):
    text = reYou are right. Let's stick with the `gr.Interface` template you|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2}(?:st|nd|rd|th)?(?:,?\s\d{4})?|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:,?\s\d{4})?)\b'
    extracted_entities["dates"] = re.findall(date_pattern,.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower().strip()
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len( prefer and fix the underlying issue. The problem is very likely in how the data is being returned from your function to match what the `gr.JSON` output component expects.

The `gr.JSON` component expects a Python dictionary or a list that can be serialized into a JSON object. We will ensure the entity extraction output is formatted perfectly for it.

### The text, re.IGNORECASE)
    complaint_grammar = r"VP: {<VB.*>+<RB|RP|JJ|VBN|VBG>*}"
    complaint_parser = nltk.RegexpParser(complaint_grammar)
    complaint_tree = complaint_parser.parse(pos_tags)
    complaint_phrases = set()
    root_complaint_words = ['broken',word) > 2]
    return " ".join(processed_tokens)

def extract_entities_advanced(text):
    extracted_entities = {"products": [], "dates": [], "complaint_phrases": []}
    grammar = r"NP: {<JJ.*>*<NN.*>+}"
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
 Correction

The main change is in the `process_and_predict` function. I will ensure it returns a clean Python dictionary for the entities, which the `gr.JSON` output can handle without any issues.

**Here is the complete, corrected code for your `app.py` file using the `gr.Interface` template.**

```python
import gradio as gr
import joblib
import pandas as pd
import re
import nltk
from textblob import TextBlob
 'defective', 'faulty', 'working', 'fail', 'error', 'crash', 'slow', 'late', 'delayed', 'missing', 'lost', 'unresponsive']
    for subtree in complaint_tree.subtrees(filter=lambda t: t.label() == 'VP'):
        phrase_words = [word for word, tag in subtree.leaves()]
        if any(root_word in phrase_words for root_word in root_complaint_words):
            complaint_phrases.add(" ".join(phrase_words))
    extracted_entities["complaint_phrases"] = list(complaint_phrases)
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    found_products = set()
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        potential_product = " ".join(word for word, tag in subtree.leaves())
        for known_product in product_list:
            if known_product.lower() in potential_product.lower():
                found_products.add(known_product)
    extracted_entities["products"] = list(found_products)
    date_pattern = r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2}(?:st|ndfrom scipy.sparse import hstack
import json

# --- NLTK Resource Downloads ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError    return extracted_entities

# --- 4. Main Prediction Function (Corrected for gr.Interface) ---
def process_and_predict(raw_text):
    # Handle empty or whitespace-only input
    if not raw_text or not raw_text.strip():
        return "N/A", "N/A", {}

    # ML Prediction part
    processed_text_for_model = preprocess_text(raw_text)
    text_tfidf = tfidf_vectorizer.transform([processed_text_for_model])
    ticket_len = len(raw_text.split())
    sentiment = TextBlob(raw_text).sentiment.polarity
    additional_features = scaler.transform([[ticket_len, sentiment]])
    combined_features = hstack([text|rd|th)?(?:,?\s\d{4})?|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:,?\s\d{4})?)\b'
    extracted_entities["dates"] = re.findall(date_pattern, text, re.IGNORECASE)
    complaint_grammar = r"VP: {<VB.*>+<RB|RP|JJ|VBN|VBG>*}:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# --- Load Models and Artifacts ---
issue_classifier = joblib.load('models/issue_classifier.joblib')
urgency_classifier = joblib.load('models/urgency_classifier.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')
product_list = joblib.load('models/product_list.joblib')

# --- Initialize NLTK Components ---
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# --- Helper Functions (from notebook) ---
def preprocess_text(text):
    _tfidf, additional_features])
    
    predicted_issue = issue_classifier.predict(combined_features)[0]
    predicted_urgency = urgency_classifier.predict(combined_features)[0]
    
    # Entity Extraction part
    entities = extract_entities_advanced(raw_text)
    
    # Return a tuple of (string, string, dictionary) to match the outputs
    return predicted_issue, predicted_urgency, entities

# ---"
    complaint_parser = nltk.RegexpParser(complaint_grammar)
    complaint_tree = complaint_parser.parse(pos_tags)
    complaint_phrases = set()
    root_complaint_words = ['broken', 'defective', 'faulty', 'working', 'fail', 'error', 'crash', 'slow', 'late', 'delayed', 'missing', 'lost', 'unresponsive']
    for subtree in complaint_tree.subtrees(filter=lambda t: t.label() == 'VP'):
        phrase_words = [word for word, tag in subtree.leaves()]
        if any(root_word in phrase_words for root_word in root_complaint_words):
            complaint_phrases.add(" ".join(phrase_words))
    extracted_entities["complaint_phrases"] = list(complaint_phrases)
    return extracted_entities

# --- Main Prediction Function (Made More Robust) ---
text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower().strip()
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(processed_tokens)

def extract_entities_advanced(text):
    extracted_entities = {"products": [], "dates 5. Gradio Interface Definition ---
if __name__ == "__main__":
    app_interface = gr.Interface(
        fn=process_and_predict,
        inputs=gr.Textbox(lines=8, placeholder="Please enter the customer ticket text here..."),
        outputs=[
            gr.Label(label="Predicted Issue Type"),
            gr.Label(label="Predicted Urgency Level"),
            gr.JSON(label="Extracted Entities")
        ],
        title="Customer Support Ticket Analyzer",
        description="Enter a customer support ticket text to automatically classify its issue type, urgency level, and extract key entities",
        allow_flagging="never",
        examples=[
            ["My FitRun Treadmill is broken and the screen won't turn on. I ordered it on May 1st, 2024."],
            ["I was double-charged for my PowerMaxdef process_and_predict(raw_text):
    # Check for empty or whitespace-only input
    if not raw_text or not raw_text.strip():
        return "No prediction", "No prediction", {"message": "Please enter some text."}

    try:
        # ML Prediction part
        processed_text_for_model = preprocess_text(raw_text)
        
        # Another safety check: if preprocessing results in empty text
": [], "complaint_phrases": []}
    grammar = r"NP: {<JJ.*>*<NN.*>+}"
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    found_products = set()
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        potential_product = " ".join(word for word, tag in subtree.leaves())
        for known_product in product_list:
            if known_product.lower() in potential_product.lower():
                found_products.add(known_product)
    extracted_entities["products"] = list(found_products)
    date_pattern = r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|(?:Jan|Feb|Mar|Apr Battery order #8876. Please issue a refund immediately."],
            ["I have a question about the warranty for the SoundWave 300."]
        ]
    )

    app_interface.launch()