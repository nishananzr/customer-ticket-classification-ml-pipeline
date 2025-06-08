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
issue_classifier = joblib.load('models/issue_classifier.joblib')
urgency_classifier = joblib.load('models/urgency_classifier.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/scaler.joblib')
product_list = joblib.load('models/product_list.joblib')

# --- 3. Initialize NLTK Components ---
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# --- 4. Helper Functions (from notebook) ---
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower().strip()
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(processed_tokens)

def extract_entities_advanced(text):
    extracted_entities = {"products": [], "dates": [], "complaint_phrases": []}
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
    date_pattern = r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2}(?:st|nd|rd|th)?(?:,?\s\d{4})?|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:,?\s\d{4})?)\b'
    extracted_entities["dates"] = re.findall(date_pattern, text, re.IGNORECASE)
    complaint_grammar = r"VP: {<VB.*>+<RB|RP|JJ|VBN|VBG>*}"
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

# --- 5. Main Prediction Function ---
def process_and_predict(raw_text):
    # Handle empty or whitespace-only input
    if not raw_text or not raw_text.strip():
        return "N/A", "N/A", {"message": "Please enter some text."}

    try:
        # ML Prediction part
        processed_text_for_model = preprocess_text(raw_text)
        
        # Another safety check
        if not processed_text_for_model:
            return "Could not classify", "Could not classify", {"message": "Input text too short after cleaning."}

        text_tfidf = tfidf_vectorizer.transform([processed_text_for_model])
        ticket_len = len(raw_text.split())
        sentiment = TextBlob(raw_text).sentiment.polarity
        additional_features = scaler.transform([[ticket_len, sentiment]])
        combined_features = hstack([text_tfidf, additional_features])
        
        predicted_issue = issue_classifier.predict(combined_features)[0]
        predicted_urgency = urgency_classifier.predict(combined_features)[0]
        
        # Entity Extraction part
        entities = extract_entities_advanced(raw_text)
        
        return predicted_issue, predicted_urgency, entities
    
    except Exception as e:
        # Catch any other unexpected errors and return a helpful message
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message) # This will print to the Hugging Face logs for debugging
        return "Error", "Error", {"error": error_message}

# --- 6. Gradio Interface Definition ---
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
        description="Enter a customer support ticket text to automatically classify its issue type, urgency level, and extract key entities.",
        allow_flagging="never",
        examples=[
            ["My FitRun Treadmill is broken and the screen won't turn on. I ordered it on May 1st, 2024."],
            ["I was double-charged for my PowerMax Battery order #8876. Please issue a refund immediately."],
            ["I have a question about the warranty for the SoundWave 300."]
        ]
    )

    app_interface.launch()