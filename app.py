# app.py
# A simple Flask backend for NLP tasks using spaCy.

from flask import Flask, request, jsonify
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import spacy
from collections import Counter

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing requests from the extension

# Load the spaCy English model
# You'll need to download this model first: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define common stop words (can be expanded)
# spaCy's default stop words are pretty good, but we can add more if needed
# For this example, we'll rely mostly on noun chunks and entities which naturally filter some noise.
CUSTOM_STOP_WORDS = {"page", "site", "website", "article", "content", "information", "click", "view", "menu", "home", "search", "contact", "news", "post", "blog", "comment", "read", "more", "share", "like", "follow", "also", "however", "therefore", "example", "e.g.", "i.e.", "etc", "fig", "image", "photo", "video", "copyright", "rights", "reserved", "terms", "privacy", "policy", "subscribe", "login", "logout", "register", "account", "learn", "help", "faq", "support", "services", "products", "company", "about", "us", "welcome", "thank", "thanks", "nbsp", "amp", "quot", "apos", "lt", "gt"}


@app.route('/analyze', methods=['POST'])
def analyze_text_route():
    """
    Analyzes text to extract keywords using noun chunks and named entities.
    Expects JSON input: {"text": "your text here"}
    Returns JSON: {"keywords": ["kw1", "kw2", ...]} or {"error": "message"}
    """
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing 'text' in JSON payload"}), 400

    text_to_analyze = request.json['text']

    if not text_to_analyze or not isinstance(text_to_analyze, str) or len(text_to_analyze.strip()) < 50: # Require some minimum text length
        return jsonify({"error": "Text is too short or invalid."}), 400

    try:
        # Process the text with spaCy
        doc = nlp(text_to_analyze)

        # Extract keywords:
        # 1. Noun chunks (phrases like "web development", "artificial intelligence")
        # 2. Named Entities (ORG, PERSON, GPE, etc.)
        
        potential_keywords = []

        # Add noun chunks
        for chunk in doc.noun_chunks:
            # Clean the chunk text: lowercase, remove leading/trailing punctuation/spaces
            clean_chunk = chunk.text.lower().strip(" .,;:!?-()[]{}'\"")
            # Filter out very short chunks or chunks that are mostly stop words or numbers
            if len(clean_chunk) > 3 and clean_chunk not in CUSTOM_STOP_WORDS and not clean_chunk.isdigit():
                # Further filter: ensure it's not just spaCy's stop words
                is_meaningful = False
                for token in chunk:
                    if not token.is_stop and not token.is_punct and not token.is_space:
                        is_meaningful = True
                        break
                if is_meaningful:
                    potential_keywords.append(clean_chunk)

        # Add named entities
        for ent in doc.ents:
            # Filter by entity types you find relevant (e.g., ORG, PRODUCT, EVENT, WORK_OF_ART, LAW, LOC, FAC, PERSON, NORP, GPE)
            # You can customize this list
            relevant_entity_labels = {"ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "LOC", "PERSON", "NORP", "GPE", "FACILITY"}
            if ent.label_ in relevant_entity_labels:
                clean_ent = ent.text.lower().strip(" .,;:!?-()[]{}'\"")
                if len(clean_ent) > 2 and clean_ent not in CUSTOM_STOP_WORDS and not clean_ent.isdigit():
                    potential_keywords.append(clean_ent)
        
        # Count frequencies and get top N keywords
        if not potential_keywords:
            return jsonify({"keywords": [], "message": "No significant keywords found after NLP processing."})

        keyword_counts = Counter(potential_keywords)
        # Get top 7 most common keywords, ensure they are somewhat frequent
        top_keywords = [kw for kw, count in keyword_counts.most_common(10) if count >= 1] # Adjust count threshold if needed

        # Remove duplicates that might arise if a noun chunk is also an entity
        unique_top_keywords = sorted(list(set(top_keywords)), key=lambda x: keyword_counts[x], reverse=True)[:7]


        if not unique_top_keywords:
             return jsonify({"keywords": [], "message": "Keywords found but did not meet frequency or uniqueness criteria."})

        return jsonify({"keywords": unique_top_keywords})

    except Exception as e:
        print(f"Error during NLP processing: {e}") # Log to server console
        return jsonify({"error": f"An internal error occurred during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    # Runs the Flask development server.
    # For production, use a proper WSGI server like Gunicorn.
    app.run(debug=True, port=5000)
