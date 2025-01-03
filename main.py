from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the model and tokenizer for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text from the form
    article_text = request.form['article_text']
    
    # Summarize the article
    summary = summarizer(article_text, max_length=1000, min_length=30, do_sample=False)
    return jsonify(summary=summary[0]['summary_text'])

if __name__ == '__main__':
    app.run(debug=True)
