from flask import Flask, request, render_template
from markupsafe import escape
import os
from rag_engine import RAGEngine
import re

# Initialize Flask app
app = Flask(__name__)

# Initialize RAGEngine without any large files
rag_engine = RAGEngine()
print("RAG engine initialized without FAISS index. Similarity search disabled.")

# Helper function to check if input is a valid URL

def is_valid_url(url):
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)

        # Check if input is a URL
        if is_valid_url(news):
            analysis = rag_engine.analyze_news_from_url(news)
            analysis_type = "URL"
        else:
            analysis = rag_engine.analyze_news(news)
            analysis_type = "Text"

        # No similarity search available
        similar_articles_text = "Similarity search is disabled (FAISS index not available)."

        return render_template(
            "prediction.html",
            prediction_text=f"News analysis: {analysis.get('analysis', 'N/A')}",
            confidence=analysis.get('confidence', 0.0),
            similar_articles=similar_articles_text,
            realtime_news=analysis.get('realtime_news', []),
            reason=analysis.get('reason', ''),
            verification_details=analysis.get('verification_details', {}),
            analysis_type=analysis_type,
            extracted_data=analysis.get('extracted_data', {})
        )
    else:
        return render_template(
            "prediction.html",
            prediction_text="",
            confidence=0.0,
            similar_articles="",
            realtime_news=[],
            reason="",
            verification_details={},
            analysis_type="",
            extracted_data={}
        )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)  