from flask import Flask, request, render_template
from markupsafe import escape
import pickle
import os
from rag_engine import RAGEngine
import re


rag_engine = RAGEngine()


if not os.path.exists("news_index.faiss"):
    print("Creating new index...")
    rag_engine.load_and_prepare_data("news.csv")
    rag_engine.create_index()
    rag_engine.save_index("news_index.faiss")
else:
    print("Loading existing index...")
    rag_engine.load_index("news_index.faiss")
    rag_engine.load_and_prepare_data("news.csv")

app = Flask(__name__)

def is_valid_url(url):
    """Check if the input is a valid URL"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
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
            # Analyze news from URL
            analysis = rag_engine.analyze_news_from_url(news)
            analysis_type = "URL"
        else:
            analysis = rag_engine.analyze_news(news)
            analysis_type = "Text"
        
        # Get similar articles for context
        similar_articles = analysis['similar_articles']
        similar_articles_text = "\n\n".join([f"Similar article {i+1}: {text[:200]}..." for i, (text, _) in enumerate(similar_articles)])
        
        return render_template(
            "prediction.html",
            prediction_text=f"News analysis: {analysis['analysis']}",
            confidence=analysis['confidence'],
            similar_articles=similar_articles_text,
            realtime_news=analysis['realtime_news'],
            reason=analysis.get('reason', ''),
            verification_details=analysis.get('verification_details', {}),
            analysis_type=analysis_type,
            extracted_data=analysis.get('extracted_data', {})
        )
    else:
        # Provide default values for GET request
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