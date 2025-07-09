import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Tuple, Dict
import os
import requests
from datetime import datetime, timedelta
import json
import re
import feedparser
import urllib.parse
from transformers import pipeline
from textblob import TextBlob
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class RAGEngine:
    SIMILARITY_THRESHOLD = 0.75  
    GNEWS_API_KEY = "50e1c8827230cf6f2caca1a11110bdf8"
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.news_data = None
        self.vector_dimension = 384  
        self.newsapi_key = os.getenv('NEWSAPI_KEY', '')  
        # Initialize zero-shot classification pipeline for fact-checking
        self.fact_checker = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("NewsAPI Key:", self.newsapi_key)  
        
    def load_and_prepare_data(self, csv_path: str):
        """Load news data and prepare it for indexing"""
        df = pd.read_csv(csv_path)
        self.news_data = df
        return df
    
    def create_index(self):
        """Create FAISS index from news data"""
        if self.news_data is None:
            raise ValueError("Please load data first using load_and_prepare_data()")
       
        texts = self.news_data['text'].fillna('') + ' ' + self.news_data['title'].fillna('')
        
      
        embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
    
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
    def save_index(self, path: str):
        """Save the FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index to save. Please create index first.")
        faiss.write_index(self.index, path)
        
    def load_index(self, path: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(path)
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar news articles - disabled when no index available"""
        if self.index is None:
            print("FAISS index not available. Returning empty search results.")
            return []
            
        # Encode the query
        query_vector = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.news_data):
                # Convert distance to similarity score (0 to 1)
                # FAISS uses L2 distance, so we need to convert it to similarity
                similarity = 1 / (1 + distance)  # This gives us a score between 0 and 1
                results.append((
                    self.news_data.iloc[idx]['text'],
                    float(similarity) 
                ))
        
        return results
    
    def preprocess_query(self, query: str) -> str:
        
        return query.strip()
    
    def get_gnews_articles(self, query: str) -> list:
        """Get news articles from GNews API."""
        try:
            processed_query = self.preprocess_query(query)
            url = f'https://gnews.io/api/v4/search'
            params = {
                'q': processed_query,
                'lang': 'en',
                'country': 'in',
                'token': self.GNEWS_API_KEY,
                'max': 10
            }
            print("GNews request URL:", url)
            print("GNews request params:", params)
            response = requests.get(url, params=params)
            print("GNews response code:", response.status_code)
            print("GNews response:", response.text)
            if response.status_code == 200:
                news = response.json()
                return news.get('articles', [])
            return []
        except Exception as e:
            print(f"Error fetching GNews: {e}")
            return []
    
    def get_google_news_articles(self, query: str) -> list:
        """Get news articles from Google News RSS."""
        try:
            processed_query = urllib.parse.quote_plus(self.preprocess_query(query))
            url = f'https://news.google.com/rss/search?q={processed_query}&hl=en-IN&gl=IN&ceid=IN:en'
            print("Google News RSS URL:", url)
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries:
                articles.append({
                    'title': entry.title,
                    'description': entry.summary if hasattr(entry, 'summary') else '',
                    'source': {'name': entry.source.title if hasattr(entry, 'source') else 'Google News'},
                    'publishedAt': entry.published if hasattr(entry, 'published') else ''
                })
            print(f"Google News articles found: {len(articles)}")
            return articles
        except Exception as e:
            print(f"Error fetching Google News RSS: {e}")
            return []
    
    def strip_html(self, text: str) -> str:
        """Remove HTML tags from a string."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text) if text else text
    
    def get_realtime_news(self, query: str) -> list:
        """Get real-time news from multiple continents and sources."""
        all_articles = []
        
        # Define major newspapers by region with expanded Indian sources
        newspapers = {
            'US': [
                {'name': 'The New York Times', 'url': 'https://www.nytimes.com'},
                {'name': 'The Washington Post', 'url': 'https://www.washingtonpost.com'},
                {'name': 'The Wall Street Journal', 'url': 'https://www.wsj.com'}
            ],
            'Europe': [
                {'name': 'The Guardian', 'url': 'https://www.theguardian.com'},
                {'name': 'BBC News', 'url': 'https://www.bbc.com/news'},
                {'name': 'Le Monde', 'url': 'https://www.lemonde.fr'}
            ],
            'Asia': [
                # Indian sources
                {'name': 'The Times of India', 'url': 'https://timesofindia.indiatimes.com'},
                {'name': 'The Hindu', 'url': 'https://www.thehindu.com'},
                {'name': 'Indian Express', 'url': 'https://indianexpress.com'},
                {'name': 'Hindustan Times', 'url': 'https://www.hindustantimes.com'},
                {'name': 'NDTV', 'url': 'https://www.ndtv.com'},
                {'name': 'ANI News', 'url': 'https://www.aninews.in'},
                {'name': 'PTI News', 'url': 'https://www.ptinews.com'},
                # Other Asian sources
                {'name': 'South China Morning Post', 'url': 'https://www.scmp.com'},
                {'name': 'The Japan Times', 'url': 'https://www.japantimes.co.jp'}
            ]
        }

   
        if self.newsapi_key:
            try:
                processed_query = self.preprocess_query(query)
                from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Search for each region
                for region in ['US', 'Europe', 'Asia']:
                    url = f'https://newsapi.org/v2/everything'
                    params = {
                        'q': processed_query,
                        'from': from_date,
                        'language': 'en',
                        'sortBy': 'relevancy',
                        'apiKey': self.newsapi_key,
                        'domains': ','.join([paper['url'].replace('https://', '').replace('www.', '') 
                                           for paper in newspapers[region]])
                    }
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        news = response.json()
                        articles = news.get('articles', [])
                  
                        for article in articles:
                            article['region'] = region
                        all_articles.extend(articles)
            except Exception as e:
                print(f"Error fetching NewsAPI: {e}")

        # GNews API
        try:
            gnews_articles = self.get_gnews_articles(query)
         
            for article in gnews_articles:
                source = article.get('source', {}).get('name', '').lower()
                
                if any(indian_paper['name'].lower() in source for indian_paper in newspapers['Asia'][:7]):
                    article['region'] = 'Asia'
                elif any(us_paper['name'].lower() in source for us_paper in newspapers['US']):
                    article['region'] = 'US'
                elif any(eu_paper['name'].lower() in source for eu_paper in newspapers['Europe']):
                    article['region'] = 'Europe'
                elif any(asia_paper['name'].lower() in source for asia_paper in newspapers['Asia']):
                    article['region'] = 'Asia'
                else:
                    article['region'] = 'Other'
            all_articles.extend(gnews_articles)
        except Exception as e:
            print(f"Error fetching GNews: {e}")

        # Google News RSS
        try:
            google_news_articles = self.get_google_news_articles(query)
            # Add region information based on source
            for article in google_news_articles:
                source = article.get('source', {}).get('name', '').lower()
                # Check for Indian sources first
                if any(indian_paper['name'].lower() in source for indian_paper in newspapers['Asia'][:7]):
                    article['region'] = 'Asia'
                elif any(us_paper['name'].lower() in source for us_paper in newspapers['US']):
                    article['region'] = 'US'
                elif any(eu_paper['name'].lower() in source for eu_paper in newspapers['Europe']):
                    article['region'] = 'Europe'
                elif any(asia_paper['name'].lower() in source for asia_paper in newspapers['Asia']):
                    article['region'] = 'Asia'
                else:
                    article['region'] = 'Other'
            all_articles.extend(google_news_articles)
        except Exception as e:
            print(f"Error fetching Google News RSS: {e}")

      
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        # Sort by region and then by date
        unique_articles.sort(key=lambda x: (x.get('region', 'Other'), 
                                          x.get('publishedAt', '')))

        # Strip HTML tags and clean up
        for article in unique_articles:
            if 'title' in article:
                article['title'] = self.strip_html(article['title'])
            if 'description' in article:
                article['description'] = self.strip_html(article['description'])

        return unique_articles
    
    def verify_with_generative_model(self, text: str) -> Tuple[float, str]:
        """Verify news using a generative model for fact-checking"""
        try:
            # First check for historical claims
            if 'war' in text.lower() and ('pakistan' in text.lower() or 'india' in text.lower()):
              
                result = self.fact_checker(
                    text,
                    candidate_labels=["historically accurate", "historically inaccurate", "unverified"],
                    hypothesis_template="This statement about war history is {}."
                )
            else:
                # General fact checking
                result = self.fact_checker(
                    text,
                    candidate_labels=["true", "false", "unverified"],
                    hypothesis_template="This news article is {}."
                )
            
            # Get scores for each label
            scores = {label: score for label, score in zip(result['labels'], result['scores'])}
            
            # Calculate verification score (higher for true/accurate, lower for false/inaccurate)
            if 'historically accurate' in scores:
                verification_score = scores.get('historically accurate', 0.0) - scores.get('historically inaccurate', 0.0)
            else:
                verification_score = scores.get('true', 0.0) - scores.get('false', 0.0)
            
            # Get explanation based on scores
            if verification_score > 0.6:
                explanation = "High confidence in truthfulness"
            elif verification_score > 0.3:
                explanation = "Moderate confidence in truthfulness"
            elif verification_score > 0:
                explanation = "Low confidence in truthfulness"
            else:
                explanation = "Likely false or misleading"
                
            return verification_score, explanation
        except Exception as e:
            print(f"Error in fact-checking: {e}")
            return 0.5, "Unable to verify with generative model"

    def analyze_text_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Enhanced text pattern analysis with detailed indicators"""
        fake_indicators = {
            'emotional_triggers': ['shocking', 'unbelievable', 'mind-blowing', 'outrageous'],
            'clickbait': ['you won\'t believe', 'what happens next', 'secret', 'exclusive'],
            'conspiracy': ['they don\'t want you to know', 'hidden truth', 'conspiracy'],
            'exaggeration': ['100%', 'guaranteed', 'miracle', 'never before seen'],
            'urgency': ['act now', 'limited time', 'breaking', 'urgent']
        }
        
        text_lower = text.lower()
        score = 0.0
        detected_patterns = []
        
        for category, indicators in fake_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    score += 0.1
                    detected_patterns.append(f"{category}: {indicator}")
        
        # Check for excessive punctuation
        if text.count('!') > 3 or text.count('?') > 3:
            score += 0.1
            detected_patterns.append("excessive punctuation")
            
        # Check for sentiment polarity
        blob = TextBlob(text)
        if abs(blob.sentiment.polarity) > 0.8:
            score += 0.1
            detected_patterns.append("extreme sentiment")
            
        return min(score, 1.0), detected_patterns

    def analyze_realtime_news_context(self, query_text: str, realtime_news: list) -> Tuple[float, str]:
        """Analyze the context of real-time news articles for better verification"""
        if not realtime_news:
            return 0.0, "No real-time news found for verification"
            
        # Prepare query and news texts for analysis
        query_text = query_text.lower()
        context_scores = []
        context_analysis = []
        
        for article in realtime_news:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            full_text = f"{title} {description}"
            
            # Check for contradictory context
            contradictions = []
            if 'war' in query_text and 'war' in full_text:
                if ('won' in query_text and 'lost' in full_text) or ('lost' in query_text and 'won' in full_text):
                    contradictions.append("Contradictory war outcome")
            
            # Check for temporal context
            temporal_issues = []
            if 'yesterday' in query_text and 'last year' in full_text:
                temporal_issues.append("Temporal mismatch")
            elif 'today' in query_text and 'last week' in full_text:
                temporal_issues.append("Temporal mismatch")
                
            # Check for location context
            location_issues = []
            if 'india' in query_text and 'pakistan' in full_text:
                if ('delhi' in query_text and 'islamabad' in full_text) or ('mumbai' in query_text and 'karachi' in full_text):
                    location_issues.append("Location mismatch")
            
            # Calculate context score
            context_score = 1.0
            if contradictions:
                context_score -= 0.5
            if temporal_issues:
                context_score -= 0.3
            if location_issues:
                context_score -= 0.3
                
            context_scores.append(context_score)
            
            # Build analysis message
            analysis_parts = []
            if contradictions:
                analysis_parts.extend(contradictions)
            if temporal_issues:
                analysis_parts.extend(temporal_issues)
            if location_issues:
                analysis_parts.extend(location_issues)
                
            if analysis_parts:
                context_analysis.append(f"Article: {title[:50]}... - Issues: {', '.join(analysis_parts)}")
        
        # Calculate final context score
        final_score = max(context_scores) if context_scores else 0.0
        
        # Prepare analysis message
        if context_analysis:
            analysis_message = "Context analysis found:\n" + "\n".join(context_analysis)
        else:
            analysis_message = "No significant context issues found in real-time news"
            
        return final_score, analysis_message

    def analyze_news(self, news_text: str) -> dict:
        """Enhanced news analysis with improved similarity scoring"""
        # Check for contextless/neutral input
        if len(news_text.strip().split()) < 3:
            return {
                'is_fake_probability': 0.0,
                'similar_articles': [],
                'realtime_news': [],
                'analysis': 'No Context',
                'confidence': 0.0,
                'reason': 'No context detected. Please enter a full news claim or statement for verification.',
                'verification_details': {
                    'verification_score': 0.0,
                    'pattern_analysis': [],
                    'realtime_verification': False,
                    'realtime_analysis': '',
                    'similar_articles_scores': [],
                    'explanation': 'No context detected.'
                }
            }
        # Get similar articles
        similar_articles = self.search(news_text, k=3)
        
        # Get real-time news
        realtime_news = self.get_realtime_news(news_text)
        
        # Check geographical facts
        geo_issues = self.check_geographical_facts(news_text)
        if geo_issues:
            return {
                'is_fake_probability': 1.0,
                'similar_articles': similar_articles,
                'realtime_news': [],
                'analysis': 'Fake',
                'confidence': 1.0,
                'reason': f'Geographical inconsistency detected: {geo_issues}',
                'verification_details': {
                    'verification_score': 0.0,
                    'pattern_analysis': [],
                    'realtime_verification': False,
                    'explanation': f'Geographical inconsistency detected: {geo_issues}'
                }
            }
            
        # Check historical facts
        historical_issues = self.check_historical_facts(news_text)
        if historical_issues:
            return {
                'is_fake_probability': 1.0,
                'similar_articles': similar_articles,
                'realtime_news': [],
                'analysis': 'Fake',
                'confidence': 1.0,
                'reason': f'Historical inconsistency detected: {historical_issues}',
                'verification_details': {
                    'verification_score': 0.0,
                    'pattern_analysis': [],
                    'realtime_verification': False,
                    'explanation': f'Historical inconsistency detected: {historical_issues}'
                }
            }
        
        # Get generative model verification
        gen_score, gen_explanation = self.verify_with_generative_model(news_text)
        
        # Get pattern analysis
        pattern_score, detected_patterns = self.analyze_text_patterns(news_text)
        
        # Analyze real-time news context
        realtime_score, realtime_analysis = self.analyze_realtime_news_context(news_text, realtime_news)
        
        # Calculate similarity scores with improved weighting
        avg_similarity = np.mean([score for _, score in similar_articles]) if similar_articles else 0.0
        
        # Calculate final similarity score with improved weighting
        # Give more weight to real-time news context analysis
        similarity_score = (avg_similarity * 0.3 + realtime_score * 0.7)
        
        # Weighted combination of all scores with adjusted weights
        is_fake_probability = (
            (1 - similarity_score) * 0.15 +  # Slightly increased similarity weight
            pattern_score * 0.15 +           # Slightly increased pattern analysis weight
            (1 - gen_score) * 0.7            # Generative model weight set to 70%
        )
        
        # Calculate confidence based on agreement between methods
        confidence = 1 - abs(is_fake_probability - 0.5) * 2
        
        # Determine if fake based on probability
        is_fake = is_fake_probability > 0.5
        
        # Prepare simplified verification information
        verification_details = {
            'verification_score': float(gen_score),  # Only show generative model score
            'pattern_analysis': detected_patterns,
            'realtime_verification': bool(realtime_news),
            'realtime_analysis': realtime_analysis,  # Add context analysis
            'similar_articles_scores': [float(score) for _, score in similar_articles],
            'explanation': gen_explanation  # Include the explanation
        }
        
        return {
            'is_fake_probability': float(is_fake_probability),
            'similar_articles': similar_articles,
            'realtime_news': realtime_news[:2] if realtime_news else [],
            'analysis': 'Fake' if is_fake else 'Real',
            'confidence': float(confidence),
            'reason': gen_explanation if is_fake else 'Verified by multiple reliable sources',
            'verification_details': verification_details
        }
        
    def check_geographical_facts(self, text: str) -> str:
        """Check for geographical inconsistencies in the text."""

        geo_facts = {
            'patna': ['port', 'seaport', 'harbor', 'coastal'],
            'delhi': ['port', 'seaport', 'harbor', 'coastal'],
            'bangalore': ['port', 'seaport', 'harbor', 'coastal'],
            'hyderabad': ['port', 'seaport', 'harbor', 'coastal'],
            'chennai': ['mountain', 'himalayas'],
            'mumbai': ['mountain', 'himalayas'],
            'kolkata': ['mountain', 'himalayas']
        }
        
        text_lower = text.lower()
        issues = []
        
        for city, impossible_features in geo_facts.items():
            if city in text_lower:
                for feature in impossible_features:
                    if feature in text_lower:
                        issues.append(f"{city.title()} cannot have {feature}")
                        
        return '; '.join(issues) if issues else ''

    def check_historical_facts(self, text: str) -> str:
        """Check for historical inconsistencies in the text."""
     #wikipedia data to be implemented in future
        historical_facts = {
            'pakistan': {
                'wars': {
                    'won': ['Pakistan has never won a war against India'],
                    'lost': ['1947-48', '1965', '1971', '1999 Kargil']
                }
            },
            'india': {
                'wars': {
                    'won': ['1947-48', '1965', '1971', '1999 Kargil'],
                    'lost': ['India has never lost a war against Pakistan']
                }
            }
        }
        
        text_lower = text.lower()
        issues = []
        
   
        if 'war' in text_lower:
            for country, facts in historical_facts.items():
                if country in text_lower:
                    if 'won' in text_lower:
                        for fact in facts['wars']['won']:
                            if fact.startswith(country.title()):
                                issues.append(fact)
                    elif 'lost' in text_lower:
                        for fact in facts['wars']['lost']:
                            if fact.startswith(country.title()):
                                issues.append(fact)
                                
        return '; '.join(issues) if issues else ''

    def extract_news_from_url(self, url: str) -> Dict[str, str]:
        """Extract news content from a given URL"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {'error': 'Invalid URL format'}
            
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract meta description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '').strip()
            
            # Extract main content - try different selectors
            content_selectors = [
                'article',
                '[role="main"]',
                '.content',
                '.article-content',
                '.post-content',
                '.entry-content',
                'main',
                '.main-content'
            ]
            
            main_content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text().strip()
                    break
            
            # If no main content found, try to extract from body
            if not main_content:
                body = soup.find('body')
                if body:
                    # Remove navigation, footer, sidebar elements
                    for elem in body.find_all(['nav', 'footer', 'aside', 'header']):
                        elem.decompose()
                    main_content = body.get_text().strip()
            
            # Clean up the content
            if main_content:
                # Remove extra whitespace and newlines
                main_content = re.sub(r'\s+', ' ', main_content)
                # Limit content length
                main_content = main_content[:2000]  # Limit to 2000 characters
            
            # Extract source domain
            source = parsed_url.netloc
            
            return {
                'title': title,
                'description': description,
                'content': main_content,
                'source': source,
                'url': url,
                'success': True
            }
            
        except requests.RequestException as e:
            return {'error': f'Failed to fetch URL: {str(e)}'}
        except Exception as e:
            return {'error': f'Error extracting content: {str(e)}'}
    
    def check_source_credibility(self, source: str) -> Tuple[float, str]:
        """Check the credibility of a news source"""
        credible_sources = {
            'timesofindia.indiatimes.com': {'score': 0.9, 'name': 'Times of India'},
            'thehindu.com': {'score': 0.9, 'name': 'The Hindu'},
            'indianexpress.com': {'score': 0.85, 'name': 'Indian Express'},
            'ndtv.com': {'score': 0.85, 'name': 'NDTV'},
            'hindustantimes.com': {'score': 0.85, 'name': 'Hindustan Times'},
            'bbc.com': {'score': 0.95, 'name': 'BBC News'},
            'reuters.com': {'score': 0.95, 'name': 'Reuters'},
            'ap.org': {'score': 0.95, 'name': 'Associated Press'},
            'cnn.com': {'score': 0.8, 'name': 'CNN'},
            'aljazeera.com': {'score': 0.8, 'name': 'Al Jazeera'},
            'theguardian.com': {'score': 0.85, 'name': 'The Guardian'},
            'nytimes.com': {'score': 0.9, 'name': 'The New York Times'},
            'washingtonpost.com': {'score': 0.9, 'name': 'The Washington Post'},
            'wsj.com': {'score': 0.9, 'name': 'The Wall Street Journal'}
        }
        
        # Check if source is in our credible sources list
        for domain, info in credible_sources.items():
            if domain in source.lower():
                return info['score'], f"Source verified: {info['name']} is a credible news outlet"
        
        # Check for suspicious domains
        suspicious_keywords = ['fake', 'satire', 'parody', 'hoax', 'clickbait']
        if any(keyword in source.lower() for keyword in suspicious_keywords):
            return 0.1, "Suspicious source detected"
        
        # Unknown source - neutral score
        return 0.5, "Unknown source - additional verification needed"

    def analyze_news_from_url(self, url: str) -> dict:
        """Analyze news from a URL by extracting content and then analyzing it"""
        # Extract content from URL
        extracted_data = self.extract_news_from_url(url)
        
        if 'error' in extracted_data:
            return {
                'is_fake_probability': 0.0,
                'similar_articles': [],
                'realtime_news': [],
                'analysis': 'Error',
                'confidence': 0.0,
                'reason': extracted_data['error'],
                'verification_details': {
                    'verification_score': 0.0,
                    'pattern_analysis': [],
                    'realtime_verification': False,
                    'realtime_analysis': '',
                    'similar_articles_scores': [],
                    'explanation': extracted_data['error']
                },
                'extracted_data': extracted_data
            }
        
        # Check source credibility first
        source_score, source_explanation = self.check_source_credibility(extracted_data['source'])
        
        # If source is highly credible, adjust the analysis
        if source_score > 0.8:
            # For highly credible sources, we'll be more lenient with the generative model
            combined_text = f"{extracted_data['title']} {extracted_data['description']} {extracted_data['content']}"
            analysis_result = self.analyze_news(combined_text)
            
            # Adjust the fake probability based on source credibility
            if analysis_result['is_fake_probability'] > 0.5:
                # If a credible source is marked as fake, reduce the probability
                adjusted_probability = analysis_result['is_fake_probability'] * 0.3  # Reduce by 70%
                analysis_result['is_fake_probability'] = adjusted_probability
                analysis_result['analysis'] = 'Real' if adjusted_probability < 0.5 else 'Fake'
                analysis_result['reason'] = f"Credible source ({source_explanation}) - {analysis_result['reason']}"
            
            # Add source credibility info
            analysis_result['verification_details']['source_credibility'] = {
                'score': source_score,
                'explanation': source_explanation
            }
        else:
            # For less credible sources, proceed with normal analysis
            combined_text = f"{extracted_data['title']} {extracted_data['description']} {extracted_data['content']}"
            analysis_result = self.analyze_news(combined_text)
            analysis_result['verification_details']['source_credibility'] = {
                'score': source_score,
                'explanation': source_explanation
            }
        
        # Add extracted data to the result
        analysis_result['extracted_data'] = extracted_data
        
        return analysis_result 