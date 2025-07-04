# Contributing to RAG-Powered Fake News Detection System

Thank you for your interest in contributing to our Fake News Detection project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork the Repository
1. Go to the main repository page
2. Click the "Fork" button in the top right corner
3. This creates a copy of the repository in your GitHub account

### 2. Clone Your Fork
```bash
git clone https://github.com/yourusername/rag-fake-news-detector.git
cd rag-fake-news-detector
```

### 3. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 4. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 5. Make Your Changes
- Write clean, well-documented code
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 6. Test Your Changes
```bash
# Run the application
python app.py

# Test the functionality
# Add unit tests if applicable
```

### 7. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature description"
```

### 8. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 9. Create a Pull Request
1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template
5. Submit the PR

## 📋 Pull Request Guidelines

### PR Template
```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tested locally
- [ ] Added unit tests
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🛠 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions small and focused

### File Structure
```
project/
├── app.py              # Main Flask application
├── rag_engine.py       # RAG engine implementation
├── requirements.txt    # Dependencies
├── templates/          # HTML templates
├── static/            # CSS, JS, images
├── tests/             # Unit tests
└── docs/              # Documentation
```

### Testing
- Write unit tests for new features
- Test edge cases and error conditions
- Ensure all tests pass before submitting PR

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Update API documentation if needed

## 🐛 Reporting Bugs

### Bug Report Template
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS, Ubuntu]
- Python Version: [e.g., 3.8, 3.9]
- Browser: [e.g., Chrome, Firefox]

## Additional Information
Screenshots, logs, etc.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature would be useful

## Proposed Implementation
How you think it should be implemented

## Alternatives Considered
Other approaches you've considered
```

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- pip
- Git
- API keys for NewsAPI and GNews

### Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/rag-fake-news-detector.git
cd rag-fake-news-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
python app.py
```

### Environment Variables
Create a `.env` file with:
```env
NEWSAPI_KEY=your_newsapi_key_here
GNEWS_API_KEY=your_gnews_api_key_here
FLASK_ENV=development
FLASK_DEBUG=1
```

## 📚 Documentation

### Code Documentation
- Use docstrings for all functions and classes
- Follow Google docstring format
- Include type hints where appropriate

### API Documentation
- Document all API endpoints
- Include request/response examples
- Specify error codes and messages

## 🚀 Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Heroku
git push heroku main
```

## 🤝 Community Guidelines

### Be Respectful
- Be kind and respectful to all contributors
- Welcome newcomers and help them get started
- Provide constructive feedback

### Communication
- Use clear, concise language
- Be specific when reporting issues
- Ask questions if something is unclear

### Code of Conduct
- Follow the project's code of conduct
- Report inappropriate behavior
- Help maintain a positive community

## 📞 Getting Help

### Resources
- [GitHub Issues](https://github.com/yourusername/rag-fake-news-detector/issues)
- [GitHub Discussions](https://github.com/yourusername/rag-fake-news-detector/discussions)
- [Documentation](https://github.com/yourusername/rag-fake-news-detector/wiki)

### Contact
- Email: your-email@example.com
- Discord: [Server Link]
- Twitter: [@YourHandle]

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to making the internet a more trustworthy place! 🌐✨ 