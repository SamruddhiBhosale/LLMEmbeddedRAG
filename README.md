# LLMEmbeddedRAG

## Model - 
model params   = 7.25 B
model name     = Mistral-7B-Instruct-v0.3

## Steps to run :
### Prerequisites:
Python 3.10+ (for best compatibility).
Homebrew (for installing Ollama locally on macOS).
PyTorch and related dependencies (required by sentence-transformers).
### Install Required Dependencies:
pip install -r requirements.txt

### Install Ollama (for Local LLM):
brew install ollama
ollama run mistral  # This will download and start the Mistral model

### Run the Streamlit App:
streamlit run app.py


<img width="1438" alt="image" src="https://github.com/user-attachments/assets/5859b041-f697-4285-9bd2-aad92f123d5a" />
