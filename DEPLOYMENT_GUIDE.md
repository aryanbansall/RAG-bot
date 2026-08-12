# 🚀 RAG Document QA System: Local Setup & Deployment Guide

This guide explains how to run the interactive RAG Document Question Answering System locally and deploy it to cloud hosting platforms.

---

## 🛠️ 1. Local Setup & Running the UI

### Step 1: Install Dependencies
Ensure you are in your project environment, then install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (Optional)
Create a `.env` file in the project root directory with your API keys:

```env
API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```
*(Note: You can also enter API keys directly inside the application UI sidebar).*

### Step 3: Run the Streamlit Application
Start the server using:

```bash
streamlit run app.py
```

The app will launch automatically in your browser at `http://localhost:8501`.

---

## ☁️ 2. Deploying to Streamlit Community Cloud (Recommended - Free)

Streamlit Community Cloud provides 1-click free hosting directly from your GitHub repository.

### Step-by-Step Deployment:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Upgrade Streamlit UI and add deployment config"
   git push origin main
   ```
2. **Go to Streamlit Cloud**:
   Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. **Deploy New App**:
   - Click **"New app"**.
   - Select your Repository (`Question-Answering-System-using-RAG`), Branch (`main`), and Main file path (`app.py`).
4. **Configure Secrets / Environment Variables**:
   - Click **"Advanced settings"** -> **"Secrets"**.
   - Paste your secret keys:
     ```toml
     API_KEY = "your_groq_api_key_here"
     GOOGLE_API_KEY = "your_google_gemini_api_key_here"
     ```
5. **Click "Deploy!"**:
   Your app will be built and hosted at `https://<your-app-name>.streamlit.app`.

---

## 🐳 3. Deploying with Docker (Cloud Run, Railway, Render, VPS)

The project includes a production `Dockerfile`.

### Build & Run Container Locally:

```bash
# Build the Docker image
docker build -t rag-qa-app .

# Run the container locally on port 8501
docker run -d -p 8501:8501 --env-file .env --name rag-app rag-qa-app
```

Access the app at `http://localhost:8501`.

### Deploying Docker Image to Google Cloud Run:

```bash
# Submit build to Google Artifact Registry / Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rag-qa-app

# Deploy to Cloud Run
gcloud run deploy rag-qa-app \
  --image gcr.io/YOUR_PROJECT_ID/rag-qa-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_KEY=your_groq_key,GOOGLE_API_KEY=your_google_key
```

---

## 🤗 4. Deploying to Hugging Face Spaces (Free)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Push your repository code to the Hugging Face Space repository.
4. Add `API_KEY` and `GOOGLE_API_KEY` under Space **Settings** -> **Variables and Secrets**.

---

## 🌟 Key Features of the Upgraded UI
- 💬 **Conversational Chat Interface**: Dynamic message history (`st.chat_message` & `st.chat_input`).
- 📁 **Flexible Document Ingestion**: Ingest PDFs from `./Assets` or drag-and-drop custom PDF files.
- 🔍 **Source Attribution**: Collapsible cards showing exact retrieved text chunks and page numbers.
- ⚡ **Performance Metrics**: Live timing indicators for answer generation.
- 🎛️ **Sidebar Control Panel**: API Key inputs, Groq LLM model selection, temperature slider, chunk size controls, and index reset capabilities.
