# Intelligent Support Ticket Classification with RAG

An end-to-end AI-powered customer support system that automates support ticket classification and response generation using Retrieval-Augmented Generation (RAG). The system combines traditional machine learning, transformer models, semantic search, and cloud deployment to improve customer support efficiency.

---

## Features

- Automatic ticket classification into:
  - Type
  - Queue
  - Priority
- AI-generated customer responses using Retrieval-Augmented Generation (RAG)
- Semantic retrieval with Sentence-BERT embeddings and Azure AI Search
- Azure Machine Learning model deployment
- FastAPI REST API backend
- JWT-based authentication and authorization
- SQLite database for users, tickets, responses, logs, retraining_alerts and feedback
- Client and Administrator dashboards
- MLflow experiment tracking
- Monitoring and automated retraining recommendations

---
---

## Dataset

This project is based on the **Multilingual Customer Support Tickets** dataset from Kaggle. Since this project focuses on English-language support ticket classification and response generation, the English subset of the dataset was used after data cleaning and preprocessing.

**Dataset:** https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

## System Architecture

```text
Client
   │
   ▼
Frontend
   │
   ▼
FastAPI Backend
   │
   ├────────► Azure Machine Learning
   │              │
   │              ▼
   │      Type • Queue • Priority
   │
   ├────────► Azure AI Search
   │              │
   │              ▼
   │      Similar Tickets
   │
   ├────────► Qwen LLM
   │              │
   │              ▼
   │      AI Response
   │
   ▼
SQLite Database
   │
   ▼
Admin Dashboard
```

---

## Technologies Used

### Programming Language

- Python

### Machine Learning & NLP

- Scikit-learn
- TF-IDF
- LinearSVC
- DistilBERT
- Sentence-BERT
- Retrieval-Augmented Generation (RAG)
- Qwen2.5-7B-Instruct

### Cloud & Deployment

- Azure Machine Learning
- Azure AI Search

### Backend

- FastAPI
- JWT Authentication
- SQLAlchemy

### Database

- SQLite

### MLOps

- MLflow
- Model Monitoring
- Automated Retraining Alerts

---

## Project Structure

```
Intelligent-Support-Ticket-System/
│
├── backend/
│   ├── backend/
│   ├── backend_connector/
│   └── Mlops/
│
│
├── rag/
│   ├── build_vector_store.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── reports/
│   ├── traditional/
│   ├── bert/
│   ├── rag/
│   ├── comparison/
│   ├── processed_eda/
│   └── *.png
│
├── src/
│   ├── clean.py
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── leakage_check.py
│   ├── eda_raw.py
│   ├── eda_processed.py
│   ├── traditional_model.py
│   ├── train_classifier.py
│   ├── traditional_confusion_matrix.py
│   └── distilbert_confusion_matrix.py
│
├── admin_dashboard.html
├── client_dashboard.html
├── main.html
├── requirements.txt
├── Infographic_designer.jpg
├── support_ticket_report.pdf
├── project_documentation.pdf
└── README.md
```

---

## Workflow

1. User submits a support ticket.
2. Azure ML predicts:
   - Ticket Type
   - Queue
   - Priority
3. Azure AI Search retrieves similar historical tickets.
4. Qwen generates a context-aware response.
5. Ticket, prediction, and response are stored in SQLite.
6. User can submit feedback or reopen the ticket.
7. Administrators monitor system performance and retraining alerts.

---

## Machine Learning Models

Three approaches were evaluated:

| Model | Purpose |
|--------|----------|
| TF-IDF + LinearSVC | Traditional ticket classification |
| DistilBERT | Transformer-based classification |
| RAG | Semantic retrieval and response generation |

---

## MLOps

The project includes:

- MLflow experiment tracking
- Performance monitoring
- Confidence monitoring
- Feedback monitoring
- Ticket reopen monitoring
- Automated retraining recommendations

---

## Installation

### Clone the repository

```bash
git clone https://github.com/rahiqkhalid12/Intelligent-Support-Ticket-System.git

cd Intelligent-Support-Ticket-System
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file and configure:

```
AZURE_ML_ENDPOINT=
AZURE_ML_API_KEY=

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX=

JWT_SECRET_KEY=

DATABASE_URL=
```

---

## Running the Project

Start the backend

```bash
uvicorn app:app --reload
```

The API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

Launch the frontend and access the application through your browser.

---

## Future Improvements

- Multilingual support
- Larger domain-specific language models
- Cloud database integration
- Continuous automated retraining
- Additional support ticket categories

---

## Contributors

- **Rahiq Khaled** – Dataset collection, data cleaning, exploratory data analysis (EDA) on raw and processed data, MLflow experiment tracking and monitoring, and JWT-based authentication and authorization, SQLite database, login/signup page.
- **Tasneem Khaled** – Data preprocessing, TF-IDF and Sentence-BERT embeddings,Backend & Integration using FastAPI.
- **Aya Elsayed** – Azure Machine Learning model deployment, KPI dashboard implementation, retraining trigger development.
- **Radwa Ahmed** – Model development (TF-IDF, DistilBERT, and RAG), model evaluation using Accuracy, F1-score, and BLEU, confusion matrix analysis, and performance comparison between Traditional ML, DistilBERT, and RAG.
- **Aya Abdelshafy** – Azure AI Search (Vector Database) integration, semantic retrieval pipeline.

### Frontend
The frontend (client dashboard and administrator dashboard) was developed collaboratively by all team members.
---

## License

This project was developed for academic and educational purposes.
