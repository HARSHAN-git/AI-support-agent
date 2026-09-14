# AI Support Agent

### RAG-Based Customer Support Automation using Historical AmazonHelp Conversations

An AI-powered customer support agent that classifies customer issues, retrieves similar historical support conversations, generates a grounded response, and decides whether the issue can be safely handled automatically or should be escalated to a human agent.

The system is built using **Retrieval-Augmented Generation (RAG)** and historical customer-support conversations from the **Twitter Customer Support Dataset**.

---

## 🚀 Overview

Customer support systems need to handle large numbers of repetitive queries while avoiding incorrect or unsafe automated responses.

This project demonstrates an AI support workflow that combines:

- Intent classification
- Semantic search over historical support conversations
- Retrieval-Augmented Generation (RAG)
- Grounded response generation
- Automated escalation decisions
- A Streamlit-based interactive interface

The historical responses from **AmazonHelp** are used as the primary knowledge source for generating customer-facing responses.

---

## 🧠 System Architecture

```text
Customer Message
       │
       ▼
Intent Classifier
       │
       ▼
Semantic Search / RAG
       │
       ▼
Response Generator
       │
       ▼
Escalation Decision
       │
   ┌───┴────┐
   ▼        ▼
AUTO-     ESCALATE
HANDLE
```

---

## ✨ Features

### 1. Intent Classification

The incoming customer message is classified into one of the predefined support intents:

- Delivery & Tracking
- Returns & Refunds
- Account Access & Security
- Prime Membership & Benefits
- Promotions & Offers
- Digital Content & Device Issues
- Product Quality & Listing
- Payment & Billing
- Customer Service Experience
- Employment & Flex

### 2. Historical Support Retrieval

Historical AmazonHelp customer-support conversations form the knowledge base.

Customer messages are converted into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

FAISS is used for efficient semantic similarity search.

### 3. Grounded Response Generation

Retrieved historical Amazon responses are supplied to the LLM as context.

The response generator is instructed to:

- Use historical responses as the source of truth
- Avoid inventing policies or procedures
- Avoid inventing refunds or compensation
- Avoid unsupported claims
- Escalate when information is insufficient
- Avoid requesting sensitive information

### 4. Escalation Decision

The system chooses:

```text
AUTO-HANDLE
```

or

```text
ESCALATE
```

It considers account-specific investigation, financial/security concerns, unusual issues, available historical guidance, and the risk of incorrect automated advice.

When uncertain, the system defaults to escalation.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| LangChain | LLM and RAG orchestration |
| Groq | LLM inference |
| OpenAI GPT-OSS 120B | Classification, response generation and escalation |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Pandas | Dataset processing |
| NumPy | Embedding storage |
| Streamlit | Web interface |
| Git LFS | Large ML artifacts |

---

## 📊 Knowledge Base

The project uses customer-support conversations involving **AmazonHelp** from the Twitter Customer Support Dataset.

After extracting customer → Amazon response pairs, the knowledge base contains approximately:

```text
168,823 customer-response pairs
```

Each entry contains:

```text
customer_message
amazon_response
```

These historical responses provide the grounding information used by the response generator.

---

## 📁 Project Structure

```text
AI-support-agent/
│
├── app/
│   ├── app.py
│   └── rag.py
│
├── data/
│   ├── amazon_pairs.csv
│   └── twcs.csv
│
├── ETL and rag/
│   ├── embeddings.ipynb
│   ├── explore.ipynb
│   ├── label.ipynb
│   └── rag.ipynb
│
├── artifacts/
│   ├── amazon_embeddings.npy
│   └── amazon_intents.json
│
├── amazon_embedding_checkpoints/
│   ├── checkpoint_10000.pkl
│   ├── checkpoint_20000.pkl
│   ├── ...
│   └── checkpoint_168823.pkl
│
├── .env
├── .env.sample
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### Step 1 — Dataset Exploration

The original Twitter Customer Support Dataset contains fields such as:

```text
tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id
```

The dataset was analyzed to identify brands with sufficient customer-support interactions.

**AmazonHelp** was selected because it had a large number of support responses.

### Step 2 — Customer/Response Pair Extraction

The dataset was transformed into:

```text
Customer Message → Amazon Response
```

These pairs form the RAG knowledge base.

### Step 3 — Embedding Generation

Customer messages are converted into 384-dimensional embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are stored in:

```text
artifacts/amazon_embeddings.npy
```

### Step 4 — FAISS Search

FAISS indexes the embedding matrix.

For example:

```text
"My package hasn't arrived yet. Where is my order?"
```

is compared against historical customer messages to retrieve semantically similar cases.

### Step 5 — Intent Classification

The LLM maps the message to one predefined support intent.

Example:

```text
Customer:
My package says delivered but I never received it.

Intent:
Delivery & Tracking
```

### Step 6 — Response Generation

The top historical cases are supplied to the response generator, which creates a response grounded in those Amazon responses.

### Step 7 — Escalation

The system determines whether the issue can safely be automated or requires human support.

---

## 🔐 Safety and Grounding

A major design goal is to avoid unsupported automated responses.

The response-generation rules include:

```text
1. Historical responses are the source of truth.
2. Do not invent policies or procedures.
3. Do not invent refunds or compensation.
4. Do not fabricate delivery dates or claims.
5. Escalate when historical information is insufficient.
6. Do not request sensitive information publicly.
```

The escalation system follows a conservative strategy:

```text
Uncertain → ESCALATE
```

---

## 🧪 Example

### Input

```text
My package says delivered but I never received it.
```

### Intent

```text
Delivery & Tracking
```

### Retrieved Cases

The system retrieves similar AmazonHelp conversations involving:

- Missing packages
- Packages marked as delivered
- Tracking issues
- Delivery-status questions

### Generated Response

```text
I'm sorry you haven't received your parcel yet.
Please have a member of our support team look into this for you.
```

### Decision

```text
ESCALATE
```

### Reason

```text
The issue may require investigation of the specific order or delivery.
```

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/HARSHAN-git/AI-support-agent.git
cd AI-support-agent
```

Create a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

A template is provided in:

```text
.env.sample
```

Never commit your actual API keys to GitHub.

---

## ▶️ Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit interface will open in your browser.

---

## 🖥️ Application Workflow

The application allows you to:

1. Enter a customer-support message
2. Classify the issue
3. Retrieve similar historical cases
4. Generate a grounded response
5. View the escalation decision
6. Inspect retrieved historical cases

Example messages are also provided in the UI.

---

## 📈 Evaluation Strategy

### Intent Classification

Recommended metrics:

- Accuracy
- Precision
- Recall
- Macro-F1

Macro-F1 is useful when intent classes are imbalanced.

### Retrieval

Evaluate whether relevant historical conversations appear within:

```text
Recall@1
Recall@3
Recall@5
```

### Response Generation

Evaluate:

- Relevance
- Groundedness
- Helpfulness
- Factual consistency
- Unsupported claims

### Escalation

Evaluate:

- Escalation precision
- Escalation recall
- Unsafe auto-handle rate
- Incorrect auto-response rate

---

## ⚠️ Limitations

### Historical Data

Historical responses may contain outdated information, case-specific links, or incomplete context. Retrieved content should therefore not automatically be treated as universal policy.

### Account-Specific Issues

The system cannot independently access:

- Customer orders
- Account information
- Payment information
- Delivery systems
- Internal support tools

Such cases should generally be escalated.

### Language Coverage

The prototype focuses primarily on English-language AmazonHelp conversations.

### RAG Retrieval Quality

Semantic similarity does not guarantee identical resolutions. A similar historical case may still require a different action, which is why retrieval is combined with conservative response generation and escalation.

---

## 🔮 Future Improvements

- Hybrid keyword + vector retrieval
- Cross-encoder reranking
- Better conversation/thread reconstruction
- Multilingual support
- Fine-tuned intent classification
- Confidence scoring
- Retrieval-quality evaluation
- LLM-as-a-judge response evaluation
- Human feedback loops
- Conversation memory
- Production monitoring
- PII detection and redaction
- Better escalation policies
- Integration with real customer-support platforms

---

## 🎯 Key Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation
- Vector databases and semantic search
- FAISS
- Sentence Transformers
- LangChain
- LLM-based classification
- Prompt engineering
- Grounded generation
- AI safety and hallucination mitigation
- Automated escalation
- Customer-support automation
- Streamlit application development
- Large-scale embedding generation
- Git LFS for large ML artifacts

---

## 📌 Why RAG?

Instead of expecting an LLM to memorize support policies, the system retrieves relevant historical examples at runtime.

```text
Traditional LLM
     │
     ▼
Model Knowledge
     │
     ▼
Potential Hallucination
```

With RAG:

```text
Customer Message
     │
     ▼
Semantic Retrieval
     │
     ▼
Historical Amazon Cases
     │
     ▼
LLM
     │
     ▼
Grounded Response
```

This makes the system more suitable for customer-support scenarios where responses should be based on known examples rather than unsupported model knowledge.

---

## 📦 Large Files

The project contains large datasets and embedding artifacts.

Git LFS is used for:

```text
data/amazon_pairs.csv
data/twcs.csv
artifacts/amazon_embeddings.npy
amazon_embedding_checkpoints/*.pkl
```

This keeps large ML artifacts out of normal Git object storage.

---

## 👨‍💻 Author

**HARSHAN R**

GitHub:  
https://github.com/HARSHAN-git

---

## 📄 License

This project is licensed under the MIT License.
