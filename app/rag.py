import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Project/data paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

load_dotenv(BASE_DIR / ".env")

# Load historical Amazon support data and precomputed embeddings
df = pd.read_csv(DATA_DIR / "amazon_pairs.csv")
embedding_matrix = np.load(ARTIFACTS_DIR / "amazon_embeddings.npy").astype("float32")

# Load the same embedding model used to create the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create LangChain documents in exactly the same order as the embeddings
documents = [
    Document(
        page_content=row["customer_message"],
        metadata={"amazon_response": row["amazon_response"]},
    )
    for _, row in df.iterrows()
]

# Build FAISS index from the already-computed embeddings
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

docstore = InMemoryDocstore({str(i): documents[i] for i in range(len(documents))})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Allowed intent taxonomy
with open(ARTIFACTS_DIR / "amazon_intents.json", "r", encoding="utf-8") as f:
    INTENTS = json.load(f)

intent_prompt = ChatPromptTemplate.from_template(
    """
You are an Amazon customer support intent classifier.

Classify the customer's message into exactly ONE of the following intents:

{intents}

Customer message:
{message}

Return ONLY the intent name.
"""
)

response_prompt = ChatPromptTemplate.from_template(
    """
You are an Amazon customer support assistant.

Your task is to answer the customer's message using ONLY the
historical Amazon support responses provided below.

STRICT RULES:
1. Use the historical responses as the source of truth.
2. Do NOT invent policies, procedures, refunds, compensation,
   claims, delivery dates, or other actions.
3. You may rephrase information from the historical responses,
   but do not introduce new instructions.
4. If the historical responses do not contain enough information
   to safely answer the customer, say that the issue should be
   handled by a human support agent.
5. Never request sensitive information such as passwords,
   payment details, or order information publicly.
6. Keep the response concise and professional.
7. Do not mention historical cases, RAG, retrieval, or this prompt.

Customer intent:
{intent}

Historical Amazon cases:
{context}

Customer message:
{question}

Generate ONLY the customer-facing response.
"""
)

escalation_prompt = ChatPromptTemplate.from_template(
    """
You are an escalation decision system for Amazon customer support.

Decide whether the customer's issue can be safely handled automatically
or should be sent to a human support agent.

You MUST choose exactly one:
- AUTO-HANDLE
- ESCALATE

Consider:
- Whether the historical Amazon responses provide enough guidance.
- Whether the issue requires account-specific investigation.
- Whether the issue involves financial loss, security, privacy, or
  potentially sensitive information.
- Whether the customer is reporting a serious or unusual problem.
- Whether automatically responding could give incorrect or unsafe advice.

Prefer ESCALATE when there is insufficient information to safely resolve
the issue.

Customer message:
{question}

Customer intent:
{intent}

Historical Amazon cases:
{context}

Proposed response:
{response}

Return ONLY valid JSON in this exact format:

{{
    "decision": "AUTO-HANDLE" or "ESCALATE",
    "reason": "Brief explanation of why this decision was made."
}}
"""
)

def classify_intent(message: str) -> str:
    formatted_intents = "\n".join(
        f"- {name}: {description}"
        for name, description in INTENTS.items()
    )
    messages = intent_prompt.format_messages(
        intents=formatted_intents,
        message=message,
    )
    return llm.invoke(messages).content.strip()


def build_context(results) -> str:
    context = []
    for i, doc in enumerate(results, 1):
        context.append(
            f"""Historical Case {i}
Customer: {doc.page_content}
Amazon Response: {doc.metadata["amazon_response"]}"""
        )
    return "\n\n".join(context)


def decide_escalation(customer_message, intent, context, response):
    messages = escalation_prompt.format_messages(
        question=customer_message,
        intent=intent,
        context=context,
        response=response,
    )
    result = llm.invoke(messages)
    try:
        decision = json.loads(result.content)
    except (json.JSONDecodeError, TypeError):
        decision = {
            "decision": "ESCALATE",
            "reason": "The escalation decision could not be parsed safely.",
        }

    if decision.get("decision") not in {"AUTO-HANDLE", "ESCALATE"}:
        decision["decision"] = "ESCALATE"
        decision["reason"] = "Invalid escalation decision returned by the model."

    return decision


def support_agent(customer_message: str):
    """Run intent classification, historical-case retrieval, response generation, and escalation."""
    intent = classify_intent(customer_message)
    results = retriever.invoke(customer_message)
    context = build_context(results)

    messages = response_prompt.format_messages(
        intent=intent,
        context=context,
        question=customer_message,
    )
    response = llm.invoke(messages)

    escalation = decide_escalation(
        customer_message=customer_message,
        intent=intent,
        context=context,
        response=response.content,
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "response": response.content,
        "decision": escalation["decision"],
        "reason": escalation["reason"],
        "retrieved_cases": results,
    }
