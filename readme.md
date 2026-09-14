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
                ┌─────────────────┐
                │ Intent Classifier│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Semantic Search │
                │      / RAG      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Response        │
                │ Generator       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Escalation      │
                │ Decision        │
                └────────┬────────┘
                         │
                  ┌──────┴──────┐
                  ▼             ▼
             AUTO-HANDLE     ESCALATE
