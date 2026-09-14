import streamlit as st
from rag import support_agent


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Amazon Support AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 16px;
            border: 1px solid rgba(128,128,128,.25);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0 0 .35rem 0;
        }

        .hero p {
            margin: 0;
            opacity: .75;
            font-size: 1.05rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,.25);
            min-height: 100px;
        }

        .decision-auto {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border: 1px solid #2e8b57;
            background: rgba(46,139,87,.10);
        }

        .decision-escalate {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border: 1px solid #d9534f;
            background: rgba(217,83,79,.10);
        }

        .small-label {
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            opacity: .65;
            margin-bottom: .25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📦 Amazon Support AI</h1>
        <p>
            AI customer support powered by intent classification,
            historical-case RAG, and escalation detection.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("About the Agent")

    st.write(
        "This prototype analyzes a customer message, classifies "
        "its support intent, retrieves similar historical Amazon "
        "support cases, generates a grounded response, and decides "
        "whether the case should be auto-handled or escalated."
    )

    st.divider()

    st.markdown("**Pipeline**")
    st.markdown(
        """
        1. 🎯 Intent classification
        2. 🔎 FAISS similarity search
        3. 💬 Grounded response generation
        4. 🛡️ Escalation decision
        """
    )

    st.divider()

    st.caption("Knowledge base: 168,823 historical Amazon cases")
    st.caption("Embedding model: all-MiniLM-L6-v2")
    st.caption("LLM: GPT OSS 120B via Groq")


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------
st.subheader("Customer Message")

example_messages = {
    "Select an example": "",
    "📦 Missing package": "My package says delivered but I never received it.",
    "💰 Unknown charge": "I was charged for something I didn't buy.",
    "🔐 Account hacked": "Someone hacked my Amazon account and changed my payment details.",
    "↩️ Return request": "I want to return an item I bought yesterday.",
    "📱 Device problem": "My Kindle is not turning on.",
}

selected_example = st.selectbox(
    "Try an example",
    list(example_messages.keys()),
)

default_text = example_messages[selected_example]

customer_message = st.text_area(
    "Enter the customer's message",
    value=default_text,
    height=130,
    placeholder="Example: My package says delivered but I never received it.",
)

analyze = st.button(
    "🤖 Analyze Customer Message",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Run agent
# ---------------------------------------------------------
if analyze:
    if not customer_message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Running support agent..."):
        try:
            result = support_agent(customer_message.strip())
        except Exception as exc:
            st.error("The support agent encountered an error.")
            st.exception(exc)
            st.stop()

    st.session_state["result"] = result


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()

    # Top-level summary
    st.subheader("Agent Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Detected Intent</div>
                <strong>{result["intent"]}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        decision = result["decision"]

        if decision == "AUTO-HANDLE":
            st.markdown(
                """
                <div class="decision-auto">
                    <div class="small-label">Support Decision</div>
                    <strong>✅ AUTO-HANDLE</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="decision-escalate">
                    <div class="small-label">Support Decision</div>
                    <strong>🚨 ESCALATE TO HUMAN</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Decision reason
    st.subheader("Why this decision?")

    if result["decision"] == "AUTO-HANDLE":
        st.success(result["reason"])
    else:
        st.error(result["reason"])

    # Generated response
    st.subheader("💬 Suggested Customer Response")

    st.chat_message("assistant").write(result["response"])

    # Retrieved evidence
    st.subheader("🔎 Historical Evidence")

    st.caption(
        "These are the top historical Amazon cases retrieved by "
        "the RAG system and used as context for response generation."
    )

    for i, doc in enumerate(result["retrieved_cases"], start=1):
        with st.expander(f"Historical Case {i}"):
            st.markdown("**Customer message**")
            st.write(doc.page_content)

            st.markdown("**Historical Amazon response**")
            st.write(doc.metadata.get("amazon_response", ""))

    # Raw result for debugging
    with st.expander("Developer / Debug Information"):
        st.json(
            {
                "customer_message": result["customer_message"],
                "intent": result["intent"],
                "decision": result["decision"],
                "reason": result["reason"],
                "retrieved_case_count": len(result["retrieved_cases"]),
            }
        )
else:
    st.info(
        "Enter a customer message above and click "
        "**Analyze Customer Message** to run the agent."
    )
