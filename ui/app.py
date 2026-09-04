import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("AI Assistant")
st.markdown(
    "Welcome to the AI Assistant. Ask questions, compute arithmetic expressions, "
    "or query knowledge base documents through the unified backend."
)

with st.sidebar:
    st.header("Configuration")
    st.info(f"**Connected Backend:**\n`{BACKEND_URL}`")
    st.markdown("---")
    st.markdown(
        "**Features:**\n"
        "- Document Q&A (RAG)\n"
        "- Math Calculator Tool\n"
        "- Automatic Topic Categorization\n"
        "- Automatic Transient Retries\n"
        "- In-Memory Rate Limiting\n"
        "- Automated LLM Fallback\n"
        "- In-Memory Response Caching"   )

# Input form
with st.form("ask_form", clear_on_submit=False):
    question = st.text_area(
        label="Your Question:",
        placeholder="e.g. What is binary search? or What is 25 multiplied by 18?",
        height=100,
    )
    submitted = st.form_submit_button("Submit Question", type="primary", use_container_width=True)

if submitted:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Please enter a non-empty question before submitting.")
    else:
        with st.spinner("Processing your request with AI Assistant..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": cleaned_question},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    topic = data.get("topic", "General")

                    st.markdown("### Response")
                    st.write(answer)
                    st.markdown(f"**Detected Topic:** `{topic}`")

                elif response.status_code == 429:
                    try:
                        detail = response.json().get("detail", "Rate limit exceeded. Please try again later.")
                    except Exception:
                        detail = "Rate limit exceeded. Please try again later."
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        st.warning(f" **Rate Limit Exceeded:** {detail} (Please retry after {retry_after}s)")
                    else:
                        st.warning(f" **Rate Limit Exceeded:** {detail}")

                elif response.status_code == 503:
                    try:
                        detail = response.json().get("detail", "The AI service is temporarily unavailable.")
                    except Exception:
                        detail = "The AI service is temporarily unavailable."
                    st.error(f" **Service Temporarily Unavailable:** {detail}")

                elif response.status_code in (400, 422):
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    st.error(f" **Invalid Request:** {detail}")

                else:
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    st.error(f" **Backend Error (HTTP {response.status_code}):** {detail}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"🔌 **Backend Unavailable:** Could not connect to the backend server at `{BACKEND_URL}`. "
                    "Please verify that the FastAPI server is running."
                )
            except requests.exceptions.Timeout:
                st.error("⏳ **Request Timeout:** The backend took too long to respond. Please try again.")
            except Exception as exc:
                st.error(f"**Unexpected Error:** {str(exc)}")
