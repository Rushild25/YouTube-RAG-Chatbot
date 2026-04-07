from __future__ import annotations
import json
import requests
import streamlit as st

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("YouTube RAG Chatbot")
st.caption("Frontend for FastAPI backend ingest and ask flow")

def _normalize_base_url(url: str) -> str:
    val = (url or "").strip().rstrip("/")
    if not val:
        return "http://127.0.0.1:8000"
    return val

def api_health(base_url: str, timeout_s: int) -> tuple[bool, str]:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout_s)
        r.raise_for_status()
        payload = r.json()
        return True, json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return False, str(e)
    
def api_ingest(base_url: str, youtube_url: str, transcript_mode: str, timeout_s: int) -> tuple[bool, dict | str]:
    body = {
        "url":  youtube_url.strip(),
        "transcript_mode": transcript_mode
    }
    try:
        r = requests.post(f"{base_url}/ingest", json = body, timeout = timeout_s)
        if r.status_code >= 400:
            try:
                return False, r.json()
            except Exception:
                return False, r.text
            
        return True, r.json()
    except Exception as e:
        return False, str(e)
    
def api_ask(base_url: str, video_id: str, question: str, top_k: int, timeout_s: int) -> tuple[bool, dict | str]:
    body = {
        "video_id": video_id.strip(),
        "question": question.strip(),
        "top_k": int(top_k)
    }

    try:
        r = requests.post(f"{base_url}/ask", json=body, timeout=timeout_s)
        if r.status_code >= 400:
            try:
                return False, r.json()
            except Exception:
                return False, r.text
            
        return True, r.json()
    except Exception as e:
        return False, str(e)
    
with st.sidebar:
    st.subheader("Backend Settings")
    base_url_input = st.text_input("FastAPI Base URL", value = "http://127.0.0.1:8000")
    base_url = _normalize_base_url(base_url_input)
    timeout_s = st.number_input("Request Timeout (seconds)", min_value=5, max_value=600, value = 600, step=5)
    ask_top_k = st.number_input("Ask top_k", min_value=1, max_value=20, value = 5, step=1)

    if st.button("Health Check"):
        ok, info = api_health(base_url, int(timeout_s))
        if ok:
            st.success(f"Backend healthy: {info}")
        else:
            st.error(f"Health Check failed: {info}")

st.divider()

st.subheader("1. Ingest Video")
ingest_url = st.text_input("YouTube URL", value="", placeholder="https://www.youtube.com/watch?v=...")
transcript_mode = st.selectbox("Transcript Mode", options=["auto", "transcript-api", "groq-whisper"], index=0)

if st.button("Ingest"):
    if not ingest_url.strip():
        st.error("Please enter a valid YouTube URL")
    else:
        with st.spinner("Ingesting..."):
            ok, result = api_ingest(base_url, ingest_url, transcript_mode, int(timeout_s))

        if ok:
            st.success("Ingestion complete.")
            st.json(result)
            st.session_state["video_id"] = result.get("video_id", "")
            st.session_state["language"] = result.get("language", "")
            st.session_state["transcript_source"] = result.get("transcript_source", "")
        else:
            st.error("Ingestion failed.")
            st.write(result)


st.divider()

st.subheader("2. Ask Questions")
default_video_id = st.session_state.get("video_id", "")
ask_video_id = st.text_input("Video ID", value = default_video_id, placeholder="Auto filled after ingestion")
question = st.text_area("Question", value="", placeholder="Ask about the ingested video...")

if st.button("Ask"):
    if not ask_video_id.strip():
        st.error("Video ID is required. Ingest first or enter video ID manually.")
    elif not question.strip():
        st.error("Question field is empty.")
    else:
        with st.spinner("Generating answer..."):
            ok, result = api_ask(base_url, ask_video_id, question, int(ask_top_k), int(timeout_s))
        if ok:
            answer = result.get("answer", "")
            contexts = result.get("contexts", [])
            st.success("Answer generated")
            st.markdown("Answer")
            st.write(answer)

            st.markdown("Retrieved Contexts")
            st.write(f"Count: {len(contexts)}")
            for i, ctx in enumerate(contexts, start=1):
                with st.expander(f"Context {i}"):
                    st.write(ctx)
        else:
            st.error("Ask failed.")
            st.write(result)