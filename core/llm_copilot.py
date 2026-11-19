import json
import os
import re
import socket
import requests
import urllib3.util.connection as urllib3_cn
from dotenv import load_dotenv

load_dotenv()
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

def _call_ollama(prompt: str, max_tokens: int = 280) -> str:
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate").replace("localhost", "127.0.0.1")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    r = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        },
        stream=True,
        timeout=(5, 180),
    )
    r.raise_for_status()
    chunks = []
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        chunks.append(data.get("response", ""))
        if data.get("done"):
            break
    return "".join(chunks)

def warm_ollama() -> bool:
    try:
        _call_ollama("Reply OK.", max_tokens=5)
        return True
    except Exception:
        return False

def _ask_llm(prompt: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_local = os.getenv("USE_LOCAL_ONLY", "false").lower() == "true"
    if gemini_key and not use_local:
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
        try:
            r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=(5, 12))
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as err:
            print(f"[BIC WARN] Gemini skipped ({err}). Using local Ollama...")
    return _call_ollama(prompt)

def _extract_sql(raw: str) -> str:
    m = re.search(r"```sql\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else raw).strip().strip("`")

def generate_sql(question: str, schema: str, history: list[dict] | None = None) -> str:
    ctx = ""
    if history:
        recent = history[-3:]
        ctx = "Previous conversation:\n" + "\n".join(
            f"- Q: {h['question']} | SQL: {h.get('sql', '')}" for h in recent
        ) + "\n\n"
    prompt = (
        f"You are a DuckDB SQL expert. Available schema:\n{schema}\n\n{ctx}"
        f"Write ONE valid DuckDB SELECT query to answer: {question}\n"
        "Return ONLY the SQL query inside ```sql ... ``` with no explanation."
    )
    return _extract_sql(_ask_llm(prompt))

def repair_sql(question: str, schema: str, bad_sql: str, error_msg: str) -> str:
    prompt = (
        f"Fix this DuckDB SQL query.\nSchema:\n{schema}\n"
        f"Question: {question}\nBroken SQL: {bad_sql}\nError: {error_msg}\n"
        "Return ONLY the corrected SQL inside ```sql ... ```."
    )
    return _extract_sql(_ask_llm(prompt))

def summarize_insight(question: str, df_preview: str) -> str:
    prompt = (
        f"User question: {question}\nSQL query result:\n{df_preview}\n\n"
        "Give a concise 2-3 sentence business insight with exact numbers from the result."
    )
    return _ask_llm(prompt).strip()