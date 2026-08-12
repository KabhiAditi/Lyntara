import os
import numpy as np
from scipy.sparse import hstack
import pickle
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load .env and initialize Groq client
# ─────────────────────────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# Load classifier
# ─────────────────────────────────────────────
import os
import pickle

_dir = os.path.dirname(os.path.abspath(__file__))
_pkl_path = os.path.join(_dir, "srd_mrd_classifier_v2.pkl")
if not os.path.exists(_pkl_path):
    _pkl_path = "srd_mrd_classifier_v2.pkl"

with open(_pkl_path, "rb") as f:
    vectorizer, model = pickle.load(f)

def predict_query_with_confidence(query):
    vec = vectorizer.transform([query])
    length = np.array([len(query.split())]).reshape(-1, 1)
    final = hstack([vec, length])
    pred = model.predict(final)[0]

    try:
        proba = model.predict_proba(final)[0]
        confidence = max(proba)
    except AttributeError:
        confidence = 1.0

    return pred, confidence


# ─────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────
conversation_history = []

def add_to_history(user_query, resolved_query):
    conversation_history.append({
        "user": user_query,
        "resolved": resolved_query
    })

def build_history_text(max_turns=3):
    recent = conversation_history[-max_turns:]
    if not recent:
        return "No prior context."

    lines = []
    for i, turn in enumerate(recent, 1):
        lines.append(f"Turn {i}:")
        lines.append(f"  User asked   : {turn['user']}")
        lines.append(f"  Resolved to  : {turn['resolved']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MRD combiner
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a query resolution assistant for a text-to-SQL system.
Your job: rewrite a follow-up question into a complete, self-contained query
by resolving all references using the conversation history provided.

Rules:
- Replace all pronouns (it, they, their, those, that, same) with the actual entity.
- Carry forward any filters, time ranges, or groupings from prior turns if still relevant.
- If the new question adds a new filter, merge it with the prior context.
- Output ONLY the rewritten query — no explanation, no preamble, no SQL.
- Keep it in plain natural language.
"""

def combine_mrd_query(current_query, confidence=1.0, mrd_confidence_threshold=0.65):
    if confidence < mrd_confidence_threshold:
        print(f"[Combiner] Low confidence ({confidence:.2f}) — treating as SRD.")
        return current_query

    if not conversation_history:
        print("[Combiner] No history yet — passing query through as-is.")
        return current_query

    history_text = build_history_text(max_turns=3)

    prompt = f"""Conversation history:
{history_text}

Follow-up question: "{current_query}"

Rewrite the follow-up question as a complete, standalone query."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=150,
    )

    resolved = response.choices[0].message.content.strip()
    print(f"[Combiner] '{current_query}'")
    print(f"        → '{resolved}'")
    return resolved


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run_pipeline(query):
    # Step 1: Classify
    pred, confidence = predict_query_with_confidence(query)
    is_mrd = str(pred).upper() in ("MRD", "1")
    print(f"\n[Classifier] Label: {pred} | Confidence: {confidence:.2f}")

    # Step 2: Combine if MRD, pass through if SRD
    if is_mrd:
        resolved_query = combine_mrd_query(query, confidence)
    else:
        resolved_query = query
        print(f"[SRD] Passing through: '{resolved_query}'")

    # Step 3: Save to history
    add_to_history(query, resolved_query)

    # Step 4: Schema retrieval + SQL generation will come here next
    return resolved_query


# ─────────────────────────────────────────────
# Input loop
# ─────────────────────────────────────────────
# while True:
#     query = input("\nEnter your query (type 'exit' to stop): ")

#     if query.lower() == "exit":
#         print("Stopped.")
#         break

#     resolved = run_pipeline(query)
#     print("👉 Resolved query:", resolved)