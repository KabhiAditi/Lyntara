import os
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
import pickle
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load .env and initialize Groq client
# ─────────────────────────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# Feature extraction
# Defined HERE in the module — never loaded from pkl.
# The pkl only stores (vectorizer, model).
# This avoids the AttributeError crash caused by
# sklearn version mismatches when unpickling functions.
# ─────────────────────────────────────────────
_COREF_RE = re.compile(
    r'\b(them|they|their|those|this|these|that|same|such|also|too|'
    r'he|she|his|her|it)\b',
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r'\b(also|too|as well|additionally|furthermore|moreover|now|then|next|'
    r'what about|how about|do the same|add|include|exclude|filter|sort|order|'
    r'for each of them|of them|of these|for them)\b',
    re.IGNORECASE,
)
_SC_RE = re.compile(
    r'\b(how many|how much|which|who is|who are|what is|what are|list|show all|'
    r'find all|give me all|top \d+|count of|number of|total|average|maximum|minimum|'
    r'in \d{4}|year \d{4}|from \d{4}|revenue|spending|spender|subscription|popular)\b',
    re.IGNORECASE,
)
_MODIFIER_START_RE = re.compile(
    r'^(only|sort|order|filter|add|include|exclude|also|and|but|just|now|then|next)\b',
    re.IGNORECASE,
)

def extract_features(queries):
    features = []
    for q in queries:
        words = q.split()
        q_lower = q.lower()
        features.append([
            len(words),
            len(q),
            len(re.findall(_COREF_RE, q)),
            len(re.findall(_FOLLOWUP_RE, q)),
            len(re.findall(_SC_RE, q)),
            int(bool(_MODIFIER_START_RE.match(q))),
            int(bool(re.search(r'\bin \d{4}\b', q_lower))),
            int(bool(re.search(r'\btop \d+\b', q_lower))),
            int(bool(re.search(r'\bhow (many|much)\b', q_lower))),
            int(bool(re.search(r'\bwh(o|at|ich|en|ere)\b', q_lower))),
            int(len(words) <= 4),
            int(len(words) >= 8),
            int(bool(re.search(_COREF_RE, q))),
            int(bool(re.search(_FOLLOWUP_RE, q))),
            int(bool(re.search(_SC_RE, q))),
        ])
    return np.array(features, dtype=float)


# ─────────────────────────────────────────────
# Load classifier
# pkl stores (vectorizer, model) — 2-tuple.
# Old 3-tuple pkl (with bundled function) also handled.
# ─────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "srd_mrd_classifier_v2.pkl")
with open(PKL_PATH, "rb") as f:
    _pkl = pickle.load(f)

if len(_pkl) == 3:
    vectorizer, model, _ = _pkl   # discard bundled function from old pkl
else:
    vectorizer, model = _pkl


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────
def predict_query_with_confidence(query: str):
    vec  = vectorizer.transform([query])
    hand = csr_matrix(extract_features([query]))
    x    = hstack([vec, hand])
    pred = model.predict(x)[0]
    try:
        proba      = model.predict_proba(x)[0]
        confidence = float(max(proba))
    except AttributeError:
        confidence = 1.0
    return pred, confidence


# ─────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────
conversation_history: list[dict] = []

def add_to_history(user_query: str, resolved_query: str) -> None:
    conversation_history.append({
        "user":     user_query,
        "resolved": resolved_query,
    })

def build_history_text(max_turns: int = 3) -> str:
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
# Groq combiner  (MRD only)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a query resolution assistant for a text-to-SQL system.
Your ONLY job is to replace unresolved references in a follow-up question so it
becomes a complete, self-contained query. Do NOT enrich, pad, or add context.

Rules you MUST follow:
1. Replace pronouns (it, they, their, those, that, same, he, she, his, her)
   with the actual entity from history.
2. Carry forward filters, time ranges, or groupings ONLY if the new question
   explicitly references them (e.g. "those from last year", "same filter").
3. If the new question adds a new filter, merge it with the prior context.
4. If the question is ALREADY self-contained (no pronouns, no implicit references),
   return it UNCHANGED -- word for word.
5. Do NOT invent, assume, or carry forward a time range / year if the new
   question does not mention time at all.
6. Do NOT add filters, topics, or data points that are not in the new question.
7. Do NOT change the intent -- if the question is about subscriptions, keep it
   about subscriptions even if prior history was about spenders.
8. If the question is a fragment with no subject (e.g. "Only last year"),
   resolve the missing subject from history. Do NOT fabricate new constraints.
9. If the user is making a correction ("I did not say X", "remove the X filter",
   "without X"), return the last resolved query with that element removed.
   Do not add anything new.
10. Output ONLY the rewritten query -- no explanation, no preamble, no SQL,
    no quotation marks around the output.

Examples:

History:
  Turn 1: User asked: top 5 spenders | Resolved to: top 5 spenders

Follow-up: "what are their names"
Output: names of the top 5 spenders

---

History:
  Turn 1: User asked: top 5 spenders of year 2023 | Resolved to: top 5 spenders of year 2023

Follow-up: "name of the top spender"
Output: name of the top spender
(Reason: no pronoun, no reference -- return unchanged.)

---

History:
  Turn 1: User asked: top 5 spenders of year 2023 | Resolved to: top 5 spenders of year 2023
  Turn 2: User asked: what are their names | Resolved to: names of the top 5 spenders of year 2023

Follow-up: "what is the most used subscription?"
Output: what is the most used subscription?
(Reason: self-contained question about a new topic -- return unchanged.)

---

History:
  Turn 1: User asked: top 5 spenders of year 2023 | Resolved to: top 5 spenders of year 2023
  Turn 2: User asked: their average spending | Resolved to: average spending of the top 5 spenders of year 2023

Follow-up: "i did not say in 2023"
Output: average spending of the top 5 spenders
(Reason: user is correcting -- remove the year from the last resolved query.)

---

History:
  Turn 1: User asked: list customers | Resolved to: list all customers

Follow-up: "also show their order count"
Output: list all customers along with their order count
""".strip()


def combine_mrd_query(
    current_query: str,
    confidence: float = 1.0,
    mrd_confidence_threshold: float = 0.65,
) -> str:
    """
    Resolve a follow-up query using conversation history via Groq.

    Guards (in order):
      G1 - No history yet        -> pass through (nothing to resolve)
      G2 - Low ML confidence     -> treat as SRD, pass through

    If ML says MRD, Groq is always called. Hallucination prevention is
    handled entirely by the system prompt rules, not by a regex gate here.
    """
    # G1: first turn -- nothing to resolve
    if not conversation_history:
        print("[Combiner] G1 - No history. Passing through.")
        return current_query

    # G2: low ML confidence -- safer to treat as SRD
    if confidence < mrd_confidence_threshold:
        print(f"[Combiner] G2 - Low confidence ({confidence:.2f}). Passing through.")
        return current_query

    history_text = build_history_text(max_turns=3)
    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f'Follow-up question: "{current_query}"\n\n'
        "Rewrite the follow-up question as a complete, standalone query."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=150,
    )

    resolved = response.choices[0].message.content.strip().strip('"').strip("'")
    print(f"[Combiner] '{current_query}'")
    print(f"        -> '{resolved}'")
    return resolved


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run_pipeline(query: str) -> str:
    # Hard rule: first-turn queries are always SRD.
    # Runs BEFORE ML -- no model call wasted.
    if not conversation_history:
        print("[Pipeline] First turn - forcing SRD regardless of ML.")
        resolved_query = query
        add_to_history(query, resolved_query)
        return resolved_query

    # Step 1: Classify
    pred, confidence = predict_query_with_confidence(query)
    is_mrd = str(pred).upper() in ("MRD", "1")
    print(f"\n[Classifier] Label: {'MRD' if is_mrd else 'SRD'} | Confidence: {confidence:.2f}")

    # Step 2: Combine if MRD, pass through if SRD
    if is_mrd:
        resolved_query = combine_mrd_query(query, confidence)
    else:
        resolved_query = query
        print(f"[SRD] Passing through: '{resolved_query}'")

    # Step 3: Save to history
    add_to_history(query, resolved_query)

    return resolved_query


# ─────────────────────────────────────────────
# Input loop
# ─────────────────────────────────────────────
# if __name__ == "__main__":
#     while True:
#         query = input("\nEnter your query (type 'exit' to stop): ").strip()
#         if not query:
#             continue
#         if query.lower() == "exit":
#             print("Stopped.")
#             break
#         resolved = run_pipeline(query)
#         print("Resolved query:", resolved)
