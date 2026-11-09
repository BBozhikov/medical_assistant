"""Score retrieval: does the top-k search return the text that holds each key fact?

Questions come from tests/eval_questions.json.
- Questions with key_facts: pass if every key fact is in the top-k chunks.
- Questions with expect_refusal: pass if even the nearest chunk is farther than
  MAX_DISTANCE, so the pipeline answers "no information found" without the model.
- Other questions (no key_facts) test the language model, not retrieval: skipped.
Searches use every topic group, which is what the GUI does by default.

Usage:  python scripts/eval_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" imports work

from src.rag.rag_pipeline import MAX_DISTANCE
from src.rag.retrieval import open_collection, search

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "tests" / "eval_questions.json"
KS = (3, 5, 10)


def main():
    collection = open_collection()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    hits = {k: 0 for k in KS}
    answerable = 0
    refusal_total = refusal_ok = 0
    for q in questions:
        chunks = search(collection, q["question"], k=max(KS))
        nearest = chunks[0].distance
        if q.get("expect_refusal"):
            refusal_total += 1
            ok = nearest > MAX_DISTANCE
            refusal_ok += ok
            print(f"{'PASS' if ok else 'MISS'}  {q['id']:10s} nearest {nearest:.2f} (refuse above {MAX_DISTANCE})")
            continue
        if not q["key_facts"] or q.get("note"):
            continue
        answerable += 1
        docs = [c.text.lower() for c in chunks]
        # smallest k where every key fact appears somewhere in the top-k chunks
        found_at = next((k for k in KS if all(f.lower() in " ".join(docs[:k]) for f in q["key_facts"])), None)
        for k in KS:
            hits[k] += found_at is not None and found_at <= k
        cutoff_ok = nearest <= MAX_DISTANCE  # a good question must not be refused
        print(
            f"{'PASS' if found_at == KS[0] else 'MISS'}  {q['id']:10s} found at top-{found_at}, "
            f"nearest {nearest:.2f}{'' if cutoff_ok else '  <-- would be refused by the cutoff'}"
        )
    print()
    for k in KS:
        print(f"top-{k}: {hits[k]}/{answerable}")
    print(f"refused correctly: {refusal_ok}/{refusal_total}")


if __name__ == "__main__":
    main()
