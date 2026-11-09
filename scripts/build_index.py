"""Chunk the source pages and store them in a local ChromaDB index.

Pages come from data/verbatim (hand-picked full pages) and data/medlineplus/pages
(every MedlinePlus health topic, made by build_medlineplus.py).
Uses Chroma's default embedding model (all-MiniLM-L6-v2). The index is written to
vector_db/, which is gitignored.

Each page lists its GUI topic groups in a `groups` line. Chroma metadata holds
single values only, so every group becomes its own true/false field (group_key).

Usage:
  python scripts/build_index.py                         # rebuild the index
  python scripts/build_index.py --query "question"      # test a search
  python scripts/build_index.py --query "question" --group "Diabetes Mellitus"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" imports work

from src.rag.retrieval import COLLECTION, group_key, open_client, open_collection, search as search_index

DATA_DIRS = (Path("data/verbatim"), Path("data/medlineplus/pages"))
MAX_CHARS = 600  # chunk size target; paragraphs are never split mid-sentence
MIN_CHARS = 150  # drop tiny fragments such as menu items


def read_page(path):
    """Return (metadata dict, body text) from a file with '---' front matter."""
    text = path.read_text(encoding="utf-8")
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def chunk_page(body):
    """Yield (section title, chunk text). Sections start at markdown headings."""
    section, buffer = "", []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            if buffer:  # never let a chunk span two sections
                yield section, "\n\n".join(buffer)
                buffer = []
            # A heading can arrive glued to its first paragraph by a single newline
            heading, _, block = block.partition("\n")
            section = heading.lstrip("#").strip()
            block = block.strip()
            if not block:
                continue
        buffer.append(block)
        if sum(len(b) for b in buffer) >= MAX_CHARS:
            yield section, "\n\n".join(buffer)
            buffer = []
    if buffer:
        yield section, "\n\n".join(buffer)


def build_chunks():
    ids, docs, metas = [], [], []
    for data_dir in DATA_DIRS:
        for path in sorted(data_dir.glob("*.md")):
            meta, body = read_page(path)
            groups = [g.strip() for g in meta.get("groups", "").split("|") if g.strip()]
            n = 0
            for section, text in chunk_page(body):
                if len(text) < MIN_CHARS:
                    continue
                # Start each chunk with its topic/section so it makes sense on its own
                title = meta["topic"] if section in ("", meta["topic"]) else f"{meta['topic']} - {section}"
                ids.append(f"{data_dir.name}-{path.stem}-{n}")
                docs.append(f"{title}\n{text}")
                chunk_meta = {"topic": meta["topic"], "source": meta["source"], "url": meta["url"], "section": section, "file": path.stem}
                chunk_meta.update({group_key(g): True for g in groups})
                metas.append(chunk_meta)
                n += 1
    return ids, docs, metas


def build_index():
    ids, docs, metas = build_chunks()
    client = open_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(COLLECTION)
    batch = 500  # keep each insert small
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.add(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
    print(f"indexed {len(ids)} chunks from {len({m['file'] for m in metas})} pages")


def search(question, groups=None, k=3):
    for chunk in search_index(open_collection(), question, groups, k):
        print(f"\n[{chunk.distance:.3f}] {chunk.topic} / {chunk.section}")
        print(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--group", action="append", help="topic group; repeat to allow several")
    args = parser.parse_args()
    if args.query:
        search(args.query, args.group)
    else:
        build_index()
