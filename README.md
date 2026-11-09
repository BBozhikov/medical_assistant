# Local Medical RAG Assistant: PyQt5 GUI

A desktop chat app for a fully local medical RAG assistant. Each question is searched
in a local ChromaDB index of MedlinePlus health topics (plus a few full WHO/MedlinePlus
pages). The closest passages are given to a local llama.cpp model, which answers from
them only. Every answer lists the passages it used. If nothing in the index is close
enough, the assistant says so instead of guessing.

No network connections are made while the app runs.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [Get the data and the model](#3-get-the-data-and-the-model)
4. [Build the index](#4-build-the-index)
5. [Run the app](#5-run-the-app)
6. [Using the app](#6-using-the-app)
7. [Smoke test and retrieval evaluation](#7-smoke-test-and-retrieval-evaluation)
8. [Backend integration](#8-backend-integration)
9. [Thread lifecycle](#9-thread-lifecycle)

## 1. Prerequisites

- Python 3.10 or newer.
- On an Apple Silicon Mac: Xcode command line tools and CMake (needed to build
  llama-cpp-python with Metal, the GPU backend).
- On Windows: a C++ build toolchain may be needed if no prebuilt llama-cpp-python
  wheel is available for your Python version.
- About 5 GB of disk space for the GGUF model, plus about 80 MB for the embedding model.

The data, the model, and the index are not in git (they are gitignored), so each
machine builds or downloads its own copy.

## 2. Install

Create and activate a virtual environment in the project root, if you don't have one:

```sh
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
```

```powershell
python -m venv .venv                  # Windows
.\.venv\Scripts\Activate.ps1
```

Then install the packages in this order.

### Step A: llama-cpp-python (local LLM runtime)

`llama-cpp-python` is **not** installed by `requirements.txt`, so install it first - this is done because you can't install the metal built version on mac.
The pinned version is 0.3.36.

**Apple Silicon Mac (Metal GPU, recommended):**

```sh
CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/python -m pip install --no-cache-dir llama-cpp-python==0.3.36
```

Without Metal, answers run on the CPU and are much slower. Python 3.13 has no prebuilt
Metal wheel, so this step compiles from source.

**Windows or Linux (CPU):**

```powershell
.\.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.36
```

On Linux, use `.venv/bin/python` instead.

### Step B: the other packages

```sh
.venv/bin/python -m pip install -r requirements.txt        # macOS / Linux
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
```

This installs PyQt5, requests, and chromadb. Installing them after Step A leaves
llama-cpp-python unchanged.

## 3. Get the data and the model

### Model

Place a GGUF model at:

```
models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

You can also point the app at any other `.gguf` file in **Settings**.

### Source pages

```sh
python scripts/fetch_sources.py data/verbatim        # 5 full pages (downloads)
```

### MedlinePlus topics (manual download)

This step is not automated. Download the **MedlinePlus Health Topic XML** (the plain XML,
not the compressed `.zip` and not the "Health Topic Group" file). It is listed under the
newest "Files generated on" date at https://medlineplus.gov/xml.html. The file name follows
this pattern:

```
https://medlineplus.gov/xml/mplus_topics_YYYY-MM-DD.xml
```

For example, the file generated on November 7, 2026 is:

```sh
mkdir -p data/medlineplus
curl -L -o data/medlineplus/mplus_topics.xml https://medlineplus.gov/xml/mplus_topics_2026-11-07.xml
```

Save it as `data/medlineplus/mplus_topics.xml` (the name the build script reads). Its header
shows the generation date in `date-generated`. Write that date down, because MedlinePlus
replaces the files and older dated files are removed. A different release can give different
topics and different index results.

Then convert it to one markdown page per topic:

```sh
python scripts/build_medline_plus.py
```

## 4. Build the index

Build the index once, after the data is in place:

```sh
python scripts/build_index.py          # chunk, embed, and store in vector_db/
python scripts/evaluate_retrieval.py   # check retrieval on tests/eval_questions.json
```

The first `build_index.py` run downloads Chroma's default embedding model
(`all-MiniLM-L6-v2`, about 80 MB) into `~/.cache/chroma`.

## 5. Run the app

From the project root, with the virtual environment active:

```sh
python main.py                         # macOS / Linux
```

```powershell
& .\.venv\Scripts\python.exe main.py   # Windows
```

The first question is slower, because it loads the embedding model and the GGUF model.

If the app says the model file is missing, check Step 3 or pick the file in **Settings**.

## 6. Using the app

- Chat history, selectable plain-text messages, and a multiline question input.
- **Enter** sends; **Shift+Enter** inserts a newline.
- **Topics:** medical topics start collapsed with all 44 MedlinePlus topic groups
  selected (such as "Infections" or "Blood, Heart and Circulation"). Expand the scrolling
  list to narrow the next request's retrieval scope. Use "Select all" and "Clear" for quick
  changes. The heading shows the selection count. At least one topic must be selected.
- **Sources:** hidden until citations arrive. Then it shows the count, titles, locations,
  and excerpts for the latest answer. Clearing the chat hides it.
- **Settings:** choose the local GGUF file, temperature, maximum output tokens, and
  whether retrieved context is visible. Settings apply to the next request and last only
  for the session.
- **Export (save icon):** writes chat and citation excerpts to a local JSON file. Raw
  retrieved context is excluded. Exports can contain private health information, so store
  them carefully. Nothing is saved automatically.
- A medical disclaimer is always shown, and each answer includes it.

## 7. Smoke test and retrieval evaluation

**Smoke test:** checks that the GGUF model loads, uses the GPU if available, and generates
text. It does not use retrieval. It reports load time and tokens per second:

```sh
python scripts/smoke_test.py
python scripts/smoke_test.py --model path/to/other.gguf --question "..." --max-tokens 128
```

**Retrieval evaluation:** runs the questions in `tests/eval_questions.json` against the
index built in Step 4. For questions with `key_facts`, it passes if every key fact appears
in the top-k retrieved chunks. For questions with `expect_refusal`, it passes if the index
has nothing close enough. Other questions test the language model, so they are skipped:

```sh
python scripts/evaluate_retrieval.py
```

Neither script checks the quality of the final answers. Check that on your own machine
after the backend is connected.

## 8. Backend integration

The integration boundary is `src/rag/rag_pipeline.py`. `RAGPipeline` there is the real
implementation: index search in `src/rag/retrieval.py`, then llama.cpp. To use something
else, pass another object to `MainWindow(pipeline=...)`. The constructor must stay
lightweight. The worker is the only caller of the synchronous `answer` method:

```python
from src.rag.rag_pipeline import (
    GenerationSettings, RAGResponse, SourceCitation,
)


class LocalPipeline:
    def answer(
        self,
        question: str,
        *,
        topics: tuple[str, ...],
        settings: GenerationSettings,
    ) -> RAGResponse:
        # Perform local retrieval and generation here, not in the GUI thread.
        raise NotImplementedError("Connect the local backend here")
```

Return `RAGResponse(answer=answer_text, sources=tuple_of_citations,
context=retrieved_context)`. Each citation is
`SourceCitation(title=title, location=source_path_or_url, excerpt=chunk_text)`.
Use only retrieved documents, and return empty sources when none were found. Source URLs
are shown as text and are never fetched or opened automatically.

The real backend should:

1. Lazily load and cache the GGUF and embedding models inside `answer()`, including
   reloading when the model path changes. Configure Metal there (for example
   `n_gpu_layers=-1` on the M1 Pro).
2. Use the pre-downloaded embedding model with local-only loading, a persistent Chroma
   index, and telemetry disabled. Do not use remote LLM or embedding fallbacks.
3. Embed the question, filter retrieval by the selected topic metadata, and retrieve the
   top-k chunks. An empty topic tuple must match nothing, not everything.
4. Fill the provided `PROMPT_TEMPLATE` with the retrieved evidence. Keep the prompt plus
   output tokens within the model's context size. Say when evidence is insufficient, and
   link citations only to chunks actually used.
5. Apply `settings.temperature` and `settings.max_tokens`. Raise a descriptive exception
   for missing files or model-loading failures. The worker shows it in a warning dialog
   without a traceback.

The GUI sends only the current question. Chat history is for display and export, not
model memory. Resolve follow-up references in a future backend change if needed.

## 9. Thread lifecycle

`GenerationWorker(QThread)` runs the blocking pipeline and emits `response_ready` or
`failed`, followed by Qt's `finished` signal. Only main-thread slots update widgets. One
request runs at a time. Settings, topic filters, export, and clear are disabled while it
runs. The window holds a strong reference to the worker until it finishes, then schedules
its deletion.

Closing during a request is deferred: the window stays open until you close it again after
the request completes. The app never forcibly terminates a llama.cpp call. Backend
cancellation is intentionally not simulated, so a hung native call must be ended externally.
The backend must not require GUI-thread access.
