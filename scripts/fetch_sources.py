"""Download the full text of each source page and save it as clean markdown.

Needs the `requests` package (listed in requirements.txt).
Usage:  python scripts/fetch_sources.py [output_dir]
Default output_dir is data/verbatim.
"""

import re
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests


GROUPS = {
    "Hypertension": "Older Adults | Blood, Heart and Circulation",
    "Type 2 Diabetes": "Endocrine System | Diabetes Mellitus",
    "Metabolic Syndrome": "Blood, Heart and Circulation | Endocrine System | Metabolic Problems | Diabetes Mellitus",
}

# (file name, topic, source name, url)
SOURCES = (
    ("hypertension_medlineplus", "Hypertension", "MedlinePlus", "https://medlineplus.gov/highbloodpressure.html"),
    ("hypertension_who", "Hypertension", "World Health Organization (WHO) fact sheet", "https://www.who.int/news-room/fact-sheets/detail/hypertension"),
    ("type2_diabetes_medlineplus", "Type 2 Diabetes", "MedlinePlus", "https://medlineplus.gov/diabetestype2.html"),
    ("type2_diabetes_who", "Type 2 Diabetes", "World Health Organization (WHO) fact sheet", "https://www.who.int/news-room/fact-sheets/detail/diabetes"),
    ("metabolic_syndrome_medlineplus", "Metabolic Syndrome", "MedlinePlus", "https://medlineplus.gov/metabolicsyndrome.html"),
)

SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"}
HEADING_TAGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}
BLOCK_TAGS = {"p", "li", "div", "br", "tr", "ul", "ol", "table", "section"}


class TextExtractor(HTMLParser):
    """Keeps headings and body text; drops navigation, scripts and footers."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0
        self.heading = None

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        elif not self.skip_depth:
            if tag in HEADING_TAGS:
                self.parts.append("\n\n" + HEADING_TAGS[tag])
                self.heading = tag
            elif tag == "li":
                self.parts.append("\n- ")
            elif tag == "p":
                self.parts.append("\n\n")
            elif tag in BLOCK_TAGS:
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag == self.heading:
            self.parts.append("\n")
            self.heading = None

    def handle_data(self, data):
        # Collapse source line breaks so only block tags start new lines
        if not self.skip_depth:
            self.parts.append(re.sub(r"\s+", " ", data))

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        out, blank = [], False
        for line in lines:
            if line:
                out.append(line)
                blank = False
            elif not blank:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def fetch(url):
    response = requests.get(
        url, headers={"User-Agent": "ki-medical-rag-course-project/1.0"}, timeout=30
    )
    response.raise_for_status()
    return response.text


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/verbatim")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, topic, source, url in SOURCES:
        try:
            parser = TextExtractor()
            parser.feed(fetch(url))
            body = parser.text()
        except Exception as error:  # keep going so one bad page doesn't stop the rest
            print(f"FAILED {name}: {error}")
            continue
        front = (
            f"---\ntopic: {topic}\ngroups: {GROUPS[topic]}\nsource: {source}\nurl: {url}\n"
            f"retrieved: {date.today().isoformat()}\ntext_type: verbatim\n---\n\n"
        )
        (out_dir / f"{name}.md").write_text(front + body + "\n", encoding="utf-8")
        print(f"saved {name}.md ({len(body)} characters)")
        time.sleep(1)  # being nice here


if __name__ == "__main__":
    main()
