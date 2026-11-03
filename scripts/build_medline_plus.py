"""Convert the MedlinePlus health-topics XML into one markdown page per topic.

Input:  data/medlineplus/mplus_topics.xml  (https://medlineplus.gov/xml.html)
Output: data/medlineplus/pages/<topic>.md  with the same front matter as the
        other pages, plus a `groups` line used for the GUI topic filter.

Usage:  python scripts/build_medlineplus.py
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from fetch_sources import TextExtractor

XML_PATH = Path("data/medlineplus/mplus_topics.xml")
OUT_DIR = Path("data/medlineplus/pages")


def clean_html(fragment):
    parser = TextExtractor()
    parser.feed(fragment)
    return parser.text()


def main():
    root = ET.parse(XML_PATH).getroot()
    generated = datetime.strptime(root.get("date-generated").split()[0], "%m/%d/%Y").date().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for topic in root:
        if topic.get("language") != "English":
            continue
        title, url = topic.get("title"), topic.get("url")
        body = clean_html(topic.findtext("full-summary") or "")
        if not body:
            continue
        groups = " | ".join(g.text for g in topic.findall("group"))
        also_called = ", ".join(a.text for a in topic.findall("also-called"))
        front = (
            f"---\ntopic: {title}\ngroups: {groups}\nsource: MedlinePlus (U.S. National Library of Medicine)\n"
            f"url: {url}\nretrieved: {generated}\ntext_type: verbatim topic summary\n---\n\n"
        )
        lines = [f"# {title}"]
        if also_called:
            lines.append(f"Also called: {also_called}")
        name = url.rsplit("/", 1)[-1].removesuffix(".html") or topic.get("id")
        (OUT_DIR / f"{name}.md").write_text(front + "\n\n".join(lines) + "\n\n" + body + "\n", encoding="utf-8")
        written += 1
    print(f"wrote {written} topic pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
