import json
import re
from collections import defaultdict

def clean_markdown(text):
    # Remove fenced code blocks (```...```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code (`code`)
    text = re.sub(r'`[^`]+`', '', text)
    # Remove markdown tables (lines starting with | ... |)
    text = re.sub(r'^\|.*\|\s*$', '', text, flags=re.MULTILINE)
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove standalone URLs
    text = re.sub(r'http\S+', '', text)
    # Remove images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove markdown formatting characters (#, *, _, >, -, ~)
    text = re.sub(r'[>#*_~\-]+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load notebook ---
with open("modelAndEDA3-refined6.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

section_word_counts = defaultdict(int)
cell_details = []

current_section = "No Section"

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        # Detect section heading (first # heading line)
        heading_match = re.search(r'^\s*#+\s+(.*)', text, flags=re.MULTILINE)
        if heading_match:
            current_section = heading_match.group(1).strip()

        clean_text = clean_markdown(text)
        words = clean_text.split()
        word_count = len(words)
        section_word_counts[current_section] += word_count

        cell_details.append({
            "index": i,
            "section": current_section,
            "word_count": word_count,
            "preview": clean_text[:80] + ("..." if len(clean_text) > 80 else "")
        })

# --- Output summary ---
total_words = sum(section_word_counts.values())

print(f"\n📊 Total Markdown Words (URLs & Tables excluded): {total_words:,}\n")

print("📂 Section Breakdown:")
for section, count in section_word_counts.items():
    print(f"  - {section or 'No Section'}: {count:,} words")

print("\n🧩 Per-Cell Breakdown (first few shown):")
for d in cell_details[:10]:
    print(f"Cell {d['index']:>3} | Section: {d['section'][:25]:<25} | Words: {d['word_count']:>4} | {d['preview']}")
