import json

# Load the notebook
with open("modelAndEDA3-refined4.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract words from Markdown cells
word_count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        text = "".join(cell['source'])
        word_count += len(text.split())

print("Total words (Markdown only):", word_count)