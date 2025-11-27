import json
import re

# Enhancement: Skip the Table of Contents markdown cell automatically.
# The TOC cell starts with "# Table of Contents" after recent insertion.
# We detect and ignore any markdown cell whose first non-empty line matches that pattern.

def extract_markdown_content(notebook_path):
    """
    Extract markdown content excluding references, appendix, headings, and tables
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    content_parts = []
    skip_cell = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            cell_content = ''.join(cell['source']).strip()

            # Skip Table of Contents cell
            if cell_content:
                first_line = cell_content.split('\n', 1)[0].strip()
                if re.match(r'^#+\s*Table of Contents$', first_line, flags=re.IGNORECASE) or first_line.lower() == 'table of contents':
                    continue
            
            # Check if this cell starts a section to skip (References / Appendix)
            if (cell_content.startswith('# References') or 
                cell_content.startswith('#References') or
                'references:' in cell_content.lower() or
                cell_content.startswith('# Appendix')):
                skip_cell = True
                continue
            
            # If we hit a new major section, stop skipping
            if cell_content.startswith('#') and skip_cell:
                if not any(keyword in cell_content.lower() for keyword in ['references', 'appendix']):
                    skip_cell = False
            
            # Skip if we're in a section to skip
            if skip_cell:
                continue
            
            # Process the cell content
            lines = cell_content.split('\n')
            valid_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip headings (lines starting with #)
                if line.startswith('#'):
                    continue
                
                # Skip table rows (lines with multiple |)
                if '|' in line and line.count('|') >= 2:
                    continue
                
                # Skip table separator lines
                if re.match(r'^[-\s|:]+$', line):
                    continue
                
                # Skip figure captions and references
                if (line.lower().startswith('figure ') or 
                    re.match(r'^\(figure \d+\)', line.lower())):
                    continue
                
                # Skip standalone URLs
                if line.startswith('http'):
                    continue
                
                # Skip TODO comments
                if line.lower().startswith('#todo') or 'todo:' in line.lower():
                    continue
                
                # Keep the line but clean up markdown
                cleaned_line = line
                
                # Remove markdown links but keep link text
                cleaned_line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned_line)
                
                # Remove inline code but keep the text
                cleaned_line = re.sub(r'`([^`]+)`', r'\1', cleaned_line)
                
                # Remove bold/italic markers but keep text
                cleaned_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_line)
                cleaned_line = re.sub(r'\*([^*]+)\*', r'\1', cleaned_line)
                cleaned_line = re.sub(r'__([^_]+)__', r'\1', cleaned_line)
                cleaned_line = re.sub(r'_([^_]+)_', r'\1', cleaned_line)
                
                # Clean up extra whitespace
                cleaned_line = ' '.join(cleaned_line.split())
                
                if cleaned_line:
                    valid_lines.append(cleaned_line)
            
            if valid_lines:
                content_parts.extend(valid_lines)
    
    return content_parts

def count_words(content_lines):
    """Simple word counting"""
    all_text = ' '.join(content_lines)
    words = all_text.split()
    return len(words), all_text

# Extract content
notebook_path = '/Users/matthewbutler/Documents/TsunamiRiskModelConda/modelAndEDA3-refined9 rerun.ipynb'
content_lines = extract_markdown_content(notebook_path)
word_count, full_text = count_words(content_lines)

print(f"Word count (excluding headings, tables, references, appendix): {word_count}")
print(f"\nNumber of content lines extracted: {len(content_lines)}")
print(f"\nFirst few lines of content:")
for i, line in enumerate(content_lines[:10]):
    print(f"{i+1:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")

print(f"\nSample of full text (first 800 chars):")
print(full_text[:800] + "...")