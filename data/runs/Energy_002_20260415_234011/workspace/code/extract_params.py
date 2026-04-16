import PyPDF2
import re
import json

def extract_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for p in reader.pages:
            text += p.extract_text() + '\n'
        return text

p0 = extract_text('related_work/paper_000.pdf')
p1 = extract_text('related_work/paper_001.pdf')

params = {}
# Extracting CAPEX and OPEX from paper_000
capex_matches = re.findall(r'([A-Za-z\s]+)\s+Capex\s+([\d,.]+)\s+€/MW', p0)
for match in capex_matches:
    params[match[0].strip()] = float(match[1].replace(',', ''))

print(json.dumps(params, indent=2))
