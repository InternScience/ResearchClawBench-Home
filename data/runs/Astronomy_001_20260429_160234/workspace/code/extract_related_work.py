from pathlib import Path
from pypdf import PdfReader
for p in sorted(Path('related_work').glob('*.pdf')):
    r=PdfReader(str(p))
    text='\n'.join((page.extract_text() or '') for page in r.pages)
    out=Path('outputs')/(p.stem+'.txt')
    out.write_text(text)
    print(p, len(r.pages), len(text), out)
