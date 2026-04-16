import os
import sys

# Look for fuxi model code or weights
for root, dirs, files in os.walk('.'):
    for f in files:
        if 'fuxi' in f.lower() or 'model' in f.lower():
            print(os.path.join(root, f))
