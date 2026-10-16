"""Apply small compatibility fixes to the original project notebook before execution.

This does not change the tracked source notebook; it creates a patched copy for
reproducible execution and preserves the original file.
"""
from pathlib import Path
import nbformat

source = Path("FINALPROJ.ipynb")
target = Path("FINALPROJ.compat.ipynb")
notebook = nbformat.read(source, as_version=4)
replacements = [
    (", multi_class='ovr'", ""),
    (", multi_class=\"ovr\"", ""),
    (", multi_class='auto'", ""),
    (", multi_class=\"auto\"", ""),
]
changed = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    original = cell.source
    updated = original
    for old, new in replacements:
        if old in updated:
            updated = updated.replace(old, new)
    if updated != original:
        cell.source = updated
        changed += 1

nbformat.validate(notebook)
nbformat.write(notebook, target)
print(f"Notebook compatibility patch: updated {changed} code cell(s); wrote {target}")
