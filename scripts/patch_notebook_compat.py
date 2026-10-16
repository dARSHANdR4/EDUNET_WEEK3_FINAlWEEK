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
    ("solver='liblinear'", "solver='lbfgs'"),
    ('solver="liblinear"', 'solver="lbfgs"'),
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

# Add smoke tests for one real held-out crop row and fertilizer row after models are trained.
smoke_test = r"""
# Automated E2E smoke tests: held-out rows -> fitted preprocessing/model -> known label.
if "best_model_crop" not in globals() or "best_model_fert" not in globals():
    raise RuntimeError("Expected crop/fertilizer models were not trained.")

crop_sample = X_test_crop.iloc[[0]]
crop_expected = str(y_test_crop.iloc[0] if hasattr(y_test_crop, "iloc") else y_test_crop[0])
crop_pred_encoded = int(best_model_crop.predict(scaler_crop.transform(crop_sample))[0])
crop_pred = str(label_encoder_crop.inverse_transform([crop_pred_encoded])[0])
assert crop_pred in set(label_encoder_crop.classes_), f"Unknown crop prediction: {crop_pred}"
print(f"E2E crop smoke test: PASS (predicted={crop_pred}; held-out actual={label_encoder_crop.inverse_transform([int(crop_expected)])[0]})")

fert_sample = X_test_fert.iloc[[0]]
fert_pred_encoded = int(best_model_fert.predict(preprocessor_fert.transform(fert_sample))[0])
fert_pred = str(label_encoder_fert.inverse_transform([fert_pred_encoded])[0])
assert fert_pred in set(label_encoder_fert.classes_), f"Unknown fertilizer prediction: {fert_pred}"
assert fert_sample.notna().all().all(), "Fertilizer smoke-test sample unexpectedly contains missing values"
print(f"E2E fertilizer smoke test: PASS (predicted={fert_pred}; held-out actual={label_encoder_fert.inverse_transform([int(y_test_fert.iloc[0] if hasattr(y_test_fert, 'iloc') else y_test_fert[0])])[0]})")
"""
notebook.cells.append(nbformat.v4.new_code_cell(smoke_test))

# Keep compatibility with the source notebook's older nbformat schema (pre-4.5),
# which does not permit cell IDs. GitHub's current nbformat creates IDs by default.
if notebook.nbformat_minor < 5:
    for cell in notebook.cells:
        cell.pop("id", None)

nbformat.validate(notebook)
nbformat.write(notebook, target)
print(f"Notebook compatibility patch: updated {changed} code cell(s); wrote {target}")
