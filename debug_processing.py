import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(r"c:\Users\pc\uca_digital_assistant").resolve()))
os.environ["DJANGO_SETTINGS_MODULE"] = "core.settings"

from rag_module.offline.processing import preprocess_file, extract_text_docx, clean_text, _is_high_quality_document, safe_detect_lang, _text_metrics

file_path = r"c:\Users\pc\uca_digital_assistant\data_storage\raw\drive\Fiche Plateforme CIP Exemple.docx"

print(f"Testing {file_path}")

raw_text = extract_text_docx(file_path)
print(f"Raw text len: {len(raw_text)}")

cleaned_text = clean_text(raw_text)
print(f"Cleaned text len: {len(cleaned_text)}")
if not cleaned_text:
    print("Cleaned text is empty.")

metrics = _text_metrics(cleaned_text[: min(len(cleaned_text), 6000)])
print("Metrics:", metrics)

hq = _is_high_quality_document(cleaned_text, corpus="drive")
print(f"Is High Quality: {hq}")

lang, conf = safe_detect_lang(cleaned_text)
print(f"Language: {lang}, Confidence: {conf}")

res = preprocess_file(file_path, corpus="drive")
print(f"Resulting chunks: {len(res)}")
