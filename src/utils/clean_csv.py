"""..."""
import os
import re
import time
import json
import unicodedata
import pandas as pd


# -------------------------------------------------------------------------------------------------
# Stupid small helpers.
# -------------------------------------------------------------------------------------------------

def load_json(val):
    """Helper: safely load JSON strings, returning an empty list on error."""
    try:
        return json.loads(val)
    except Exception:
        return []


def is_missing(field):
    """Helper for `analyze_progress`, checks if a field is missing."""
    if not isinstance(field, list) or len(field) == 0:
        return True

    if all(isinstance(x, str) and x.strip() == "" for x in field):
        return True

    return False

# -------------------------------------------------------------------------------------------------
# Actual utils.
# -------------------------------------------------------------------------------------------------

def process_name(name: str) -> str:
    """Process album or artist names for consistency while preserving foreign characters."""
    # 0. Lowercase the name.
    name = name.lower()

    # 0.5 Super specific edge case :)
    name = name.replace('℮', 'e')

    # 1. Normalize the string using NFKC (this still composes characters but doesn't force ASCII)
    name = unicodedata.normalize('NFKC', name)

    # 1.5 Second round of normalization.
    decomposed_name = unicodedata.normalize('NFD', name)
    filtered_name   = []
    for char in decomposed_name:
        if unicodedata.combining(char):
            continue
        filtered_name.append(char)
    name = unicodedata.normalize('NFC', "".join(filtered_name))

    # 2. Replace dollar signs with underscore "_"
    name = name.replace('$', '_')

    # 3. Remove anything inside parentheses or square brackets (including the brackets)
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\[.*?\]', '', name)

    # 4. Replace a period between two word characters with an underscore
    name = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', '_', name)

    # 4.5. Replace any remaining periods with a space
    name = re.sub(r'\.', ' ', name)

    # 5. Replace '&' with 'and'
    name = name.replace('&', 'and')

    # 6. Remove all symbols except word characters, whitespace, hyphens, and en-dashes
    name = re.sub(r"[^\w\s\-–]", "", name)

    # 7. Collapse whitespace around hyphens (ensuring no extra spaces around them)
    name = re.sub(r'\s*([-–])\s*', r'\1', name)

    # 8. Collapse multiple spaces into one and trim leading/trailing whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # 9. Replace all remaining spaces with hyphens
    name = name.replace(' ', '-')

    return name

