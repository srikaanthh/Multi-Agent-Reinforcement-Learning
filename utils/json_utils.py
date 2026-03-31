from __future__ import annotations

import json
import re
from typing import Any


def extract_json_value(text: str) -> Any:
    """Parse a JSON object or array from free-form model output."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("Cannot parse JSON from empty text.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start == -1 or end == -1 or end <= start:
            continue
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Try fixing common issues: unescaped newlines in strings
        fixed = _fix_json_string(candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Unable to extract JSON from model output: {text[:200]!r}")


def _fix_json_string(text: str) -> str:
    """Attempt to fix common JSON formatting issues from LLM outputs."""
    # Replace literal newlines inside strings with escaped newlines
    # This regex finds strings and escapes newlines within them
    result = []
    in_string = False
    escape_next = False
    
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            continue
            
        if in_string and char == '\n':
            result.append('\\n')
            continue
            
        if in_string and char == '\r':
            result.append('\\r')
            continue
            
        if in_string and char == '\t':
            result.append('\\t')
            continue
            
        result.append(char)
    
    return ''.join(result)
