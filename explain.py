import os
import requests

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"


def explain_error(error_text, error_type):
    token = os.environ.get("HF_API_TOKEN", "")
    if token:
        result = _call_api(error_text, error_type, token)
        if result:
            return result
    return _fallback(error_text, error_type)


def _call_api(error_text, error_type, token):
    prompt = (
        f"<s>[INST] Explain this programming error briefly and clearly.\n\n"
        f"Error: {error_text}\n"
        f"Type: {error_type}\n\n"
        f"Describe what the error means, why it happens, how to fix it, "
        f"and give a short example fix. Keep it practical. [/INST]"
    )
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 400, "temperature": 0.4, "return_full_text": False}
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
    except Exception:
        pass
    return None


def _fallback(error_text, error_type):
    tips = {
        "Syntax Error": (
            "The compiler or interpreter can't parse your code because something "
            "violates the language grammar. Common causes: missing semicolons, "
            "unmatched brackets, wrong keywords, or bad indentation.\n\n"
            "Check the line number in the error and the line before it. "
            "Look for missing colons (Python), semicolons (C++/Java/JS), "
            "or mismatched braces.\n\n"
            "```\n"
            "// example fix (javascript)\n"
            "// before: if (x == 5 { console.log(x) }\n"
            "// after:  if (x == 5) { console.log(x); }\n"
            "```"
        ),
        "Type Error": (
            "You tried an operation on a type that doesn't support it, "
            "or mixed incompatible types. This happens across most languages "
            "when you combine strings with numbers, call non-functions, "
            "or pass wrong argument types.\n\n"
            "Check variable types and convert explicitly when needed.\n\n"
            "```\n"
            "# python: use str() or f-strings\n"
            "msg = f'count: {num}'\n\n"
            "// js: check for null/undefined before accessing properties\n"
            "if (obj != null) { console.log(obj.name); }\n"
            "```"
        ),
        "Index Error": (
            "You accessed a position that doesn't exist in an array, list, "
            "or string. Remember most languages use zero-based indexing, "
            "so an array of 5 elements has valid indices 0 through 4.\n\n"
            "Check collection length before accessing elements. "
            "Watch out for off-by-one errors in loops.\n\n"
            "```\n"
            "// check bounds first\n"
            "if (i < arr.length) { value = arr[i]; }\n"
            "```"
        ),
        "Name Error": (
            "The name you used (variable, function, class) isn't recognized. "
            "Usually a typo, missing import, or scope issue. In Python it's "
            "NameError, in JS it's ReferenceError, in Java it's 'cannot find symbol'.\n\n"
            "Double-check spelling, make sure imports are at the top, and verify "
            "the variable exists in the current scope.\n\n"
            "```\n"
            "# make sure you import what you use\n"
            "import pandas as pd  # then pd.read_csv() works\n"
            "```"
        ),
        "Import Error": (
            "The module or package you're trying to use can't be found. "
            "Either it's not installed, there's a name mismatch, or you're "
            "in the wrong environment.\n\n"
            "Install the package with your language's package manager:\n\n"
            "```\n"
            "pip install pandas          # python\n"
            "npm install express          # node.js\n"
            "go get github.com/pkg/...   # golang\n"
            "```\n\n"
            "Some packages have different install and import names "
            "(scikit-learn -> sklearn, Pillow -> PIL)."
        ),
        "Runtime Error": (
            "Something went wrong during execution. This covers division by zero, "
            "null pointer dereferences, stack overflows, segmentation faults, "
            "and other issues that crash the program at runtime.\n\n"
            "Common fixes: check for null/nil/None before use, validate "
            "denominators before division, add base cases to recursive functions, "
            "and handle file/network errors with try-catch.\n\n"
            "```\n"
            "// check for null before use\n"
            "if (ptr != null) { ptr.doSomething(); }\n\n"
            "# check denominator\n"
            "avg = total / count if count != 0 else 0\n"
            "```"
        ),
    }
    return tips.get(error_type,
        f"This is classified as a {error_type}. "
        "Check the error message for the line number and look at that section of code."
    )
