import os
from groq import Groq


def explain_error(error_text, error_type):

    api_key = os.getenv("LLM_API")

    if api_key:
        try:
            client = Groq(api_key=api_key)

            prompt = f"""
                Analyze the programming error below.

                Error message:
                {error_text}

                Error category:
                {error_type}

                Provide a short and clear explanation that helps quickly understand the issue.

                Include:
                • what the error means in simple terms
                • If a line number is present in the error message, mention the line number and indicate that the issue likely originates near that line.
                • the main technical reason it occurs
                • the programming concept involved (type mismatch, null reference, syntax rule, scope issue, index bounds, import issue, etc.)
                • what to check in the code to fix it

                Guidelines:
                • keep the explanation concise (3–5 sentences)
                • use clear technical terms but simple language
                • help the user quickly locate the mistake
                • do NOT provide corrected code
                • do NOT output code blocks
                • do NOT mention AI or model names


"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            explanation = response.choices[0].message.content

            if explanation:
                return explanation.strip()

        except Exception as e:
            print("LLM error:", e)

    return _fallback(error_text, error_type)



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
            "# python example\n"
            "msg = f'count: {num}'\n\n"
            "// js example\n"
            "if (obj != null) { console.log(obj.name); }\n"
            "```"
        ),

        "Index Error": (
            "You accessed a position that doesn't exist in an array, list, "
            "or string. Most languages use zero-based indexing, "
            "so an array of size 5 has valid indices 0–4.\n\n"
            "Check length before accessing elements and avoid off-by-one errors.\n\n"
            "```\n"
            "if (i < arr.length) { value = arr[i]; }\n"
            "```"
        ),

        "Name Error": (
            "The variable, function, or class name is not recognized. "
            "Usually caused by typos, missing declarations, or scope issues.\n\n"
            "Ensure the name is defined before use and imports are correct.\n\n"
            "```\n"
            "import pandas as pd\n"
            "```"
        ),

        "Import Error": (
            "The required package or module is not available in the environment. "
            "It may not be installed or the name may be incorrect.\n\n"
            "Install using the appropriate package manager.\n\n"
            "```\n"
            "pip install package_name\n"
            "npm install package_name\n"
            "```"
        ),

        "Runtime Error": (
            "The program failed during execution due to invalid operations such as "
            "null references, division by zero, invalid memory access, or logic issues.\n\n"
            "Validate values before performing operations and handle exceptions properly.\n\n"
            "```\n"
            "if (obj != null) { obj.method(); }\n"
            "```"
        ),
    }

    return tips.get(
        error_type,
        f"This is classified as a {error_type}. Review the error message and check the related code."
    )