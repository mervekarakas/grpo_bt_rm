def score5_v1(post: str, summary: str) -> str:
    aspect = ("overall TL;DR quality (faithfulness/accuracy, focus on the main issue, "
              "clarity/conciseness, appropriate tone, and intent alignment/actionability)")
    return f"""Please act as an impartial judge and evaluate the {aspect} of the response below.

Input
[The start of the dialog]
POST:
{post}
[The end of the dialog]

[The start of the summary]
SUMMARY:
{summary}
[The end of the summary]

Write a brief analysis (2–4 sentences). Then assign the response a single overall score from 1.0 to 5.0,
using either an integer or a decimal with up to 0.1 precision. A higher score indicates a higher quality TL;DR.

Strict format:
- Output the score enclosed exactly in <s></s> on the LAST line, like: <s>3.2</s>
- Do NOT use <3.2> or any other brackets.
- Stop immediately after the score line.
"""

def score5_v2(post: str, summary: str) -> str:
    """Score-first, guided by aspect string (your first variant)."""
    aspect = (
        "Faithfulness/accuracy; focus on the main issue; clarity & conciseness; "
        "tone appropriateness; intent alignment/actionability."
    )
    return f"""You are an impartial judge scoring a TL;DR.

POST:
{post}

SUMMARY:
{summary}

Task:
1) Output the score FIRST on its own line exactly as: <s>x.y</s> where x.y is 1.0–5.0 (one decimal).
2) Then write a brief explanation (2–4 sentences) justifying the score based on the criteria: {aspect}.

Rules:
- No bullet points, no numbered lists.
- Do not output any other scores or additional <s> tags.
"""


def score5_v3(post: str, summary: str) -> str:
    """Score-first, no criteria (your second variant)."""
    return f"""You are an impartial judge scoring a TL;DR.

POST:
{post}

SUMMARY:
{summary}

Task:
1) Output the score FIRST on its own line exactly as: <s>x.y</s> where x.y is 1.0–5.0 (one decimal).
2) Then write a brief explanation (2–4 sentences) for why you chose that score.

Rules:
- No bullet points, no numbered lists.
- Do not output any other scores or additional <s> tags.
"""


def score5_v4(post: str, summary: str) -> str:
    """Score-first, explicit rubric scale (your third variant)."""
    return f"""You are an impartial judge scoring a TL;DR.

POST:
{post}

SUMMARY:
{summary}

Score criteria:
- Faithfulness/accuracy (no invented facts)
- Core issue focus
- Clarity & conciseness
- Tone appropriateness (non-sensational, no slurs)
- Intent alignment/actionability

Score scale (use the full range):
- 5.0: Excellent — faithful, captures the core issue, clear/concise, appropriate tone.
- 4.0: Good — mostly faithful and focused; minor omissions or slight wording issues.
- 3.0: Mixed — somewhat correct but misses key points or is vague/unclear.
- 2.0: Poor — major missing info, misleading emphasis, or significant confusion.
- 1.0: Very poor — largely inaccurate/invented or misses the point.

Task:
1) Output the score FIRST on its own line exactly as: <s>x.y</s> where x.y is 1.0–5.0 (one decimal).
2) Then write a brief justification (2–4 sentences) that references the criteria.

Rules:
- No bullet points, no numbered lists, no code blocks/backticks.
- Do not output any other scores or additional <s> tags.
"""
