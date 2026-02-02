def score100_v1(post: str, summary: str) -> str:
    aspect = (
        "Faithfulness/accuracy; focus on the main issue; clarity & conciseness; "
        "tone appropriateness; intent alignment/actionability."
    )
    return f"""You are an impartial judge scoring a TL;DR.

POST:
{post}

SUMMARY:
{summary}

Criteria: {aspect}

Score scale (0–100, use the full range):
- 90–100: excellent
- 70–89: good
- 40–69: mixed/partial
- 20–39: poor
- 0–19: very poor

Task:
1) Output the score FIRST on its own line exactly as: <s>NN</s> where NN is an integer 0–100.
2) Then write a brief explanation (2–4 sentences) justifying the score based on the criteria.

Rules:
- No bullet points, no numbered lists, no code blocks/backticks.
- Do not output any other scores or additional <s> tags.
"""
