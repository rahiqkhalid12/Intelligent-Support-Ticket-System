import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ============================================
# Load Environment Variables
# ============================================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(api_key=HF_TOKEN)

# ============================================
# Useless answer patterns to filter out
# These are dataset template placeholders that
# add no value as context for Qwen.
# ============================================
USELESS_PATTERNS = [
    "<tel_num>",
    "<acc_num>",
    "<email>",
    "<url>",
    "<name>",
    "<date>",
]

USELESS_PHRASES = [
    "contact us at",
    "please call us",
    "reach us at",
    "we will get back to you",
    "we will be in touch",
]

MIN_ANSWER_LENGTH = 80


def is_useful_answer(answer: str) -> bool:
    """Return True only if the answer has real content worth
    passing to Qwen as context."""
    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        return False
    answer_lower = answer.lower()
    # Contains unfilled placeholder tokens
    if any(p in answer for p in USELESS_PATTERNS):
        return False
    # Short generic phrases with no troubleshooting content
    for phrase in USELESS_PHRASES:
        if phrase in answer_lower and len(answer) < 150:
            return False
    return True


# ============================================
# Build Context
# Only include tickets with genuinely useful
# answers — if all retrieved answers are useless
# templates, Qwen will rely on its own knowledge
# rather than being misled by placeholder text.
# ============================================

def build_context(similar_tickets):

    if not similar_tickets:
        return "No similar historical tickets were retrieved."

    context = ""
    useful_count = 0

    for ticket in similar_tickets:
        answer = ticket.get("answer", "")

        if not is_useful_answer(answer):
            continue

        useful_count += 1
        context += f"""
==================================================
Historical Ticket {useful_count}

Subject:
{ticket.get("subject", "")}

Customer Problem:
{ticket.get("text", "")}

Previous Resolution:
{answer}

Queue:
{ticket.get("queue", "")}

Type:
{ticket.get("type", "")}

Priority:
{ticket.get("priority", "")}

Similarity Score:
{ticket.get("score", "")}

==================================================

"""

    if not context:
        return (
            "No useful historical resolutions were found for this issue. "
            "Use your own technical knowledge to provide the best answer."
        )

    return context


# ============================================
# Generate Response
# ============================================

def generate_response(ticket_text, prediction, similar_tickets):

    context = build_context(similar_tickets)

    company_facts = """
Headquarters: Cairo, Egypt
Countries of operation: Egypt, United Kingdom, United States
"""

    prompt = f"""
You are a senior IT customer support engineer.

Your job is to produce a professional customer support response.

----------------------------------------------------

COMPANY FACTS (use these ONLY if the customer's question matches them;
do not invent any company fact that is not listed here)

{company_facts}

----------------------------------------------------

NEW CUSTOMER TICKET

{ticket_text}

----------------------------------------------------

ML CLASSIFICATION

Queue:
{prediction["queue"]}

Type:
{prediction["type"]}

Priority:
{prediction["priority"]}

----------------------------------------------------

SIMILAR HISTORICAL TICKETS

{context}

----------------------------------------------------

Instructions

1. Carefully analyze the customer's issue.

2. Use the retrieved historical tickets as supporting knowledge
   ONLY if they contain real troubleshooting steps.

3. If previous tickets contain useful troubleshooting steps,
   adapt them to the customer's problem.

4. If no useful historical context was found, use your own
   technical knowledge to give the best possible answer.

5. Never simply copy the historical answer.

6. Never answer with only "We apologize..."

7. Give 2-4 numbered practical troubleshooting steps whenever possible.

8. If more information about the customer's TECHNICAL issue is required
   to troubleshoot it, ask ONLY for the minimum information needed —
   and only AFTER giving at least one troubleshooting step.

9. Be concise but complete.

10. Maximum 250 words.

11. Do not mention "historical tickets", "retrieved context",
    "RAG", "AI model", or "COMPANY FACTS section".

12. Write naturally as a human support engineer.

13. If the customer asks about a factual company detail that is NOT
    listed in the COMPANY FACTS section above (e.g. specific employee
    names, phone numbers, exact addresses, or dates), do NOT guess,
    use placeholder brackets, or ask the customer to clarify what they
    already clearly asked. Instead, acknowledge their specific question
    and say a team member will follow up with the exact details.

14. Do not include a sign-off, signature, or name at the end
    (no "Best regards," no "[Your Name]", no closing line at all).
    End the response after the last useful sentence.

Begin your response immediately.
"""

    try:

        response = client.chat.completions.create(

            model="Qwen/Qwen2.5-7B-Instruct",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced technical support engineer. "
                        "Always provide 2-4 numbered practical troubleshooting "
                        "steps before asking for more information. "
                        "If no historical context is useful, rely on your own "
                        "technical knowledge to give a complete answer."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=350,  # increased from 220 — gives Qwen room for full steps

        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        print("Qwen Error:", repr(e))

        return (
            "We are currently unable to generate an automated response. "
            "Please try again shortly."
        )


# ============================================
# Local Test
# ============================================

if __name__ == "__main__":

    prediction = {
        "queue": "Technical Support",
        "type": "Incident",
        "priority": "Medium"
    }

    similar = [
        {
            "subject": "Printer not printing",
            "text": "Printer detected but jobs remain in queue.",
            "answer": "Restart the Print Spooler service and reinstall the printer driver.",
            "queue": "Technical Support",
            "type": "Incident",
            "priority": "Medium",
            "score": 0.93
        }
    ]

    response = generate_response(
        "My printer stopped printing after a Windows update.",
        prediction,
        similar
    )

    print("\nGenerated Response:\n")
    print(response)