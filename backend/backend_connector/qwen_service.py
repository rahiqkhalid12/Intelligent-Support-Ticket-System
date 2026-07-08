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
# Build Context
# ============================================

def build_context(similar_tickets):

    if not similar_tickets:
        return "No similar historical tickets were retrieved."

    context = ""

    for i, ticket in enumerate(similar_tickets, start=1):

        context += f"""
==================================================
Historical Ticket {i}

Subject:
{ticket.get("subject","")}

Customer Problem:
{ticket.get("text","")}

Previous Resolution:
{ticket.get("answer","")}

Queue:
{ticket.get("queue","")}

Type:
{ticket.get("type","")}

Priority:
{ticket.get("priority","")}

Similarity Score:
{ticket.get("score","")}

==================================================

"""

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

2. Use the retrieved historical tickets as supporting knowledge.

3. If previous tickets contain useful troubleshooting steps,
adapt them to the customer's problem.

4. If the retrieved answers are generic,
use your own technical knowledge to produce a better answer.

5. Never simply copy the historical answer.

6. Never answer with only:
"We apologize..."

7. Give practical troubleshooting steps whenever possible.

8. If more information about the customer's TECHNICAL issue is required
to troubleshoot it, ask ONLY for the minimum information needed.

9. Be concise.

10. Maximum 180 words.

11. Do not mention "historical tickets",
"retrieved context",
"RAG",
"AI model",
or "COMPANY FACTS section".

12. Write naturally as a human support engineer.

13. If the customer asks about a factual company detail that is NOT
listed in the COMPANY FACTS section above (e.g. specific employee names,
phone numbers, exact addresses, or dates), do NOT guess, use placeholder
brackets, or ask the customer to clarify what they already clearly asked.
Instead, acknowledge their specific question and say a team member will
follow up with the exact details.

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
                    "content":
                    "You are an experienced technical support engineer. "
                    "Always provide practical troubleshooting steps before asking for more information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=220

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