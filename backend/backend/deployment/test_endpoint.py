import requests
import json

ENDPOINT_URL = "PASTE_YOUR_URL_HERE"
API_KEY      = "PASTE_YOUR_KEY_HERE"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Test 1 - Normal ticket
test_cases = [
    "My laptop screen is cracked and I cannot work. This is urgent.",
    "I forgot my password and cannot log in to the system.",
    "The printer on floor 3 is making a strange noise.",
]

for text in test_cases:
    payload = {"text": text}
    response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
    result = response.json()
    print(f"\nTicket : {text}")
    print(f"Status : {response.status_code}")
    print(f"Result : {json.dumps(result, indent=2)}")