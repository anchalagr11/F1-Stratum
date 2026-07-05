import requests
import json

url = 'http://localhost:8000/strategy'
payload = {
    "year": 2024,
    "gp": "Singapore",
    "driver": "NOR",
    "decision_lap": 20,
    "max_stops": 2,
    "top_n": 1
}
response = requests.post(url, json=payload)
print(response.status_code)
print(json.dumps(response.json(), indent=2))
