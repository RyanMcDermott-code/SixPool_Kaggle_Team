import requests

print(requests.get("http://localhost:11434/api/tags").json())