import sys
import requests

precision = sys.argv[1]
query = sys.argv[2]

url = "https://www.googleapis.com/customsearch/v1"
params = {"cx": "011931726167723972512:orkup7yeals",
		  "q": query,
		  "key": "AIzaSyAg_FedCkdEHFmYwRdkqS5Im2zeOjlrC4Y",
		  "num": 10}

response = requests.get(url = url, params = params)

data = response.json()

print(len(data))
# print(data)