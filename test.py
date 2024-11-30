import requests

url = "http://enderwire.local/printer"

test = requests.post(f"{url}/print/resume")
print(test)