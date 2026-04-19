import requests

url = "http://127.0.0.1:5000/upload"

# Test with a legitimate transaction
with open("uploads/predict.json", "rb") as f:
    response = requests.post(url, files={"file": f})

print("Status code:", response.status_code)
print("Response:", response.json())