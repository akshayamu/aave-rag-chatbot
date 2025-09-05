# src/download_docs.py
import requests
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# URLs to download
urls = {
    "data/aave_v2.txt": "https://docs.aave.com/developers/v/2.0/",
    "data/aave_gov.txt": "https://docs.aave.com/governance/"
}

# Download each URL
for filename, url in urls.items():
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Successfully downloaded {url} to {filename}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")

print("Download process completed.")