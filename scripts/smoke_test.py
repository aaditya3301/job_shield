"""Check a running local or containerized API using only the standard library."""
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def main(base_url):
    for attempt in range(30):
        try:
            with urlopen(f"{base_url}/health", timeout=3) as response:
                health = json.load(response)
            assert health["status"] == "ok" and health["model_loaded"] is True
            break
        except (URLError, HTTPError, TimeoutError, ConnectionError):
            if attempt == 29:
                raise
            time.sleep(1)

    payload = {
        "title": "Work from home assistant",
        "description": "Earn money immediately. No interview required. Pay a registration fee to start.",
        "requirements": "No experience necessary. Send your bank account details today.",
    }
    request = Request(f"{base_url}/predict", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    assert 0 <= result["fraud_probability"] <= 1
    assert result["fraudulent"] == int(result["fraud_probability"] >= 0.5)
    print(json.dumps({"health": health, "prediction": result}, indent=2))
    print("API smoke test passed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
