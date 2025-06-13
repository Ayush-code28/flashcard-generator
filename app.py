def query_huggingface(prompt):
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Check Content-Type before parsing
        if "application/json" not in response.headers.get("Content-Type", ""):
            return "⚠️ Error: Received non-JSON response. The model may still be loading or the API is overloaded."

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]['generated_text']
        else:
            return "⚠️ Error: Unexpected response format from the model."

    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP error {e.response.status_code}: {e.response.text}"
    except requests.exceptions.JSONDecodeError:
        return "⚠️ JSON decode error: Model response could not be parsed. Try again in 30 seconds."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"
