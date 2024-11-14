import os
import requests

# Load environment variables from .env file

class ChatGPT:
    def __init__(self):
        self.api_key = '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'
        if not self.api_key:
            raise ValueError("No API key found. Please set OPENAI_API_KEY environment variable.")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def get_response(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Provéeme observaciones de la relación del concepto de {prompt} con respecto a las siguientes competencias: \"Solución de problemas en Física, Formulación de soluciones, Identificar los fenómenos físicos, Generar estrategias de solución, Experimentación en física.\" De manera ágil y precisa."}
            ]
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response_data = response.json()
        return response_data.get('choices')[0].get('message').get('content')
