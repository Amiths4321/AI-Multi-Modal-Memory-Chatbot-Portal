import requests
from config import Config

class QwenVLPipelineHandler:
    def __init__(self):
        # Strip trailing slashes or duplicate paths from base configuration URL safely
        base_url = Config.REMOTE_SERVER_URL.rstrip('/')
        
        if not base_url.endswith('/api/chat'):
            if base_url.endswith('/api'):
                self.url = f"{base_url}/chat"
            else:
                self.url = f"{base_url}/api/chat"
        else:
            self.url = base_url

        self.model = Config.QWEN_MODEL_NAME
        print(f"📡 Target API Route established strictly at: {self.url}")

    def generate_response(self, current_message, short_term_history, long_term_memories, image_b64=None):
        """Sends the contextual payload to the remote GPU server running Qwen-VL"""
        memory_context = "\n".join([f"- {m}" for m in long_term_memories]) if long_term_memories else "No past context found."
        
        system_prompt = (
            "You are an advanced Vision-Language AI companion with long-term memory capabilities.\n"
            "Below is historical context recalled from past conversations with this user. "
            "Use it to personalize answers naturally without saying 'Based on your memory'.\n\n"
            f"=== RECALLED LONG-TERM MEMORIES ===\n{memory_context}\n===================================="
        )
        
        # Format payload specifically for Ollama's Chat API
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in short_term_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        if image_b64:
            messages.append({
                "role": "user",
                "content": current_message,
                "images": [image_b64]
            })
        else:
            messages.append({
                "role": "user",
                "content": current_message
            })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['message']['content']
        except requests.exceptions.RequestException as e:
            return f"❌ Remote GPU Server Connection Error: Failed to reach Qwen-VL. Details: {str(e)}"