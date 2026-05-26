from fastapi import FastAPI
from pydantic import BaseModel
import requests
import google.generativeai as genai
import json


GEMINI_API_KEY = ""
BACKEND_API_URL = "BACKEND_API_URL_HERE"  


genai.configure(api_key="")

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    history: list = []


def search_backend(keyword=None, max_price=None, section=None):
    try:
        payload = {
            "keyword": keyword,
            "max_price": max_price,
            "section": section
        }

        response = requests.post(BACKEND_API_URL, json=payload)
        return response.json()

    except Exception as e:
        return {"error": str(e)}


system_instruction = """
You are Phoenix AI Assistant.

Behavior rules:
1. Detect user intent:
   - If user is asking about products → extract search parameters
   - If user is asking general questions → respond normally like a helpful assistant

2. Language:
   - Respond in the SAME language as the user (Arabic or English automatically)

3. If the user is asking about products, return JSON ONLY in this format:
{
  "intent": "search",
  "keyword": "...",
  "max_price": number or null,
  "section": "market" or "donation" or null
}

4. If the user is NOT asking about products:
Return JSON in this format:
{
  "intent": "chat",
  "response": "your answer here"
}
"""


model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=system_instruction,
    generation_config={"response_mime_type": "application/json"}
)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message

    
    response = model.generate_content(user_message)

    
    clean_text = response.text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean_text)

        
        if parsed.get("intent") == "search":
            keyword = parsed.get("keyword")
            max_price = parsed.get("max_price")
            section = parsed.get("section")

            
            if not keyword:
                return {
                    "message": "Please specify what you're looking for",
                    "products": []
                }

            results = search_backend(keyword, max_price, section)

            
            if isinstance(results, dict) and "error" in results:
                return {
                    "message": "Something went wrong, please try again.",
                    "products": []
                }

            
            if not results or len(results) == 0:
                results = search_backend(keyword, None, "donation")

            return {
                "message": "",
                "products": results
            }

        
        elif parsed.get("intent") == "chat":
            return {
                "message": parsed.get("response", ""),
                "products": []
            }

    except:
        pass

    
    return {
        "message": response.text,
        "products": []
    }

