import os
import json
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


genai.configure(api_key="") 
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_PATH = 'phoenix_ai_model.h5'
CLASS_NAMES = ['gender_female', 'gender_male', 'kids', 'safe', 'unsafe']

try:
    mobilenet_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("AI System Online with FastAPI")
except Exception as e:
    print(f"Error loading model: {e}")
    mobilenet_model = None

async def get_dynamic_analysis_from_gemini(image_pil, detected_label, title="", description="", category=""):
    prompt = f"""
    The product is classified as '{detected_label}'.
    User Context:
    Title: {title}
    Description: {description}
    Category: {category}

    Tasks:
    1. Verify if the image matches the provided title, description, and category.
    2. Estimate suitable 'min_age' and 'max_age'.
    3. Confirm matching (True/False).

    Return ONLY JSON:
    {{"min_age": int, "max_age": int, "matching_confirmed": bool}}
    """
    try:
        response = gemini_model.generate_content([prompt, image_pil])
        raw_text = response.text.strip()
        
        
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1

        if start != -1 and end != -1:
            clean_json = raw_text[start:end]
            return json.loads(clean_json)
        else:
            raise ValueError("Invalid JSON from Gemini")

    except Exception:
        
        return {"min_age": 12, "max_age": 60, "matching_confirmed": True}

@app.post("/analyze-product")
async def analyze_endpoint(
    image: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    category: str = Form("")
):
    if mobilenet_model is None:
        return JSONResponse(content={"error": "AI Model not loaded"}, status_code=500)

    try:
        
        image_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        
        img_resized = pil_img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        
        predictions = mobilenet_model.predict(img_array, verbose=0)
        confidence = float(np.max(predictions[0]))
        label = CLASS_NAMES[np.argmax(predictions[0])]

        
        if confidence < 0.6:
            return {
                "status": "rejected_low_confidence",
                "data": None,
                "error_msg": "Model confidence too low."
            }

        status = "rejected"
        final_data = {"target_gender": "unknown", "min_age": 0, "max_age": 0}

        if label != 'unsafe':
            
            dynamic_data = await get_dynamic_analysis_from_gemini(
                pil_img, label, title, description, category
            )
            
            if dynamic_data.get("matching_confirmed") is True:
                status = "approved"
                
                
                target_gender = "unisex"
                if label == 'gender_male': target_gender = "male"
                elif label == 'gender_female': target_gender = "female"
                elif label == 'kids': target_gender = "kids"

                final_data = {
                    "target_gender": target_gender,
                    "min_age": int(dynamic_data.get("min_age", 12)),
                    "max_age": int(dynamic_data.get("max_age", 60))
                }
            else:
                status = "rejected_content_mismatch"
        else:
            status = "rejected_unsafe"

        return {
            "status": status,
            "data": final_data if status == "approved" else None,
            "error_msg": "" if status == "approved" else f"Analysis failed: {status}"
        }

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)