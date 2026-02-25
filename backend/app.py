import boto3
import json
import base64
import time
import csv
import os
import re
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="NovaClaim-X VLA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS Bedrock (us-east-1 required for Nova 2)
try:
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
except Exception as e:
    print(f"⚠️ AWS Init Error: {e}. Run 'aws configure' in terminal.")

class VoiceRequest(BaseModel):
    text: str

def log_telemetry(image_name, damage_type, confidence, latency):
    """Research Paper Telemetry: Logs to CSV for Paranjay's graphs."""
    file_exists = os.path.isfile('research_metrics.csv')
    with open('research_metrics.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Image_Name', 'Damage_Type', 'Confidence_Score', 'Latency_Seconds'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, damage_type, confidence, latency])

@app.post("/analyze-damage")
async def analyze_damage(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        image_bytes = await file.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # God Mode Prompt: Forcing structured JSON for the VLA pipeline
        prompt = """
        Analyze this package image for logistics damage.
        Respond ONLY with a valid JSON object in this exact format, nothing else:
        {
            "damage_type": "Crush | Puncture | Water Damage | Intact",
            "confidence_score": 0.95,
            "bounding_box": "[X, Y, W, H]",
            "summary": "1-sentence summary of the damage for the operator."
        }
        """

        response = bedrock.invoke_model(
            modelId="amazon.nova-2-lite-v1",
            body=json.dumps({
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"bytes": encoded_image}}
                ]}]
            })
        )
        
        result = json.loads(response['body'].read())
        raw_text = result['output']['text']
        
        # Strip markdown formatting if Nova adds ```json ... ```
        clean_json_str = re.sub(r"```json|```", "", raw_text).strip()
        vla_data = json.loads(clean_json_str)
        
        latency = round(time.time() - start_time, 2)
        
        # Log telemetry for the CVPR paper
        log_telemetry(file.filename, vla_data.get("damage_type", "Unknown"), vla_data.get("confidence_score", 0.0), latency)

        return {
            "status": "success", 
            "data": vla_data, 
            "latency": latency
        }
    
    except Exception as e:
        print(f"Vision Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Vision AI failed to process image.")

@app.post("/generate-voice")
async def generate_voice(request: VoiceRequest):
    try:
        system_context = "You are the NovaClaim-X Logistics Supervisor. Speak efficiently, professionally, and urgently."
        
        response = bedrock.invoke_model(
            modelId="amazon.nova-2-sonic-v1",
            body=json.dumps({
                "text": f"{system_context} {request.text}",
                "voice": "expressive_male_1"
            })
        )
        
        result = json.loads(response['body'].read())
        return {"status": "success", "audio": result.get('audio_base64', '')}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Sonic Voice generation failed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)