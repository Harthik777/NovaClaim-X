import boto3
import json
import base64
import time
import csv
import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS Bedrock
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

class VoiceRequest(BaseModel):
    text: str

def log_for_research(image_name, reasoning, latency):
    """Saves VLA metrics to a CSV file for the research paper."""
    file_exists = os.path.isfile('research_metrics.csv')
    with open('research_metrics.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Image_Name', 'Agentic_Reasoning', 'Latency_Seconds'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, reasoning, latency])

@app.post("/analyze-damage")
async def analyze_damage(file: UploadFile = File(...)):
    """Vision-Language Model: Analyzes image and tracks latency."""
    start_time = time.time()
    
    try:
        image_bytes = await file.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = """
        Analyze this package image. 
        1. Identify damage (Crush, Puncture, Leak).
        2. Generate a short 1-sentence summary of the damage for the operator.
        Format output as clean text.
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
        analysis_text = result['output']['text']
        
        end_time = time.time()
        latency = round(end_time - start_time, 2)
        
        # Log data for Paranjay's paper
        log_for_research(file.filename, analysis_text, latency)

        return {"status": "success", "analysis": analysis_text, "latency": latency}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-voice")
async def generate_voice(request: VoiceRequest):
    """Text-to-Speech: Converts reasoning to audio using Nova 2 Sonic."""
    try:
        system_context = "You are the NovaClaim-X Logistics Supervisor. Speak professionally."
        full_prompt = f"{system_context} {request.text}"

        response = bedrock.invoke_model(
            modelId="amazon.nova-2-sonic-v1",
            body=json.dumps({
                "text": full_prompt,
                "voice": "expressive_male_1"
            })
        )
        
        result = json.loads(response['body'].read())
        return {"status": "success", "audio": result.get('audio_base64', '')}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)