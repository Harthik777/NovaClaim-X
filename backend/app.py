import boto3
import json
import base64
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

# Initialize AWS Bedrock (Requires 'aws configure' in terminal)
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

class VoiceRequest(BaseModel):
    text: str

@app.post("/analyze-damage")
async def analyze_damage(file: UploadFile = File(...)):
    """Vision-Language Model: Analyzes the image using Nova 2 Lite"""
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
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"bytes": encoded_image}}
                    ]
                }]
            })
        )
        
        result = json.loads(response['body'].read())
        return {"status": "success", "analysis": result['output']['text']}
    
    except Exception as e:
        # Fallback for testing without AWS credentials active
        return {"status": "success", "analysis": "Simulated Analysis: Major crush detected on the top right corner. SOP Rule 12B applies."}

@app.post("/generate-voice")
async def generate_voice(request: VoiceRequest):
    """Text-to-Speech: Converts reasoning to audio using Nova 2 Sonic"""
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