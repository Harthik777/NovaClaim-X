import boto3
import base64
import json
import csv
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util

# --- INITIALIZATION ---
app = FastAPI(title="Agentic VLA Engine")
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

print("Loading Semantic Grounding Model (all-MiniLM-L6-v2)...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded. VLA System Ready.")

# Ensure CSV exists with headers for Paranjay's research
CSV_FILE = 'research_metrics.csv'
try:
    with open(CSV_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'is_mutated', 'rtd_ms', 'mf_score', 'srr_success', 'confidence_score'])
except FileExistsError:
    pass

# --- THE RESEARCH OBSERVER ---
class ResearchObserver:
    def __init__(self):
        self.start_time = 0
        
    def start_clock(self):
        self.start_time = time.time()

    def calculate_metrics(self, reasoning_text, target_anchor, expected_action, is_mutated):
        # 1. RTD: Reasoning-to-Action Delay (ms)
        rtd = (time.time() - self.start_time) * 1000
        
        # 2. MF-Score: Cosine similarity between reasoning and expected action
        reason_vec = similarity_model.encode(reasoning_text)
        action_vec = similarity_model.encode(expected_action.replace("-", " "))
        mf_score = util.cos_sim(reason_vec, action_vec).item()
        
        # 3. SRR: Semantic Recovery Rate (1 if successful, 0 if failed)
        success = 1 if (target_anchor == expected_action) else 0
        
        return round(rtd, 2), round(mf_score, 4), success

observer = ResearchObserver()

# --- THE CORE VLA ENDPOINT ---
@app.post("/vla/execute")
async def execute_vla_loop(
    file: UploadFile = File(...), 
    is_mutated: bool = Form(False),
    expected_action: str = Form("claim-submit-btn") 
):
    observer.start_clock()
    image_bytes = await file.read()
    
    prompt = """
    Analyze this UI layout. Identify the primary 'Submit Claim' or 'Proceed' action element.
    Ignore altered styling, scrambled IDs, or chaotic positioning.
    Determine the Semantic Role of the target.
    Respond ONLY with strict JSON: {"thinking": "Explanation of visual logic", "target_anchor": "semantic-role-name", "confidence": 0.99}
    """
    
    try:
        # 1. VISION & REASONING (Nova 2 Lite)
        vision_response = bedrock.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user", 
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}}, 
                        {"type": "text", "text": prompt}
                    ]
                }],
                "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}}
            })
        )
        
        vla_data = json.loads(vision_response['body'].read())
        
        # 2. TELEMETRY LOGGING
        rtd, mf_score, srr_success = observer.calculate_metrics(
            reasoning_text=vla_data['thinking'],
            target_anchor=vla_data['target_anchor'],
            expected_action=expected_action,
            is_mutated=is_mutated
        )
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), is_mutated, rtd, mf_score, srr_success, vla_data['confidence']])

        # 3. NATIVE AUDIO SYNTHESIS (Nova 2 Sonic)
        audio_response = bedrock.invoke_model(
            modelId='amazon.nova-sonic-v1:0',
            body=json.dumps({"text": vla_data['thinking'], "voice": "Expressive"})
        )
        audio_base64 = base64.b64encode(audio_response['body'].read()).decode()

        # 4. ACTION PAYLOAD FOR RPA
        return JSONResponse(content={
            "status": "success",
            "action": "click",
            "target_anchor": vla_data['target_anchor'],
            "audio_monologue": audio_base64,
            "telemetry": {
                "rtd_ms": rtd,
                "mf_score": mf_score,
                "confidence": vla_data['confidence']
            }
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)