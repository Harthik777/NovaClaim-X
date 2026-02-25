import boto3
import base64
import json
import csv
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

# --- 1. INITIALIZATION ---
app = FastAPI(title="NovaClaim-X Forensic VLA Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

print("Loading Semantic Grounding Model (all-MiniLM-L6-v2)...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded. Forensic VLA System Ready.")

# --- 2. RESEARCH TELEMETRY ---
CSV_FILE = 'research_metrics.csv'
try:
    with open(CSV_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'is_mutated', 'rtd_ms', 'mf_score', 'forensic_status', 'data_match', 'confidence'])
except FileExistsError:
    pass

class ResearchObserver:
    def __init__(self):
        self.start_time = 0
        
    def start_clock(self):
        self.start_time = time.time()

    def calculate_metrics(self, reasoning_text, target_anchor, expected_action):
        rtd_ms = (time.time() - self.start_time) * 1000
        
        # Calculate Monologue Fidelity (Cosine Similarity)
        reason_vec = similarity_model.encode(reasoning_text)
        action_vec = similarity_model.encode(expected_action)
        mf_score = util.cos_sim(reason_vec, action_vec).item()
        
        return round(rtd_ms, 2), round(mf_score, 4)

observer = ResearchObserver()

# --- 3. THE FORENSIC VLA ENDPOINT ---
@app.post("/vla/execute")
async def execute_vla_loop(
    ui_screenshot: UploadFile = File(...), 
    evidence_photo: UploadFile = File(...), 
    claim_json: str = Form(...),  # Dynamic JSON input instead of hardcoded checks
    is_mutated: bool = Form(False)
):
    observer.start_clock()
    
    # Process inputs
    ui_bytes = await ui_screenshot.read()
    evidence_bytes = await evidence_photo.read()
    claim_data = json.loads(claim_json)
    
    # Q1 RESEARCH PROMPT: Multi-modal Cross-Referencing
    prompt = f"""
    SYSTEM TASK: Autonomous Forensic Claim Verification & Action.
    DATA TO VERIFY: {json.dumps(claim_data)}
    
    INSTRUCTIONS:
    STEP 1 (PHYSICAL FORENSICS): Analyze the 'evidence_photo'. Is the package crushed/damaged, or is it proper/intact?
    STEP 2 (DATA INTEGRITY): Cross-reference the 'ui_screenshot' with the DATA TO VERIFY. Do the details match?
    STEP 3 (ACTION INTENT): 
      - If package is PROPER -> REJECT.
      - If data does NOT match -> REJECT.
      - If package is CRUSHED AND data MATCHES -> Locate the semantic anchor for the 'Submit Claim' button.
    
    Respond ONLY in strict JSON: 
    {{
        "forensic_status": "CRUSHED" or "PROPER",
        "data_match": true or false,
        "thinking": "Explain forensic proof and UI data check",
        "target_anchor": "semantic-role-name" or "NONE",
        "confidence": 0.99
    }}
    """
    
    try:
        # VISION & COGNITION (Nova 2 Lite handles multi-image + text)
        vision_response = bedrock.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user", 
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(evidence_bytes).decode()}}, 
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(ui_bytes).decode()}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}}
            })
        )
        
        vla_data = json.loads(vision_response['body'].read())
        
        # Determine the expected intent for the MF-Score calculation
        expected_intent = "submit claim" if (vla_data.get('forensic_status') == "CRUSHED" and vla_data.get('data_match') is True) else "reject claim"
        
        # TELEMETRY LOGGING
        rtd, mf_score = observer.calculate_metrics(
            reasoning_text=vla_data['thinking'],
            target_anchor=vla_data['target_anchor'],
            expected_action=expected_intent
        )
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), is_mutated, rtd, mf_score, 
                vla_data.get('forensic_status'), vla_data.get('data_match'), vla_data.get('confidence')
            ])

        # NATIVE AUDIO SYNTHESIS (Nova 2 Sonic)
        audio_response = bedrock.invoke_model(
            modelId='amazon.nova-sonic-v1:0',
            body=json.dumps({"text": vla_data['thinking'], "voice": "Expressive"})
        )
        audio_base64 = base64.b64encode(audio_response['body'].read()).decode()

        # FINAL PAYLOAD TO FRONTEND/RPA
        return JSONResponse(content={
            "status": "success" if expected_intent == "submit claim" else "rejected",
            "forensic_status": vla_data.get('forensic_status'),
            "data_match": vla_data.get('data_match'),
            "target_anchor": vla_data.get('target_anchor'),
            "audio_monologue": audio_base64,
            "telemetry": {
                "rtd_ms": rtd,
                "mf_score": mf_score,
                "confidence": vla_data.get('confidence')
            }
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)