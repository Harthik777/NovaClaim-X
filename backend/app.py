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
app = FastAPI(title="NovaClaim-X Full VLA Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

print("Loading Semantic Grounding Model (all-MiniLM-L6-v2)...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded. Forensic VLA System Ready.")

# --- 2. RESEARCH TELEMETRY ---
CSV_FILE = 'research_metrics.csv'
try:
    with open(CSV_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'is_mutated', 'rtd_ms', 'mf_score', 'damage_severity', 'sop_decision'])
except FileExistsError: pass

class ResearchObserver:
    def __init__(self): self.start_time = 0
    def start_clock(self): self.start_time = time.time()
    def calculate_metrics(self, reasoning_text, expected_action):
        rtd_ms = (time.time() - self.start_time) * 1000
        mf_score = util.cos_sim(similarity_model.encode(reasoning_text), similarity_model.encode(expected_action)).item()
        return round(rtd_ms, 2), round(mf_score, 4)

observer = ResearchObserver()

# --- 3. THE FORENSIC + SOP ENDPOINT ---
@app.post("/vla/execute")
async def execute_vla_loop(
    ui_screenshot: UploadFile = File(...), 
    evidence_photo: UploadFile = File(...), 
    claim_json: str = Form(...),
    is_mutated: bool = Form(False)
):
    observer.start_clock()
    
    ui_bytes = await ui_screenshot.read()
    evidence_bytes = await evidence_photo.read()
    claim_data = json.loads(claim_json)
    
    # Q1 RESEARCH PROMPT: Multi-modal + SOP Grounding + Voice Governance
    prompt = f"""
    SYSTEM TASK: Autonomous Forensic Claim Verification & SOP Enforcement.
    DATA TO VERIFY: {json.dumps(claim_data)}
    
    RETRIEVED SOP POLICY (RAG SIMULATION):
    - Policy A: If damage is MINOR (scratches/dents), Auto-Deploy the claim.
    - Policy B: If damage is MAJOR (crushed/broken), Require Escalation to human.
    - Policy C: If package is PROPER (no damage) or data mismatches, REJECT.
    
    INSTRUCTIONS:
    1. Analyze 'evidence_photo' to determine Damage Severity (MINOR, MAJOR, or PROPER).
    2. Check 'ui_screenshot' to ensure Data Integrity against DATA TO VERIFY.
    3. Ground decision in the SOP Policy to determine the Action (DEPLOY, ESCALATE, REJECT).
    4. Find the 'Submit Claim' semantic anchor in the UI.
    
    Respond strictly in JSON: 
    {{
        "damage_severity": "MINOR", "MAJOR", or "PROPER",
        "sop_decision": "DEPLOY", "ESCALATE", or "REJECT",
        "thinking": "Explain forensic proof, data check, and SOP application",
        "target_anchor": "semantic-role-name",
        "voice_prompt": "Create a short 1-sentence summary of the damage, followed exactly by: 'Say Deploy to submit or Escalate to review.'"
    }}
    """
    
    try:
        # VISION & COGNITION
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
        rtd, mf_score = observer.calculate_metrics(vla_data['thinking'], vla_data['sop_decision'])
        
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().isoformat(), is_mutated, rtd, mf_score, vla_data['damage_severity'], vla_data['sop_decision']])

        # VOICE GOVERNANCE SYNTHESIS (Nova 2 Sonic)
        # Note: We are using the 'voice_prompt' here, not the internal 'thinking'
        audio_response = bedrock.invoke_model(
            modelId='amazon.nova-sonic-v1:0',
            body=json.dumps({"text": vla_data['voice_prompt'], "voice": "Expressive"})
        )
        audio_base64 = base64.b64encode(audio_response['body'].read()).decode()

        return JSONResponse(content={
            "status": "success",
            "damage_severity": vla_data['damage_severity'],
            "sop_decision": vla_data['sop_decision'],
            "target_anchor": vla_data['target_anchor'] if vla_data['sop_decision'] == "DEPLOY" else None,
            "audio_monologue": audio_base64,
            "telemetry": {"rtd_ms": rtd, "mf_score": mf_score}
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)