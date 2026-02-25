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
app = FastAPI(title="NovaClaim-X Q1 VLA Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Knowledge Base for True Dynamic RAG
try:
    with open("knowledge_base.json", "r") as f:
        knowledge_base = json.load(f)
    kb_texts = [f"Damage: {item['historical_damage']} -> Action: {item['sop_action']}" for item in knowledge_base]
    kb_embeddings = similarity_model.encode(kb_texts)
except FileNotFoundError:
    kb_texts, kb_embeddings = [], []

# --- 2. RESEARCH TELEMETRY (Updated for Entropy Slider) ---
CSV_FILE = 'research_metrics.csv'
try:
    with open(CSV_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        # Added chaos_level to track the Ablation Study degradation curve
        writer.writerow(['timestamp', 'chaos_level', 'rtd_ms', 'mf_score', 'damage_severity', 'sop_decision'])
except FileExistsError: pass

class ResearchObserver:
    def __init__(self): self.start_time = 0
    def start_clock(self): self.start_time = time.time()
    def calculate_metrics(self, reasoning_text, expected_action):
        rtd_ms = (time.time() - self.start_time) * 1000
        mf_score = util.cos_sim(similarity_model.encode(reasoning_text), similarity_model.encode(expected_action)).item()
        return round(rtd_ms, 2), round(mf_score, 4)

observer = ResearchObserver()

# --- 3. THE FORENSIC + DYNAMIC RAG + VISUAL GROUNDING ENDPOINT ---
@app.post("/vla/execute")
async def execute_vla_loop(
    ui_screenshot: UploadFile = File(...), 
    evidence_photo: UploadFile = File(...), 
    claim_json: str = Form(...),
    chaos_level: int = Form(0) # UPGRADE 2: Entropy Slider Value (0-100)
):
    observer.start_clock()
    
    ui_bytes = await ui_screenshot.read()
    evidence_bytes = await evidence_photo.read()
    claim_data = json.loads(claim_json)
    
    # RAG PIPELINE
    reported_damage = claim_data.get("damage_reported", "")
    if len(kb_embeddings) > 0:
        query_embedding = similarity_model.encode(reported_damage)
        hits = util.semantic_search(query_embedding, kb_embeddings, top_k=2)[0]
        retrieved_sops = "\n".join([kb_texts[hit['corpus_id']] for hit in hits])
    else:
        retrieved_sops = "No SOPs available."
    
    # Q1 RESEARCH PROMPT: Added Visual Grounding (Bounding Box coordinates)
    prompt = f"""
    SYSTEM TASK: Autonomous Forensic Claim Verification & Visual Grounding.
    DATA TO VERIFY: {json.dumps(claim_data)}
    RETRIEVED SOPs: {retrieved_sops}
    
    INSTRUCTIONS:
    1. Analyze 'evidence_photo' to determine Damage Severity.
    2. Check 'ui_screenshot' to ensure Data Integrity against DATA TO VERIFY.
    3. Ground your decision entirely in the RETRIEVED SOPs to determine Action (DEPLOY, ESCALATE, REJECT).
    4. If DEPLOY is chosen, locate the 'Submit Claim' button on the 'ui_screenshot' and extract its spatial bounding box.
    
    Respond strictly in JSON: 
    {{
        "damage_severity": "MINOR", "MAJOR", or "PROPER",
        "sop_decision": "DEPLOY", "ESCALATE", or "REJECT",
        "thinking": "Explain forensic proof and SOP application",
        "target_anchor": "semantic-role-name",
        "bounding_box": [ymin, xmin, ymax, xmax] format normalized 0-1000 scale, empty list [] if rejected/escalated,
        "voice_prompt": "1-sentence damage summary. End exactly with: 'Say Deploy to submit or Escalate to review.'"
    }}
    """
    
    try:
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
        
        # Log the specific Chaos Level for the Robustness Curve
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().isoformat(), chaos_level, rtd, mf_score, vla_data['damage_severity'], vla_data['sop_decision']])

        audio_response = bedrock.invoke_model(
            modelId='amazon.nova-sonic-v1:0',
            body=json.dumps({"text": vla_data['voice_prompt'], "voice": "Expressive"})
        )
        audio_base64 = base64.b64encode(audio_response['body'].read()).decode()

        return JSONResponse(content={
            "status": "success",
            "damage_severity": vla_data['damage_severity'],
            "sop_decision": vla_data['sop_decision'],
            "target_anchor": vla_data['target_anchor'],
            "bounding_box": vla_data.get('bounding_box', []), # UPGRADE 1: Return Coordinates
            "audio_monologue": audio_base64,
            "telemetry": {"rtd_ms": rtd, "mf_score": mf_score, "chaos_level": chaos_level}
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)