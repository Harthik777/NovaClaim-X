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
app = FastAPI(title="NovaClaim-X: God-Mode VLA Engine")

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

# --- 2. DYNAMIC RAG SETUP (KNOWLEDGE BASE) ---
try:
    with open("knowledge_base.json", "r") as f:
        knowledge_base = json.load(f)
    kb_texts = [f"Historical Damage: {item['historical_damage']} -> Action: {item['sop_action']}" for item in knowledge_base]
    kb_embeddings = similarity_model.encode(kb_texts)
    print(f"Loaded {len(knowledge_base)} SOPs into Vector Space.")
except FileNotFoundError:
    print("WARNING: knowledge_base.json not found. RAG will be disabled. Create the file to enable Semantic Search.")
    kb_texts = []
    kb_embeddings = []

# --- 3. RESEARCH TELEMETRY ---
CSV_FILE = 'research_metrics.csv'
try:
    with open(CSV_FILE, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'chaos_level', 'rtd_ms', 'mf_score', 'damage_severity', 'sop_decision'])
except FileExistsError:
    pass

class ResearchObserver:
    def __init__(self):
        self.start_time = 0
        
    def start_clock(self):
        self.start_time = time.time()

    def calculate_metrics(self, reasoning_text, expected_action):
        rtd_ms = (time.time() - self.start_time) * 1000
        
        # Calculate Monologue Fidelity (Cosine Similarity)
        reason_vec = similarity_model.encode(reasoning_text)
        action_vec = similarity_model.encode(expected_action)
        mf_score = util.cos_sim(reason_vec, action_vec).item()
        
        return round(rtd_ms, 2), round(mf_score, 4)

observer = ResearchObserver()

# --- 4. THE FORENSIC VLA ENDPOINT ---
@app.post("/vla/execute")
async def execute_vla_loop(
    ui_screenshot: UploadFile = File(...), 
    evidence_photo: UploadFile = File(...), 
    claim_json: str = Form(...),
    chaos_level: int = Form(0)
):
    observer.start_clock()
    
    # Process inputs
    ui_bytes = await ui_screenshot.read()
    evidence_bytes = await evidence_photo.read()
    
    try:
        claim_data = json.loads(claim_json)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid Claim JSON format."}, status_code=400)
    
    # --- TRUE RAG PIPELINE (Semantic Search) ---
    reported_damage = claim_data.get("damage_reported", "")
    if len(kb_embeddings) > 0 and reported_damage:
        query_embedding = similarity_model.encode(reported_damage)
        hits = util.semantic_search(query_embedding, kb_embeddings, top_k=2)[0]
        retrieved_sops = "\n".join([kb_texts[hit['corpus_id']] for hit in hits])
    else:
        retrieved_sops = "No historical SOPs available. Default to standard processing."
    
    # --- THE Q1 RESEARCH PROMPT ---
    prompt = f"""
    SYSTEM TASK: Autonomous Forensic Claim Verification & Visual Grounding.
    DATA TO VERIFY: {json.dumps(claim_data)}
    
    DYNAMICALLY RETRIEVED HISTORICAL SOPs:
    {retrieved_sops}
    
    INSTRUCTIONS:
    1. Analyze 'evidence_photo' to determine Damage Severity.
    2. Check 'ui_screenshot' to ensure Data Integrity against DATA TO VERIFY.
    3. Ground your decision entirely in the RETRIEVED SOPs to determine Action (DEPLOY, ESCALATE, REJECT).
    4. If DEPLOY is chosen, locate the 'Submit Claim' button on the 'ui_screenshot' and extract its spatial bounding box.
    
    Respond strictly in JSON: 
    {{
        "damage_severity": "MINOR", "MAJOR", or "PROPER",
        "sop_decision": "DEPLOY", "ESCALATE", or "REJECT",
        "thinking": "Explain forensic proof, data check, and how retrieved SOPs were applied",
        "target_anchor": "semantic-role-name",
        "bounding_box": [ymin, xmin, ymax, xmax] format normalized 0-1000 scale, empty list [] if rejected/escalated,
        "voice_prompt": "1-sentence damage summary. End exactly with: 'Say Deploy to submit or Escalate to review.'"
    }}
    """
    
    try:
        # PASS 1: AGENT A (THE PROPOSER)
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

        # =====================================================================
        # PASS 2: AGENT B (THE CRITIC) - UNCOMMENT FOR Q1 IEEE PAPER
        # =====================================================================
        # critic_prompt = f"""
        # Verify if this proposed decision: {vla_data.get('sop_decision')} 
        # mathematically matches this SOP policy: {retrieved_sops}. 
        # Return strictly JSON: {{"approved": true/false, "feedback": "reasoning"}}
        # """
        # critic_response = bedrock.invoke_model(
        #     modelId='amazon.nova-lite-v1:0',
        #     body=json.dumps({
        #         "anthropic_version": "bedrock-2023-05-31",
        #         "messages": [{"role": "user", "content": [{"type": "text", "text": critic_prompt}]}]
        #     })
        # )
        # critique = json.loads(critic_response['body'].read())
        # if not critique.get('approved', True):
        #     print(f"CRITIC INTERVENTION: {critique.get('feedback')}")
        # =====================================================================

        # --- TELEMETRY LOGGING ---
        rtd, mf_score = observer.calculate_metrics(
            reasoning_text=vla_data.get('thinking', ''),
            expected_action=vla_data.get('sop_decision', '')
        )
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), chaos_level, rtd, mf_score, 
                vla_data.get('damage_severity'), vla_data.get('sop_decision')
            ])

        # --- NATIVE AUDIO SYNTHESIS (Nova 2 Sonic) ---
        audio_response = bedrock.invoke_model(
            modelId='amazon.nova-sonic-v1:0',
            body=json.dumps({"text": vla_data.get('voice_prompt', 'Error generating prompt.'), "voice": "Expressive"})
        )
        audio_base64 = base64.b64encode(audio_response['body'].read()).decode()

        # --- FINAL PAYLOAD ---
        return JSONResponse(content={
            "status": "success",
            "damage_severity": vla_data.get('damage_severity'),
            "sop_decision": vla_data.get('sop_decision'),
            "target_anchor": vla_data.get('target_anchor'),
            "bounding_box": vla_data.get('bounding_box', []),
            "audio_monologue": audio_base64,
            "telemetry": {
                "rtd_ms": rtd,
                "mf_score": mf_score,
                "chaos_level": chaos_level
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)