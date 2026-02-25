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
app = FastAPI(title="NovaClaim-X: Multi-Agent VLA Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

print("Loading Semantic Grounding Model (all-MiniLM-L6-v2)...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded. Multi-Agent System Ready.")

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
        # Added critic_interventions to track how often Agent B catches an error
        writer.writerow(['timestamp', 'chaos_level', 'rtd_ms', 'mf_score', 'damage_severity', 'sop_decision', 'critic_interventions'])
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

def clean_json_response(raw_text):
    """Helper to strip markdown formatting if the LLM wraps the JSON."""
    return raw_text.strip().removeprefix("```json").removesuffix("```").strip()

# --- 4. THE FORENSIC MULTI-AGENT ENDPOINT ---
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
    
    # --- RAG PIPELINE (Semantic Search) ---
    reported_damage = claim_data.get("damage_reported", "")
    if len(kb_embeddings) > 0 and reported_damage:
        query_embedding = similarity_model.encode(reported_damage)
        hits = util.semantic_search(query_embedding, kb_embeddings, top_k=2)[0]
        retrieved_sops = "\n".join([kb_texts[hit['corpus_id']] for hit in hits])
    else:
        retrieved_sops = "No historical SOPs available. Default to standard processing."
    
    # =====================================================================
    # PASS 1: AGENT A (THE PROPOSER) - Vision-Language Reasoning
    # =====================================================================
    prompt_a = f"""
    SYSTEM TASK: Autonomous Forensic Claim Verification & Visual Grounding.
    DATA TO VERIFY: {json.dumps(claim_data)}
    DYNAMICALLY RETRIEVED HISTORICAL SOPs: {retrieved_sops}
    
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
        vision_response = bedrock.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user", 
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(evidence_bytes).decode()}}, 
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(ui_bytes).decode()}},
                        {"type": "text", "text": prompt_a}
                    ]
                }],
                "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}}
            })
        )
        
        raw_proposal = json.loads(vision_response['body'].read())
        # Depending on how the API returns text, it might be in content[0]['text'] or raw depending on Bedrock's Nova wrapper.
        # Assuming standard Bedrock Converse/Messages format structure mapped here:
        if 'content' in raw_proposal:
            vla_data = json.loads(clean_json_response(raw_proposal['content'][0]['text']))
        else:
            # Fallback if the response is direct JSON string
            vla_data = raw_proposal

        # =====================================================================
        # PASS 2: AGENT B (THE CRITIC) - Agentic Reflection & Alignment
        # =====================================================================
        prompt_b = f"""
        REVIEWER TASK: Validate Agent A's proposed decision against the strict SOP.
        RETRIEVED SOP POLICY: {retrieved_sops}
        AGENT A PROPOSED PLAN: {json.dumps(vla_data)}
        
        INSTRUCTIONS:
        Check if Agent A's 'sop_decision' logically and semantically matches the policy for the described damage.
        If it violates the SOP, correct the plan to ESCALATE or REJECT.
        
        Respond strictly in JSON:
        {{
            "approved": true/false, 
            "feedback": "Explain why it was approved or rejected based on SOP.", 
            "corrected_plan": {{"damage_severity": "...", "sop_decision": "...", "target_anchor": "...", "bounding_box": [], "thinking": "...", "voice_prompt": "..."}}
        }}
        """
        
        critic_response = bedrock.invoke_model(
            modelId='amazon.nova-lite-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_b}]}]
            })
        )
        
        raw_critique = json.loads(critic_response['body'].read())
        if 'content' in raw_critique:
            critique = json.loads(clean_json_response(raw_critique['content'][0]['text']))
        else:
            critique = raw_critique

        # --- MERGE MULTI-AGENT CONSENSUS ---
        is_approved = critique.get('approved', True)
        final_vla = vla_data if is_approved else critique.get('corrected_plan', vla_data)
        critic_interventions = 0 if is_approved else 1

        if not is_approved:
            print(f"⚠️ CRITIC INTERVENTION TRIGGERED: {critique.get('feedback')}")

        # --- TELEMETRY LOGGING ---
        rtd, mf_score = observer.calculate_metrics(
            reasoning_text=final_vla.get('thinking', ''),
            expected_action=final_vla.get('sop_decision', '')
        )
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), chaos_level, rtd, mf_score, 
                final_vla.get('damage_severity'), final_vla.get('sop_decision'), critic_interventions
            ])

        # --- NATIVE AUDIO SYNTHESIS (Nova 2 Sonic) ---
        audio_response = bedrock.invoke_model(
            modelId='amazon.nova-sonic-v1:0',
            body=json.dumps({"text": final_vla.get('voice_prompt', 'Error generating prompt.'), "voice": "Expressive"})
        )
        audio_base64 = base64.b64encode(audio_response['body'].read()).decode()

        # --- FINAL PAYLOAD ---
        return JSONResponse(content={
            "status": "success",
            "damage_severity": final_vla.get('damage_severity'),
            "sop_decision": final_vla.get('sop_decision'),
            "target_anchor": final_vla.get('target_anchor'),
            "bounding_box": final_vla.get('bounding_box', []),
            "critic_approved": is_approved,
            "critic_feedback": critique.get('feedback', ''),
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