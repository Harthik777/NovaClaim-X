import os
import json
import boto3
import math
from playwright.sync_api import sync_playwright

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

def get_aws_embedding(text):
    """Fetches text embeddings strictly via Amazon Nova Multimodal Embeddings"""
    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "text": {"value": text}
        }
    }
    try:
        response = bedrock.invoke_model(
            modelId='amazon.nova-2-multimodal-embeddings-v1:0', 
            body=json.dumps(request_body)
        )
        return json.loads(response['body'].read())['embeddings'][0]['embedding']
    except Exception as e:
        print(f"AWS Embedding error: {e}")
        return [0.0] * 1024

def calculate_cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

def execute_vla_action(tracking_id, date, desc, decision, target_anchor):
    print("\n--- 🤖 VLA AUTOMATION ENGINE STARTED ---")
    print(f"Action: {decision} | Target Anchor Intent: '{target_anchor}'")

    if decision != "DEPLOY":
        print("🛑 Agentic Decision was not DEPLOY. Halting automation for Human Review.")
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Load the local target portal
        portal_path = f"file://{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'target_portal', 'portal.html'))}"
        page.goto(portal_path)
        print("🌐 Target Portal Loaded.")

        try:
            # 1. Fill out the form
            page.fill("input[type='text']", tracking_id)
            page.fill("input[type='date']", date)
            page.fill("textarea", desc)
            print("📝 Form data injected.")

            # 2. Try Standard RPA (This WILL fail if Chaos Engine is active)
            print("🤖 Attempting Standard DOM execution (ID: #standard-submit-btn)...")
            standard_btn = page.locator("#standard-submit-btn")
            
            # Using a short timeout so we don't wait forever if it's broken
            standard_btn.click(timeout=3000)
            print("✅ CLICK SUCCESSFUL (Standard RPA)")
            
        except Exception as e:
            print("\n⚠️ STANDARD RPA FAILED: Element not found! DOM mutated by Chaos Engine.")
            print("🔄 INITIATING RULE #2: SEMANTIC ANCHORING VIA NOVA MULTIMODAL EMBEDDINGS...")
            
            # Semantic Fallback: Find the next best match using Vector Math
            buttons = page.locator("button, input[type='submit'], a").all()
            best_btn = None
            best_score = -1
            
            # Vectorize the AI's intended text
            target_intent_vec = get_aws_embedding(target_anchor) 
            
            for btn in buttons:
                text = btn.inner_text().strip() or btn.get_attribute("value") or ""
                if not text: continue
                
                # Vectorize the actual button text found on the broken page
                btn_vec = get_aws_embedding(text)
                score = calculate_cosine_similarity(target_intent_vec, btn_vec)
                
                print(f"   [Semantic Check] '{text}' -> Similarity: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_btn = btn
            
            # Click the highest matching vector
            if best_btn and best_score > 0.4:
                best_btn.click(timeout=3000)
                print(f"\n🚀 SEMANTIC ANCHORING SUCCESS: Clicked element with highest vector similarity ({best_score:.4f})")
            else:
                print("\n❌ SEMANTIC ANCHORING FAILED: No visually/semantically matching element found.")
        
        finally:
            page.wait_for_timeout(3000)
            browser.close()
            print("--- AUTOMATION ENGINE SHUTDOWN ---\n")

if __name__ == "__main__":
    # Test Payload
    execute_vla_action(
        tracking_id="AWB-7742",
        date="2026-03-14",
        desc="Severe crushing damage to outer box.",
        decision="DEPLOY",
        target_anchor="Submit Claim Validation"
    )