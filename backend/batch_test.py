import requests
import time
import random
import os

# Ensure you have sample images in a 'test_assets' folder before running
EVIDENCE_IMG = "test_assets/crushed_box.jpg" 
UI_IMG = "test_assets/portal_screenshot.jpg"
API_URL = "http://localhost:8000/vla/execute"

def run_data_battery(iterations=50):
    print(f"🔋 STARTING DATA BATTERY: {iterations} ITERATIONS...")
    
    if not os.path.exists(EVIDENCE_IMG) or not os.path.exists(UI_IMG):
        print(f"Error: Please place sample images at {EVIDENCE_IMG} and {UI_IMG} first.")
        return

    for i in range(iterations):
        # Simulate varying levels of UI destruction
        chaos_level = random.choice([0, 25, 50, 75, 100]) 
        
        payload = {
            "claim_json": '{"tracking_id": "AWB-7742", "damage_reported": "Severe Crushing"}',
            "chaos_level": str(chaos_level)
        }
        
        with open(EVIDENCE_IMG, "rb") as ev_file, open(UI_IMG, "rb") as ui_file:
            files = {
                "evidence_photo": ("ev.jpg", ev_file, "image/jpeg"),
                "ui_screenshot": ("ui.jpg", ui_file, "image/jpeg")
            }
            
            try:
                print(f"[{i+1}/{iterations}] Testing Chaos Level {chaos_level}%...")
                response = requests.post(API_URL, data=payload, files=files)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   -> RTD: {data['telemetry']['rtd_ms']}ms | MF-Score: {data['telemetry']['mf_score']}")
                else:
                    print(f"   -> HTTP Error: {response.status_code}")
            except Exception as e:
                print(f"   -> Connection Failed: {e}")
                
        # Small delay to prevent AWS rate limiting
        time.sleep(2)

    print("🏁 DATA BATTERY COMPLETE. Check research_metrics.csv.")

if __name__ == "__main__":
    run_data_battery(500) # Ready to generate 500 rows for the Q1 Journal