# NovaClaim-X

**Vision-Language-Action (VLA) Logistics Agent** — An AI-powered damage assessment and claims automation system built on AWS Bedrock Nova 2.

## Overview

NovaClaim-X uses multimodal AI to analyze package damage in real time, generate agentic reasoning for operators, and automate the claims workflow through a Playwright-driven portal integration.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| AI / Vision | AWS Bedrock — Amazon Nova 2 Lite (vision) |
| Text-to-Speech | AWS Bedrock — Amazon Nova 2 Sonic |
| Embeddings | AWS Bedrock — Nova Multimodal Embeddings |
| Frontend | Tailwind CSS (Slate-900 dark-mode) |
| Automation | Playwright |
| Cloud SDK | boto3 (`us-east-1`) |

## Project Structure

```
NovaClaim-X/
├── backend/
│   └── app.py          # FastAPI server — vision + voice endpoints
├── dashboard/
│   └── index.html      # Operator dashboard UI
├── target_portal/
│   └── portal.html     # Simulated claims portal for Playwright automation
├── requirements.txt
└── README.md
```

## Key Features

- **Damage Analysis** — Uploads a package image to Nova 2 Lite; returns a structured damage summary (Crush / Puncture / Leak classification).
- **Agentic Reasoning** — Every backend response includes an `agentic_reasoning` field explaining what the AI is seeing and deciding.
- **Voice Briefing** — Converts reasoning text to audio via Nova 2 Sonic for hands-free operator alerts.
- **Semantic Anchoring** — If a portal automation step fails, Nova Multimodal Embeddings find the next best visual match instead of relying on brittle DOM selectors.

## Getting Started

### Prerequisites

- Python 3.10+
- AWS account with Bedrock access enabled in `us-east-1`
- AWS credentials configured (`aws configure`)

### Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

### Run the Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

Then open `dashboard/index.html` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/analyze-damage` | Upload an image; returns damage classification + reasoning |
| POST | `/generate-voice` | Converts text to audio via Nova Sonic |

## License

MIT
