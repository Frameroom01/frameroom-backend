# Frameroom Backend

Python/FastAPI backend for Frameroom real estate photo editor.
Handles lens correction, distortion, and perspective fixes.

## Endpoints

- `GET /` — health check
- `GET /health` — health check
- `POST /lens-correction` — apply manual lens correction
- `POST /auto-lens-correct` — auto-detect and apply lens correction

## Local Development

```bash
pip install -r requirements.txt
python main.py
```

Server runs at http://localhost:8000

## Deploy to Railway

1. Push this folder to a GitHub repository
2. Connect to Railway.app
3. Railway auto-detects Python and deploys

## Environment Variables

None required for Phase 5. Future phases will add:
- `ANTHROPIC_API_KEY` for AI features
- `SUPABASE_URL` for user storage
