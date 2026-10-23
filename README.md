# JARVIS

A voice-first AI assistant for Apple-Silicon Macs. It listens (English and Hindi), thinks on your Mac with an MLX
model, speaks back, and doubles as a data-science analyst: upload a CSV or Excel file and ask for analysis, charts,
forecasts, machine-learning models, clusters, anomalies and significance tests.

## Run it

```bash
./start.sh
```

First run creates `.venv`, installs `backend/requirements.txt` and `npm install`s the interface (a few minutes plus
model downloads). Afterwards it starts the Python brain (`ws://localhost:8765`) and the interface
(http://localhost:5173). In VS Code you can also run **Terminal → Run Task → JARVIS: start everything**.

Use **Chrome** (it is needed for the microphone), click **Start JARVIS**, and allow the microphone.

Manual start: `python backend/server.py` in one terminal, `cd frontend/jarvis-ui && npm run dev` in another.
Python 3.9+ works; 3.10+ also enables picture understanding.

## What it does

| Area | Examples |
|---|---|
| Talk | "Jarvis, what's the weather in Delhi", "दिल्ली का मौसम कैसा है", "पाँच मिनट का टाइमर लगाओ" |
| Hands-free | Turn on *Hands-free*, then say "Jarvis …" (you get a beep and a ripple). Say "stop", "chup" or "bas karo" to interrupt |
| Hindi | Hindi / Hinglish understood; replies in Hindi with a natural Indian neural voice; English sentences use a British voice, and Jarvis switches by itself with the language you speak |
| Data | Upload a file → "analyse this", "show the dashboard", "forecast revenue", "clean the data", "export a report" |
| Machine learning | "predict Churn", "train a model to predict revenue", then "what if tenure is 3 and contract is Month" |
| Groups & outliers | "find natural groups", "find unusual rows" (both also give you a CSV) |
| Statistics | "is Sales different by Region", "is Churn related to contract" (picks t-test, ANOVA, correlation or chi-square) |
| Files, docs, vision | Ask questions about PDFs and documents; "what's on my screen" |

## Better Hinglish

In Hindi/Hinglish chats JARVIS is shown a few relevant example conversations (`backend/hinglish_pack.py`, 50+
built in, plus 119 more in `backend/hinglish_data/conversations.jsonl`) so his replies sound like a real friend texting. Add your own in `backend/hinglish_data/` (jsonl, csv or txt;
see the README there). This steers the model; it does not retrain it.

## Hearing

Chrome's built-in speech recognition (free, needs internet). Allow the microphone when asked. Use the **🌐** pill in the top bar
to set the listening language (AUTO / EN / हिं); pick हिं when you speak Hindi. Hands-free beeps the moment it hears "Jarvis".

## Command palette and themes

**Ctrl/Cmd + K** opens a search over every action (new chat, hands-free, language, tone, ML commands...).
The coloured dots in the top bar switch the colour theme.

## Settings

Copy `backend/.env.example` to `backend/.env`. Voice gender can also be changed live in the Controls
panel. The brain model is chosen from your RAM (Qwen3 4B / 8B / 14B / 30B-A3B); a cloud brain is optional.

## Layout

```
backend/    server.py (WebSocket) · main.py (router) · voice.py · hindi_cmd.py
            data_assistant.py · analytics.py · ml.py + ml_report.py · charts.py · reports.py · tests/
frontend/jarvis-ui/   React + Vite interface (src/App.jsx, mic.js, audio.js)
```

Tests: `cd backend && python -m pytest tests`.
