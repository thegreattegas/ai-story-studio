# AI Story Studio — Claude Code Project Context

## Project Goal

Build a multi-agent generative AI system that takes a single text prompt and produces
a complete fairy tale video with: AI-written story, AI-generated illustrations,
AI-generated voice narration, synchronized subtitles, and a final composed mp4.

Built for a Master's degree Generative AI course portfolio.

---

## Architecture

```
USER PROMPT (e.g., "сказка про волка и лису")
                ↓
🧠 STORY WRITER AGENT (Sonnet)
   → produces JSON: {title, summary, scenes: [{narration, image_prompt, duration_sec}]}
                ↓
🎬 SCENE DIRECTOR AGENT (Sonnet)
   → enriches each scene with: detailed visual prompt, voice tone, character refs
                ↓
   ┌────────────────┴────────────────┐
   ↓ (parallel via asyncio.gather)   ↓
🎨 IMAGE AGENT          🗣️ VOICE AGENT
(Nano Banana × N)       (ElevenLabs, single voice)
   → workspace/images/  → workspace/voice.mp3
                ↓
📝 SUBTITLE AGENT (Whisper or timing heuristic)
   → workspace/subtitles.srt
                ↓
🎞️ COMPOSITOR AGENT (MoviePy + FFmpeg)
   → workspace/final.mp4
                ↓
👁️ REVIEWER AGENT (Opus)
   → APPROVED or NEEDS_FIXES (max 2 retries)
                ↓
✅ Final mp4 ready
```

---

## Tech Stack

| Layer | Library |
|-------|---------|
| LLM orchestration | `anthropic` (Claude API) |
| Image generation | `google-generativeai` (Nano Banana) |
| Voice TTS | `elevenlabs` |
| Subtitle timing | `openai` (Whisper) |
| Video composition | `moviepy` |
| Agent graph | `langgraph` |
| UI | `streamlit` |
| Config/state | `pydantic` |
| Env vars | `python-dotenv` |
| CLI logging | `rich` |

---

## Phase Status

- [x] Phase 1 — Project skeleton + core infrastructure (config, state, cache, router, base agent)
- [ ] Phase 2 — StoryWriter + SceneDirector agents (Anthropic only)
- [ ] Phase 3 — ImageAgent (Nano Banana) + VoiceAgent (ElevenLabs), parallel execution
- [ ] Phase 4 — SubtitleAgent (Whisper) + Compositor (MoviePy)
- [ ] Phase 5 — LangGraph wiring + ReviewerAgent (Opus, max 2 retries)
- [ ] Phase 6 — Streamlit UI with live progress streaming

---

## Key Conventions

### Mock Mode
Every agent checks `config.mock_mode` first. When `MOCK_MODE=true` in `.env`,
no real API calls are made. Agents return canned responses from `mock_response()`.
Use mock mode during development — real budget is ~$5.

### File-Based Caching
`src/cache.py` provides `cached_call(prompt, model, fn)`.
Cache key = `SHA-256(prompt + model)`. Results stored in `cache.json`.
Same prompt + model = cached response, zero cost.

### State Object
All agents communicate via `StoryState` (defined in `src/state.py`).
Agents receive the full state, return an `AgentResult`, and the graph
merges outputs back into state. Never pass raw strings between agents.

### Cost Tracking
Every `AgentResult` includes `tokens_in`, `tokens_out`, `cost_usd`.
`StoryState.total_cost` accumulates across agents.
Log warning if `total_cost > $1.50` per run.

Log format per agent:
```
[AgentName] [model-id] tokens_in=X tokens_out=Y cost=$Z elapsed=Ts
```

### Model Routing
```
Trivial → claude-haiku-4-5      ($0.25/$1.25 per 1M)
Medium  → claude-sonnet-4-6     ($3/$15 per 1M)
Complex → claude-opus-4-7       ($15/$75 per 1M)
```
Story Writer always forces Sonnet (creative work requires it).
Reviewer always forces Opus (quality gate).

### Workspace Sandboxing
All file I/O goes through `src/tools/file_tools.py`.
Paths are sandboxed to `workspace/` — no path traversal allowed.

---

## Folder Structure

```
ai-story-studio/
├── src/
│   ├── config.py          # AppConfig singleton, pricing, cost helper
│   ├── state.py           # StoryState + Scene pydantic models
│   ├── cache.py           # SHA-256 keyed disk cache
│   ├── router.py          # ModelRouter: task → model selection
│   ├── main.py            # CLI entry point
│   ├── graph.py           # LangGraph wiring (Phase 5)
│   ├── ui.py              # Streamlit UI (Phase 6)
│   ├── agents/
│   │   ├── base.py        # BaseAgent ABC + AgentResult
│   │   ├── story_writer.py
│   │   ├── scene_director.py
│   │   ├── image_agent.py
│   │   ├── voice_agent.py
│   │   ├── subtitle_agent.py
│   │   ├── compositor.py
│   │   └── reviewer.py
│   ├── providers/
│   │   ├── anthropic_provider.py
│   │   ├── google_provider.py
│   │   ├── elevenlabs_provider.py
│   │   └── whisper_provider.py
│   └── tools/
│       └── file_tools.py
├── workspace/             # agent outputs (gitignored except .gitkeep)
└── tests/
    └── test_router.py
```

---

## Running the Project

```bash
# Install base + dev deps
pip install -e ".[dev]"

# Install Phase 3 media deps (when ready)
pip install -e ".[media]"

# Verify config
python -c "from src.config import get_config; print(get_config())"

# Run tests
pytest tests/ -v

# Run CLI
python -m src.main

# Run UI (Phase 6)
streamlit run src/ui.py
```

---

## Environment Variables

See `.env.example`. Copy to `.env` and fill in keys.
The only required key for Phase 1-2 is `ANTHROPIC_API_KEY`.
Set `MOCK_MODE=true` during development.
