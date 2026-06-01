# AI Story Studio

> Multi-agent generative AI system for fairy tale video creation.
> Built for a Master's degree Generative AI course portfolio.

**Status: Work in Progress — Phase 1 complete**

---

## What It Does

Takes a single text prompt (any language) and produces a complete fairy tale video:

1. **Story Writer Agent** — generates structured story with scenes
2. **Scene Director Agent** — enriches scenes with visual/audio direction
3. **Image Agent** — generates illustrations via Nano Banana (Google)
4. **Voice Agent** — narrates story via ElevenLabs TTS
5. **Subtitle Agent** — generates synced subtitles via Whisper
6. **Compositor Agent** — assembles final `.mp4` via MoviePy
7. **Reviewer Agent** — quality-gates output, retries if needed

---

## Phase Checklist

- [x] Phase 1 — Skeleton: config, state, cache, model router, base agent
- [ ] Phase 2 — StoryWriter + SceneDirector (Anthropic Claude)
- [ ] Phase 3 — ImageAgent (Nano Banana) + VoiceAgent (ElevenLabs)
- [ ] Phase 4 — SubtitleAgent (Whisper) + Compositor (MoviePy)
- [ ] Phase 5 — LangGraph orchestration + ReviewerAgent
- [ ] Phase 6 — Streamlit live UI

---

## Quickstart

```bash
# 1. Clone and enter project
cd ai-story-studio

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY, leave MOCK_MODE=true

# 5. Verify setup
python -m src.main

# 6. Run tests
pytest tests/ -v
```

---

## Tech Stack

- **LLM**: Anthropic Claude (Haiku / Sonnet / Opus routing)
- **Images**: Google Generative AI (Nano Banana) — Phase 3
- **Voice**: ElevenLabs — Phase 3
- **Subtitles**: OpenAI Whisper — Phase 4
- **Video**: MoviePy — Phase 4
- **Orchestration**: LangGraph — Phase 5
- **UI**: Streamlit — Phase 6

---

## Cost Controls

- `MOCK_MODE=true` in `.env` — zero API cost during development
- File-based SHA-256 cache — repeated prompts cost nothing
- Per-agent token + cost logging
- Run-level warning if total cost exceeds $1.50
