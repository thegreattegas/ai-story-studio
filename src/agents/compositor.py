"""CompositorAgent — Phase 4.

Assembles scene images + voice narration + subtitles into a final MP4 video
using MoviePy (which wraps FFmpeg).

Design
------
* Always runs real FFmpeg — there are no external API costs for composition.
* Images are letterboxed onto a 1920 x 1080 black canvas.
* Audio duration is the ground truth for total video length.
* If actual audio is shorter than 1 second (e.g. mock stub), scene durations
  fall back to ``scene.estimated_seconds`` so the video stays watchable.
* Subtitle burning uses a TextClip per segment; gracefully disabled if the
  system font is unavailable or MoviePy raises an error.

Output
------
* ``workspace/final.mp4`` — playable 1920 x 1080 MP4, libx264 + AAC.
* ``state.final_video_path`` set to the absolute path of the output file.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from src.agents.base import AgentResult, BaseAgent
from src.state import StoryState

log = logging.getLogger(__name__)

# Font used for subtitle TextClip — Arial is available on all Windows machines.
_SUBTITLE_FONT = "C:/Windows/Fonts/arial.ttf"
_FALLBACK_FONTS = [
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdana.ttf",
]


def _find_font() -> str | None:
    """Return the first available subtitle font path, or None if none found."""
    for font_path in [_SUBTITLE_FONT] + _FALLBACK_FONTS:
        if Path(font_path).exists():
            return font_path
    return None


class CompositorAgent(BaseAgent):
    """Composes scene images + voice + subtitles into workspace/final.mp4.

    Unlike other agents this one always runs real FFmpeg — no API quota is
    consumed.  The ``mock_response`` stub is kept for BaseAgent compliance.
    """

    name: str = "Compositor"
    default_model: str = "moviepy+ffmpeg"
    system_prompt: str = ""

    OUTPUT_RESOLUTION: tuple[int, int] = (1920, 1080)
    FPS: int = 24

    def __init__(self) -> None:
        super().__init__()
        self.output_path: Path = self.config.workspace_dir / "final.mp4"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def mock_response(self, state: StoryState) -> dict[str, Any]:
        """Stub — compositor always runs real FFmpeg; mock_response unused."""
        return {"video_path": str(self.output_path), "mocked": True}

    async def run(self, state: StoryState) -> AgentResult:
        """Compose the video and write workspace/final.mp4.

        Args:
            state: Pipeline state with ``scenes`` (image_path set),
                   ``voice_path``, and optionally ``subtitle_path``.

        Returns:
            :class:`AgentResult` with file size and duration.

        Raises:
            FileNotFoundError: If any required input file is missing.
        """
        start = time.monotonic()
        log.info(
            "CompositorAgent: START — composing %d scenes into video",
            len(state.scenes),
        )

        # Validate scene images.
        for scene in state.scenes:
            if not scene.image_path:
                raise FileNotFoundError(
                    f"CompositorAgent: scene {scene.id} has no image_path set."
                )
            img_full = self.config.workspace_dir / scene.image_path
            if not img_full.exists():
                raise FileNotFoundError(
                    f"CompositorAgent: scene {scene.id} image missing: {img_full}"
                )

        # Validate voice file.
        if not state.voice_path:
            raise FileNotFoundError("CompositorAgent: state.voice_path is not set.")
        voice_full = self.config.workspace_dir / state.voice_path
        if not voice_full.exists():
            raise FileNotFoundError(
                f"CompositorAgent: voice file missing: {voice_full}"
            )

        result = await asyncio.to_thread(self._compose_sync, state)

        state.final_video_path = str(self.output_path)
        elapsed = time.monotonic() - start

        log.info(
            "CompositorAgent: DONE in %.2fs — %s (%s KB)",
            elapsed,
            self.output_path.name,
            f"{result['file_size_bytes'] // 1024:,}",
        )

        return AgentResult(
            agent_name=self.name,
            model_used=self.default_model,
            output=result,
            cost_usd=0.0,
            elapsed_sec=elapsed,
            mocked=False,
        )

    # ------------------------------------------------------------------
    # Synchronous composition (runs in thread pool)
    # ------------------------------------------------------------------

    def _compose_sync(self, state: StoryState) -> dict[str, Any]:
        """Build and write the MP4 using MoviePy 2.x API.

        Args:
            state: Pipeline state — read-only inside this thread.

        Returns:
            Dict with ``video_path``, ``duration_sec``, ``file_size_bytes``.
        """
        from moviepy import (  # noqa: PLC0415
            AudioFileClip,
            VideoClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        voice_full = self.config.workspace_dir / state.voice_path  # type: ignore[arg-type]
        audio = AudioFileClip(str(voice_full))
        audio_duration: float = audio.duration

        log.info("CompositorAgent: audio duration=%.3fs", audio_duration)

        # Use voice_segment timing when available and audio is long enough.
        # Fall back to estimated_seconds for mock/stub audio.
        use_estimated = audio_duration < 1.0

        scene_clips = []
        scenes_sorted = sorted(state.scenes, key=lambda s: s.id)
        w, h = self.OUTPUT_RESOLUTION

        for scene in scenes_sorted:
            img_full = str(self.config.workspace_dir / scene.image_path)  # type: ignore[arg-type]

            if use_estimated or scene.voice_segment_start is None:
                duration = float(scene.estimated_seconds)
            else:
                duration = scene.voice_segment_end - scene.voice_segment_start  # type: ignore[operator]

            duration = max(duration, 2.0 / self.FPS)  # never less than 2 frames

            # Prefer AI-generated video clip; fall back to Ken Burns.
            if scene.video_path:
                video_full = str(self.config.workspace_dir / scene.video_path)
                try:
                    raw_clip = VideoFileClip(video_full)
                    clip_duration = min(duration, raw_clip.duration)
                    scene_clip = raw_clip.subclipped(0, clip_duration).resized(height=h)
                    log.info(
                        "CompositorAgent: scene %02d — %.2fs from video clip %s",
                        scene.id, clip_duration, Path(video_full).name,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "CompositorAgent: scene %d video clip failed (%s) — Ken Burns fallback.",
                        scene.id, exc,
                    )
                    scene_clip = self._make_ken_burns_clip(img_full, duration, scene.mood, (w, h))
            else:
                scene_clip = self._make_ken_burns_clip(img_full, duration, scene.mood, (w, h))
                log.info(
                    "CompositorAgent: scene %02d — %.2fs Ken Burns from %s",
                    scene.id, duration, Path(img_full).name,
                )
            scene_clips.append(scene_clip)

        # Concatenate scenes.
        # When Veo 3 clips are used, keep their ambient audio in the temp file
        # so FFmpeg can mix it with ElevenLabs later.
        has_ambient = getattr(state, "has_ambient_audio", False)
        video = concatenate_videoclips(scene_clips, method="compose")

        # For non-ambient path, attach ElevenLabs audio now for subtitle burns.
        if not has_ambient:
            video = video.with_audio(audio)

        # Burn subtitles (optional — falls back gracefully).
        if state.subtitle_path:
            srt_full = self.config.workspace_dir / state.subtitle_path
            if srt_full.exists():
                try:
                    video = self._burn_subtitles(video, srt_full)
                    log.info("CompositorAgent: subtitles burned in.")
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "CompositorAgent: subtitle burning failed (%s) — proceeding without.",
                        exc,
                    )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video = self.output_path.with_name("_temp_video_only.mp4")

        if has_ambient:
            # Write video WITH Veo 3 ambient audio so FFmpeg can amix it.
            log.info("CompositorAgent: encoding video+ambient to %s ...", temp_video.name)
            video.write_videofile(
                str(temp_video),
                fps=self.FPS,
                codec="libx264",
                audio_codec="aac",
                preset="fast",
                ffmpeg_params=["-crf", "23"],
                logger=None,
            )
        else:
            # No ambient — write video-only, simple mux below.
            log.info("CompositorAgent: encoding video-only to %s ...", temp_video.name)
            video_no_audio = video.without_audio() if video.audio else video
            video_no_audio.write_videofile(
                str(temp_video),
                fps=self.FPS,
                codec="libx264",
                audio=False,
                preset="fast",
                ffmpeg_params=["-crf", "23"],
                logger=None,
            )

        # Cleanup MoviePy objects before FFmpeg step.
        audio.close()
        video.close()
        for clip in scene_clips:
            clip.close()

        log.info("CompositorAgent: muxing audio into %s ...", self.output_path.name)
        if has_ambient:
            # Mix Veo 3 ambient (20% vol) + ElevenLabs narration (100% vol).
            self._mix_ambient_and_voice(temp_video, voice_full, self.output_path)
        else:
            self._mux_audio(temp_video, voice_full, self.output_path)

        if temp_video.exists():
            temp_video.unlink()

        file_size = self.output_path.stat().st_size
        return {
            "video_path": str(self.output_path),
            "duration_sec": audio_duration,
            "file_size_bytes": file_size,
            "resolution": list(self.OUTPUT_RESOLUTION),
        }

    def _mux_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> None:
        """Mux a video file and an audio file into a single MP4 using FFmpeg directly.

        Using imageio-ffmpeg's bundled binary avoids PATH issues and is
        more reliable than MoviePy's internal audio handling on Windows.

        Args:
            video_path:  Video-only MP4 produced by MoviePy.
            audio_path:  Source audio file (MP3 / WAV).
            output_path: Destination MP4 path.
        """
        import subprocess  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio mux failed (exit {result.returncode}):\n{result.stderr[-800:]}"
            )

    def _mix_ambient_and_voice(
        self, video_path: Path, voice_path: Path, output_path: Path
    ) -> None:
        """Mix Veo 3 ambient audio (20% vol) with ElevenLabs narration (100% vol).

        Uses FFmpeg amix filter so nature sounds play softly under the narrator.

        Args:
            video_path:  MP4 with Veo 3 video + ambient audio track.
            voice_path:  ElevenLabs narration MP3.
            output_path: Final MP4 destination.
        """
        import subprocess  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),    # stream 0: video + ambient audio
            "-i", str(voice_path),    # stream 1: ElevenLabs narration
            "-filter_complex",
            "[0:a]volume=0.20[amb];[1:a]volume=1.0[voice];[amb][voice]amix=inputs=2:duration=first[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg amix failed (exit {result.returncode}):\n{result.stderr[-800:]}"
            )

    # Ken Burns effect moods → animation style
    _KB_EFFECTS: dict[str, str] = {
        "tense":       "zoom_in",
        "sad":         "zoom_in",
        "mysterious":  "zoom_in",
        "determined":  "zoom_in",
        "triumphant":  "zoom_out",
        "peaceful":    "zoom_out",
        "hopeful":     "zoom_out",
        "joyful":      "pan_right",
        "exciting":    "pan_right",
        "grateful":    "pan_left",
        "curious":     "pan_left",
    }

    def _make_ken_burns_clip(
        self,
        img_path: str,
        duration: float,
        mood: str,
        output_size: tuple[int, int],
    ) -> Any:
        """Return a VideoClip with a subtle Ken Burns zoom/pan animation.

        Uses FFmpeg's native ``zoompan`` filter for fast C-side rendering.
        Falls back to the Python frame-by-frame implementation on any FFmpeg
        error (e.g. missing binary or unsupported filter).

        The image is letterboxed onto a black canvas at ``output_size``, then a
        slow zoom or pan is applied.  Effect is ~12% zoom / ~8% pan over the
        clip duration — noticeable but not jarring.

        Args:
            img_path:    Absolute path to the source image.
            duration:    Clip length in seconds.
            mood:        Scene mood — selects zoom_in / zoom_out / pan_left / pan_right.
            output_size: ``(width, height)`` of the output frame (1920×1080).

        Returns:
            ``VideoFileClip`` (FFmpeg path) or ``VideoClip`` (Python fallback).
        """
        import subprocess  # noqa: PLC0415
        import imageio_ffmpeg  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from moviepy import VideoFileClip  # noqa: PLC0415

        ow, oh = output_size
        effect = self._KB_EFFECTS.get(mood, "zoom_in")

        # --- Letterbox image once in PIL (single resize, not per-frame) ---
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        scale = min(ow / iw, oh / ih)
        fit_w, fit_h = int(iw * scale), int(ih * scale)
        img_fit = img.resize((fit_w, fit_h), Image.LANCZOS)
        canvas = Image.new("RGB", (ow, oh), (0, 0, 0))
        canvas.paste(img_fit, ((ow - fit_w) // 2, (oh - fit_h) // 2))

        tmp_dir = self.config.workspace_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem
        temp_img = tmp_dir / f"{stem}_lb.png"
        temp_out = tmp_dir / f"{stem}_kb.mp4"
        canvas.save(str(temp_img))

        d_frames = max(1, int(duration * self.FPS))

        # zoompan expressions — on=current frame (0-based), d=total frames
        if effect == "zoom_in":
            z_expr = "1+0.12*on/d"
            x_expr = "iw/2-iw/zoom/2"
            y_expr = "ih/2-ih/zoom/2"
        elif effect == "zoom_out":
            z_expr = "1.12-0.12*on/d"
            x_expr = "iw/2-iw/zoom/2"
            y_expr = "ih/2-ih/zoom/2"
        elif effect == "pan_right":
            z_expr = "1.10"
            x_expr = "(0.46+0.08*on/d)*iw-iw/zoom/2"
            y_expr = "ih/2-ih/zoom/2"
        else:  # pan_left
            z_expr = "1.10"
            x_expr = "(0.54-0.08*on/d)*iw-iw/zoom/2"
            y_expr = "ih/2-ih/zoom/2"

        vf = (
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={d_frames}:s={ow}x{oh}:fps={self.FPS}"
        )

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-loop", "1",
            "-i", str(temp_img),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(temp_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and temp_out.exists():
            log.debug(
                "CompositorAgent: FFmpeg zoompan OK for %s (effect=%s, %.2fs)",
                stem, effect, duration,
            )
            return VideoFileClip(str(temp_out))

        # --- Fallback: Python frame-by-frame (slower but always works) ---
        log.warning(
            "CompositorAgent: FFmpeg zoompan failed for %s — using Python fallback. stderr: %s",
            stem, proc.stderr[-300:],
        )
        return self._make_ken_burns_clip_python(img_path, duration, mood, output_size)

    def _make_ken_burns_clip_python(
        self,
        img_path: str,
        duration: float,
        mood: str,
        output_size: tuple[int, int],
    ) -> Any:
        """Python frame-by-frame Ken Burns fallback (original implementation)."""
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from moviepy import VideoClip  # noqa: PLC0415

        ow, oh = output_size
        effect = self._KB_EFFECTS.get(mood, "zoom_in")

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        scale = min(ow / iw, oh / ih)
        fit_w, fit_h = int(iw * scale), int(ih * scale)
        img_fit = img.resize((fit_w, fit_h), Image.LANCZOS)
        canvas = Image.new("RGB", (ow, oh), (0, 0, 0))
        canvas.paste(img_fit, ((ow - fit_w) // 2, (oh - fit_h) // 2))
        base = np.array(canvas)

        zoom_lo, zoom_hi = 1.00, 1.12
        pan_range = 0.08

        def make_frame(t: float) -> np.ndarray:
            p = min(t / duration, 1.0) if duration > 0 else 0.0
            if effect == "zoom_in":
                zoom = zoom_lo + (zoom_hi - zoom_lo) * p
                cx, cy = ow * 0.5, oh * 0.5
            elif effect == "zoom_out":
                zoom = zoom_hi - (zoom_hi - zoom_lo) * p
                cx, cy = ow * 0.5, oh * 0.5
            elif effect == "pan_right":
                zoom = 1.10
                cx = ow * (0.46 + pan_range * p)
                cy = oh * 0.5
            else:
                zoom = 1.10
                cx = ow * (0.54 - pan_range * p)
                cy = oh * 0.5

            crop_w = ow / zoom
            crop_h = oh / zoom
            x1 = int(max(cx - crop_w / 2, 0))
            y1 = int(max(cy - crop_h / 2, 0))
            x2 = int(min(cx + crop_w / 2, ow))
            y2 = int(min(cy + crop_h / 2, oh))
            crop = base[y1:y2, x1:x2]
            import numpy as _np  # noqa: PLC0415
            from PIL import Image as _Image  # noqa: PLC0415
            frame = _Image.fromarray(crop).resize((ow, oh), _Image.LANCZOS)
            return _np.array(frame)

        return VideoClip(make_frame, duration=duration).with_fps(self.FPS)

    def _add_speech_bubble(
        self, img_path: str, dialogue: str, speaker: str
    ) -> str:
        """Overlay a Pillow-rendered speech bubble onto the image.

        Draws a white rounded-rectangle bubble with a black border in the
        upper-right corner of the image, with a triangular tail pointing
        down-left toward the character area.  The dialogue text is word-
        wrapped inside the bubble using Arial.

        Args:
            img_path:  Absolute path to the source image file.
            dialogue:  The character's spoken line (attribution already stripped).
            speaker:   Speaker name shown in smaller text above the line.

        Returns:
            Absolute path to the processed image (a temp file in workspace/tmp/).
        """
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
        except ImportError:
            log.warning("CompositorAgent: Pillow not installed — skipping speech bubble.")
            return img_path

        try:
            img = Image.open(img_path).convert("RGBA")
            iw, ih = img.size

            # ---- font setup ----
            font_path = _find_font() or "arial.ttf"
            try:
                font_speaker = ImageFont.truetype(font_path, size=max(18, ih // 55))
                font_dialogue = ImageFont.truetype(font_path, size=max(22, ih // 42))
            except OSError:
                font_speaker = ImageFont.load_default()
                font_dialogue = ImageFont.load_default()

            # ---- word-wrap dialogue text ----
            bubble_max_w = int(iw * 0.42)   # bubble occupies up to 42% of image width
            padding = int(iw * 0.018)

            def wrap_text(text: str, font: Any, max_width: int) -> list[str]:
                words = text.split()
                lines: list[str] = []
                current = ""
                dummy = Image.new("RGB", (1, 1))
                dc = ImageDraw.Draw(dummy)
                for word in words:
                    test = (current + " " + word).strip()
                    bbox = dc.textbbox((0, 0), test, font=font)
                    if bbox[2] - bbox[0] <= max_width:
                        current = test
                    else:
                        if current:
                            lines.append(current)
                        current = word
                if current:
                    lines.append(current)
                return lines or [""]

            dialogue_lines = wrap_text(dialogue, font_dialogue, bubble_max_w - 2 * padding)

            # ---- measure bubble dimensions ----
            dummy = Image.new("RGB", (1, 1))
            dc = ImageDraw.Draw(dummy)

            line_h = dc.textbbox((0, 0), "Ag", font=font_dialogue)[3] + 4
            speaker_h = (dc.textbbox((0, 0), speaker, font=font_speaker)[3] + 6) if speaker else 0
            text_h = speaker_h + len(dialogue_lines) * line_h
            text_w = max(
                (dc.textbbox((0, 0), ln, font=font_dialogue)[2] for ln in dialogue_lines),
                default=bubble_max_w,
            )
            if speaker:
                spk_w = dc.textbbox((0, 0), speaker, font=font_speaker)[2]
                text_w = max(text_w, spk_w)

            bw = min(text_w + 2 * padding, bubble_max_w)
            bh = text_h + 2 * padding

            # ---- position: upper-right quadrant, with margin ----
            margin = int(iw * 0.025)
            tail_h = int(ih * 0.045)
            bx = iw - bw - margin
            by = margin

            # ---- draw on an RGBA overlay ----
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            radius = int(min(bw, bh) * 0.15)

            # Bubble background (white, semi-opaque)
            draw.rounded_rectangle(
                [bx, by, bx + bw, by + bh],
                radius=radius,
                fill=(255, 255, 255, 230),
                outline=(30, 30, 30, 255),
                width=max(2, iw // 400),
            )

            # Tail triangle (pointing down-left from bottom-left of bubble)
            tail_tip_x = bx - int(iw * 0.04)
            tail_tip_y = by + bh + tail_h
            tail_pts = [
                (bx + radius, by + bh),
                (bx + radius + int(bw * 0.18), by + bh),
                (tail_tip_x, tail_tip_y),
            ]
            draw.polygon(tail_pts, fill=(255, 255, 255, 230))
            draw.line(
                [tail_pts[0], tail_pts[2], tail_pts[1]],
                fill=(30, 30, 30, 255),
                width=max(2, iw // 400),
            )

            # Speaker name
            ty = by + padding
            if speaker:
                draw.text(
                    (bx + padding, ty),
                    speaker,
                    font=font_speaker,
                    fill=(80, 80, 180, 255),
                )
                ty += speaker_h

            # Dialogue lines
            for line in dialogue_lines:
                draw.text((bx + padding, ty), line, font=font_dialogue, fill=(20, 20, 20, 255))
                ty += line_h

            # Composite bubble over image
            result = Image.alpha_composite(img, overlay).convert("RGB")

            # Save to temp file
            tmp_dir = self.config.workspace_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            src_name = Path(img_path).stem
            out_path = tmp_dir / f"{src_name}_bubble.jpg"
            result.save(str(out_path), "JPEG", quality=92)
            return str(out_path)

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "CompositorAgent: speech bubble failed (%s) — using original image.", exc
            )
            return img_path

    def _burn_subtitles(self, video: Any, srt_path: Path) -> Any:
        """Overlay SRT subtitle text clips onto ``video``.

        Uses one TextClip per segment positioned near the bottom of the frame.
        Requires a TTF font on the system; raises on failure so the caller
        can fall back to no subtitles.

        Args:
            video:    The base video clip.
            srt_path: Absolute path to the .srt file.

        Returns:
            ``CompositeVideoClip`` with subtitle overlays.
        """
        from moviepy import CompositeVideoClip, TextClip  # noqa: PLC0415

        font = _find_font()
        if font is None:
            raise RuntimeError("No suitable TTF font found for subtitle rendering.")

        segments = self._parse_srt(srt_path)
        text_clips: list[Any] = []

        for seg in segments:
            seg_duration = seg["end"] - seg["start"]
            if seg_duration <= 0:
                continue
            txt = (
                TextClip(
                    font=font,
                    text=seg["text"],
                    font_size=44,
                    color="white",
                    stroke_color="black",
                    stroke_width=3,
                    bg_color=(0, 0, 0, 160),
                    method="caption",
                    size=(1600, None),
                    duration=seg_duration,
                )
                .with_position(("center", 880))
                .with_start(seg["start"])
            )
            text_clips.append(txt)

        if not text_clips:
            return video

        result = CompositeVideoClip([video] + text_clips)
        # Explicitly carry audio through — CompositeVideoClip may drop it.
        if video.audio is not None:
            result = result.with_audio(video.audio)
        return result

    # ------------------------------------------------------------------
    # SRT parsing helpers
    # ------------------------------------------------------------------

    def _parse_srt(self, srt_path: Path) -> list[dict[str, Any]]:
        """Parse an SRT file into a list of ``{start, end, text}`` dicts.

        Args:
            srt_path: Absolute path to the .srt subtitle file.

        Returns:
            List of segment dicts ordered by ``start`` time.
        """
        content = srt_path.read_text(encoding="utf-8")
        segments: list[dict[str, Any]] = []

        for block in content.strip().split("\n\n"):
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            timing_line = lines[1]
            if " --> " not in timing_line:
                continue
            start_str, end_str = timing_line.split(" --> ", 1)
            text = " ".join(lines[2:])
            segments.append(
                {
                    "start": self._srt_time_to_seconds(start_str.strip()),
                    "end": self._srt_time_to_seconds(end_str.strip()),
                    "text": text,
                }
            )

        return segments

    @staticmethod
    def _srt_time_to_seconds(srt_time: str) -> float:
        """Convert ``HH:MM:SS,mmm`` SRT timestamp to float seconds.

        Args:
            srt_time: Timestamp string in SRT format.

        Returns:
            Offset in seconds.
        """
        time_part, ms_part = srt_time.split(",")
        h, m, s = time_part.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms_part) / 1000
