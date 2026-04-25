"""
app/daemon_thread.py — Background listener thread.

Chạy trong thread riêng: mic → voice auth → STT → OpenClaw.
Có stop() an toàn và callback cập nhật trạng thái tray.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _energy_rms(audio) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(audio ** 2) + 1e-10))


def _listen_chunk(duration: float):
    """Ghi âm từ microphone `duration` giây → numpy float32."""
    import sounddevice as sd
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    sd.wait()
    return audio.flatten()


class DaemonThread(threading.Thread):
    """
    Background daemon: mic → ECAPA-TDNN → Parakeet STT → OpenClaw.

    Parameters
    ----------
    config : ConfigManager
    openclaw_client : OpenClawClient
    on_status : callable(str)
        Callback cập nhật trạng thái cho tray icon.
    """

    def __init__(
        self,
        config,
        openclaw_client,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="VoiceAuthDaemon")
        self.config = config
        self.client = openclaw_client
        self.on_status = on_status or (lambda s: None)

        self._stop_event = threading.Event()
        self._verifier = None
        self._stt = None

    # ── Control ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Dừng daemon an toàn."""
        logger.info("[Daemon] Đang dừng...")
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info("[Daemon] Thread bắt đầu.")
        self.on_status("loading")

        try:
            self._load_components()
        except Exception as exc:
            logger.error(f"[Daemon] Lỗi khởi tạo: {exc}", exc_info=True)
            self.on_status("error")
            return

        self.on_status("listening")
        logger.info("[Daemon] ✅ Sẵn sàng lắng nghe.")

        auth_duration = float(self.config.get("max_auth_duration_sec", 4))
        silence_threshold = 0.005

        while not self._stop_event.is_set():
            try:
                # ── Ghi audio ────────────────────────────────────────────
                audio = _listen_chunk(auth_duration)

                if self._stop_event.is_set():
                    break

                # Bỏ silence
                if _energy_rms(audio) < silence_threshold:
                    continue

                self.on_status("processing")

                # ── Voice Auth ───────────────────────────────────────────
                passed, score = self._verifier.verify(audio, sample_rate=SAMPLE_RATE)

                if not passed:
                    logger.debug(f"[Daemon] Auth từ chối (score={score:.3f})")
                    self.on_status("listening")
                    continue

                logger.info(f"[Daemon] ✅ Auth pass | score={score:.2f}")

                # ── STT ──────────────────────────────────────────────────
                command = self._stt.transcribe(audio, sample_rate=SAMPLE_RATE)

                if not command:
                    logger.info("[Daemon] STT rỗng — bỏ qua.")
                    self.on_status("listening")
                    continue

                user = self.config.user
                logger.info(f"[Daemon] 💬 [{user}] score={score:.2f} | '{command}'")

                # ── Gửi đến OpenClaw ─────────────────────────────────────
                ok, response = self.client.send_prompt(command)
                if ok:
                    logger.info(f"[Daemon] 🤖 OpenClaw: {response[:150]}")
                else:
                    logger.warning(f"[Daemon] OpenClaw error: {response}")

                self.on_status("listening")
                time.sleep(0.5)

            except Exception as exc:
                logger.error(f"[Daemon] Lỗi trong vòng lặp: {exc}", exc_info=True)
                self.on_status("error")
                time.sleep(2)
                if not self._stop_event.is_set():
                    self.on_status("listening")

        logger.info("[Daemon] Thread kết thúc.")

    # ── Component loaders ────────────────────────────────────────────────

    def _load_components(self) -> None:
        """Load model ECAPA-TDNN + Parakeet (1 lần duy nhất)."""
        import sys
        from pathlib import Path

        # Add project root to sys.path
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        logger.info("[Daemon] Loading ECAPA-TDNN verifier...")
        from auth.verifier import VoiceVerifier
        self._verifier = VoiceVerifier(
            model_path=self.config.model_path,
            voiceprint_path=self.config.voiceprint_path,
            cohort_dir=self.config.cohort_dir,
            threshold_asnorm=self.config.threshold,
            log_path=self.config.log_path,
        )

        logger.info("[Daemon] Loading Parakeet STT...")
        from stt.parakeet_engine import ParakeetSTT
        self._stt = ParakeetSTT()

        logger.info("[Daemon] ✅ Components loaded.")
