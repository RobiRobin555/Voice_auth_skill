"""
voice_auth_daemon.py — Daemon chính của OpenClaw Voice Auth Skill.

Pipeline 3 giai đoạn:
  Microphone → [Wake Word] → Voice Auth (ECAPA-TDNN + AS-Norm)
            → STT (Parakeet-TDT-0.6B-v3) → OpenClaw

Chạy:
    python voice_auth_daemon.py
    python voice_auth_daemon.py --config path/to/config.yaml
    python voice_auth_daemon.py --once          # Test 1 lần rồi thoát
    python voice_auth_daemon.py --no-wake-word  # Bỏ qua wake word
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import time

import numpy as np
import yaml

# ── Thêm thư mục SKILL vào sys.path để import hoạt động
#    dù OpenClaw gọi từ bất kỳ working directory nào ──────────────────────
SKILL_DIR = Path(__file__).resolve().parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))


# ---------------------------------------------------------------------------
# Config loader  (resolve path tương đối từ SKILL_DIR)
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    """Nếu path là tương đối → resolve từ SKILL_DIR, không phải CWD."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(SKILL_DIR / p)


def _load_config(config_path: str = "config.yaml") -> dict:
    full = _resolve(config_path)
    try:
        with open(full, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cfg or {}
    except FileNotFoundError:
        logging.warning(f"[Config] Không tìm thấy '{full}' — dùng cấu hình mặc định.")
        return {}


def _get(cfg: dict, *keys: str, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_path: str) -> None:
    log_dir = Path(_resolve(log_path)).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] %(levelname)-7s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(
            logging.FileHandler(log_dir / "daemon.log", encoding="utf-8")
        )
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        handlers=handlers, force=True)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def listen_chunk(duration: float) -> np.ndarray:
    """Ghi âm từ microphone `duration` giây → numpy float32."""
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("❌  pip install sounddevice")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def energy_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2) + 1e-10))


# ---------------------------------------------------------------------------
# Wake Word (optional)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    def __init__(self, engine: str = "openwakeword", keyword: str = "hey openclaw") -> None:
        self.engine = engine
        self.keyword = keyword
        self._model = None
        self._load()

    def _load(self) -> None:
        if self.engine == "openwakeword":
            try:
                from openwakeword.model import Model  # type: ignore
                self._model = Model(wakeword_models=["hey_mycroft"],
                                    inference_framework="onnx")
                logger.info(f"[WakeWord] openwakeword sẵn sàng")
            except Exception as exc:
                logger.warning(f"[WakeWord] Không load được openwakeword: {exc} — bypass.")
        else:
            logger.warning(f"[WakeWord] Engine '{self.engine}' chưa hỗ trợ — bypass.")

    def detect(self, audio: np.ndarray) -> bool:
        if self._model is None:
            return True  # bypass
        try:
            audio_int16 = (audio * 32767).astype(np.int16)
            scores = self._model.predict(audio_int16)
            return any(v > 0.5 for v in scores.values())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Daemon chính
# ---------------------------------------------------------------------------

class VoiceAuthDaemon:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = _load_config(config_path)

        log_path = _get(self.cfg, "logging", "path", default="logs/auth_log.jsonl")
        _setup_logging(log_path)

        auth = self.cfg.get("auth", {})
        self.user: str = auth.get("user", "user")
        self.auth_duration: float = auth.get("max_auth_duration_sec", 4.0)
        self.silence_threshold: float = 0.005

        ww_cfg = self.cfg.get("wake_word", {})
        self.use_wake_word: bool = ww_cfg.get("enabled", True)
        self.ww_engine: str = ww_cfg.get("engine", "openwakeword")
        self.ww_keyword: str = ww_cfg.get("keyword", "hey openclaw")
        self.ww_duration: float = ww_cfg.get("listen_duration_sec", 2.0)

        self._verifier = None
        self._stt = None
        self._bridge = None
        self._wwd: Optional[WakeWordDetector] = None

    # ── Component loaders ─────────────────────────────────────────────────

    def _load_verifier(self):
        from auth.verifier import VoiceVerifier
        auth = self.cfg.get("auth", {})
        return VoiceVerifier(
            model_path=_resolve(auth.get("model_path", "models/ecapa_tdnn.pt")),
            voiceprint_path=_resolve(
                auth.get("voiceprint_path", f"auth/voiceprints/{self.user}.npy")
            ),
            cohort_dir=_resolve(auth.get("cohort_dir", "data/cohort")),
            threshold_asnorm=auth.get("threshold_asnorm", 1.5),
            log_path=_resolve(_get(self.cfg, "logging", "path",
                                   default="logs/auth_log.jsonl")),
        )

    def _load_stt(self):
        from stt.parakeet_engine import ParakeetSTT
        return ParakeetSTT()

    def _load_bridge(self):
        from dispatcher.openclaw_bridge import OpenClawBridge
        return OpenClawBridge(mode=self.cfg.get("openclaw_mode", "auto"))

    # ── Start ─────────────────────────────────────────────────────────────

    def start(self, once: bool = False, no_wake_word: bool = False) -> None:
        print("=" * 60)
        print(f"  OpenClaw Voice Auth Daemon  |  skill dir: {SKILL_DIR}")
        print(f"  User: {self.user}  |  auth_duration: {self.auth_duration}s")
        print("=" * 60)

        logger.info("[Daemon] Đang khởi tạo các component ...")
        self._verifier = self._load_verifier()
        self._stt = self._load_stt()
        self._bridge = self._load_bridge()

        if not no_wake_word and self.use_wake_word:
            self._wwd = WakeWordDetector(self.ww_engine, self.ww_keyword)

        logger.info("[Daemon] ✅ Sẵn sàng. Bắt đầu lắng nghe ...")
        try:
            self._loop(once=once, no_wake_word=no_wake_word)
        except KeyboardInterrupt:
            logger.info("[Daemon] Ctrl+C — dừng daemon.")
        finally:
            logger.info("[Daemon] Daemon đã dừng.")

    # ── Main loop ─────────────────────────────────────────────────────────

    def _loop(self, once: bool = False, no_wake_word: bool = False) -> None:
        while True:
            # Giai đoạn 0: Wake word
            if not no_wake_word and self._wwd is not None:
                ww_audio = listen_chunk(self.ww_duration)
                if not self._wwd.detect(ww_audio):
                    continue
                logger.info("[Daemon] 🔔 Wake word detected!")

            # Giai đoạn 1: Ghi audio xác thực
            audio = listen_chunk(self.auth_duration)

            if energy_rms(audio) < self.silence_threshold:
                continue

            # Giai đoạn 2: Voice auth (ECAPA-TDNN + AS-Norm)
            passed, score = self._verifier.verify(audio, sample_rate=SAMPLE_RATE)
            if not passed:
                logger.debug(f"[Daemon] Auth từ chối (score={score:.3f})")
                if once:
                    break
                continue

            logger.info(f"[Daemon] ✅ Auth pass | score={score:.2f}")

            # Giai đoạn 3: STT (Parakeet-TDT-0.6B-v3)
            command = self._stt.transcribe(audio, sample_rate=SAMPLE_RATE)
            if not command:
                logger.info("[Daemon] STT rỗng — bỏ qua.")
                if once:
                    break
                continue

            print(f"\n  💬 [{self.user}] score={score:.2f} | '{command}'")

            # Giai đoạn 4: Gửi sang OpenClaw
            response = self._bridge.dispatch(command, user=self.user)
            if response:
                print(f"  🤖 OpenClaw: {response}\n")

            time.sleep(0.5)

            if once:
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Voice Auth Daemon")
    parser.add_argument("--config", default=str(SKILL_DIR / "config.yaml"),
                        help="Đường dẫn config.yaml")
    parser.add_argument("--once", action="store_true",
                        help="Xử lý 1 lần rồi thoát (debug)")
    parser.add_argument("--no-wake-word", action="store_true", dest="no_wake_word",
                        help="Bỏ qua wake word")
    args = parser.parse_args()

    daemon = VoiceAuthDaemon(config_path=args.config)
    daemon.start(once=args.once, no_wake_word=args.no_wake_word)


if __name__ == "__main__":
    main()
