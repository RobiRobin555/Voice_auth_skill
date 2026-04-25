"""
auth/enroll.py — Đăng ký voiceprint lần đầu.

Sử dụng:
    # Ghi từ microphone (interactive, 5 lần mỗi lần 3 giây)
    python auth/enroll.py --user alice --samples 5

    # Dùng file .wav có sẵn
    python auth/enroll.py --user alice --files enroll_1.wav enroll_2.wav enroll_3.wav

    # Ví dụ giả lập (không cần mic / file thật) — để test nhanh
    python auth/enroll.py --user alice --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# FIX: Đảm bảo thư mục gốc của skill nằm trong sys.path để tránh lỗi ModuleNotFoundError 
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import yaml  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _record_from_mic(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Ghi âm từ microphone trong `duration` giây."""
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        sys.exit("❌  Cài sounddevice trước:  pip install sounddevice")

    logger.info(f"  🎙  Đang ghi âm {duration:.0f}s ... (nói đều, đừng dừng lâu)")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def _save_temp_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Lưu numpy audio → file WAV tạm, trả về path."""
    import tempfile

    import soundfile as sf  # type: ignore

    fd, path = tempfile.mkstemp(suffix=".wav")
    sf.write(path, audio, sample_rate)
    return path


def run_enroll_mic(
    user: str,
    model_path: str,
    voiceprint_path: str,
    cohort_dir: str,
    samples: int = 5,
    duration: float = 3.0,
    threshold: float = 1.5,
    log_path: str = "logs/auth_log.jsonl",
) -> None:
    """Đăng ký voiceprint bằng ghi âm từ microphone."""
    from auth.verifier import VoiceVerifier  # noqa: PLC0415

    print(f"\n{'='*55}")
    print(f"  OpenClaw Voice Auth — Enroll user: '{user}'")
    print(f"  {samples} lần ghi âm, mỗi lần {duration:.0f} giây")
    print(f"{'='*55}\n")

    verifier = VoiceVerifier(
        model_path=model_path,
        voiceprint_path="__skip__",   # bỏ qua load voiceprint khi enroll
        cohort_dir=cohort_dir,
        threshold_asnorm=threshold,
        log_path=log_path,
    )

    temp_files: list[str] = []
    try:
        for i in range(samples):
            input(f"  [Mẫu {i+1}/{samples}] Nhấn Enter rồi nói ...")
            audio = _record_from_mic(duration=duration)
            path = _save_temp_wav(audio)
            temp_files.append(path)
            logger.info(f"  ✅  Mẫu {i+1} ghi xong ({len(audio)/16000:.1f}s)")
            time.sleep(0.3)

        print("\n  Đang tính voiceprint ...")
        verifier.enroll(audio_files=temp_files, save_path=voiceprint_path)
        print(f"\n  ✅  Voiceprint cho user '{user}' đã được lưu vào '{voiceprint_path}'")
        print("  Bây giờ có thể chạy:  python voice_auth_daemon.py\n")

    finally:
        # Dọn file tạm
        import os
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


def run_enroll_files(
    user: str,
    model_path: str,
    voiceprint_path: str,
    cohort_dir: str,
    audio_files: list[str],
    threshold: float = 1.5,
    log_path: str = "logs/auth_log.jsonl",
) -> None:
    """Đăng ký voiceprint từ danh sách file .wav có sẵn."""
    from auth.verifier import VoiceVerifier  # noqa: PLC0415

    verifier = VoiceVerifier(
        model_path=model_path,
        voiceprint_path="__skip__",
        cohort_dir=cohort_dir,
        threshold_asnorm=threshold,
        log_path=log_path,
    )
    verifier.enroll(audio_files=audio_files, save_path=voiceprint_path)
    print(f"\n✅  Voiceprint cho '{user}' đã lưu → '{voiceprint_path}'")


def run_demo_enroll(
    user: str,
    voiceprint_path: str,
    emb_dim: int = 192,
) -> None:
    """
    Tạo voiceprint giả (random, L2-normalized) — chỉ dùng để test hệ thống
    mà không cần model thật hoặc microphone.
    """
    print(f"\n[DEMO] Tạo voiceprint giả cho user '{user}' ...")
    dummy = np.random.randn(emb_dim).astype(np.float32)
    dummy = dummy / np.linalg.norm(dummy)

    save_path = Path(voiceprint_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), dummy)
    print(f"[DEMO] ✅  Lưu dummy voiceprint → '{save_path}'")
    print("[DEMO] ⚠️   Đây chỉ là dữ liệu test, không có giá trị xác thực thật.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đăng ký voiceprint cho OpenClaw Voice Auth"
    )
    parser.add_argument("--user", default=None, help="Tên user (ghi đè config.yaml)")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn config.yaml")
    parser.add_argument("--samples", type=int, default=5, help="Số lần ghi âm (mic mode)")
    parser.add_argument("--duration", type=float, default=3.0, help="Giây mỗi mẫu ghi âm")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Dùng file .wav thay vì ghi mic, ví dụ: --files a.wav b.wav c.wav",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Tạo voiceprint giả (không cần mic/model) — chỉ để test",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    auth_cfg = cfg.get("auth", {})

    user = args.user or auth_cfg.get("user", "user")
    model_path = auth_cfg.get("model_path", "models/your_model.pt")
    voiceprint_path = auth_cfg.get("voiceprint_path", f"auth/voiceprints/{user}.npy")
    cohort_dir = auth_cfg.get("cohort_dir", "data/cohort")
    threshold = auth_cfg.get("threshold_asnorm", 1.5)
    log_path = cfg.get("logging", {}).get("path", "logs/auth_log.jsonl")

    if args.demo:
        run_demo_enroll(user=user, voiceprint_path=voiceprint_path)
        return

    if args.files:
        run_enroll_files(
            user=user,
            model_path=model_path,
            voiceprint_path=voiceprint_path,
            cohort_dir=cohort_dir,
            audio_files=args.files,
            threshold=threshold,
            log_path=log_path,
        )
    else:
        run_enroll_mic(
            user=user,
            model_path=model_path,
            voiceprint_path=voiceprint_path,
            cohort_dir=cohort_dir,
            samples=args.samples,
            duration=args.duration,
            threshold=threshold,
            log_path=log_path,
        )


if __name__ == "__main__":
    main()
