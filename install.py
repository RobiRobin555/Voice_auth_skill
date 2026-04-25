"""
install.py — Script thiết lập nhanh sau khi clone về.

Chạy: python install.py

Tự động:
  1. Kiểm tra Python version
  2. Cài pip dependencies
  3. Tạo các thư mục cần thiết
  4. Kiểm tra file model và cohort
  5. In hướng dẫn bước tiếp theo
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✅{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠️ {RESET} {msg}")


def err(msg: str) -> None:
    print(f"  {RED}❌{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {CYAN}ℹ️ {RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")


# ── 1. Kiểm tra Python version ─────────────────────────────────────────────

def check_python() -> bool:
    header("1. Kiểm tra Python version")
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or minor < 10:
        err(f"Cần Python 3.10+, hiện tại: {major}.{minor}")
        return False
    ok(f"Python {major}.{minor} — OK")
    return True


# ── 2. Cài pip dependencies ────────────────────────────────────────────────

def install_deps() -> bool:
    header("2. Cài pip dependencies")
    req = SKILL_DIR / "requirements.txt"
    if not req.exists():
        err(f"Không tìm thấy {req}")
        return False

    print(f"  Đang chạy: pip install -r {req} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
        capture_output=False,
    )
    if result.returncode != 0:
        err("pip install thất bại. Thử chạy thủ công:")
        info(f"  pip install -r {req}")
        return False

    ok("pip dependencies đã cài xong")
    warn("NeMo (Parakeet) cần cài riêng nếu chưa có:")
    info("  pip install nemo_toolkit[asr]")
    return True


# ── 3. Tạo thư mục cần thiết ──────────────────────────────────────────────

def create_dirs() -> None:
    header("3. Tạo thư mục cần thiết")
    dirs = [
        SKILL_DIR / "models",
        SKILL_DIR / "auth" / "voiceprints",
        SKILL_DIR / "data" / "cohort",
        SKILL_DIR / "logs",
        SKILL_DIR / "src" / "models",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        ok(f"OK: {d.relative_to(SKILL_DIR)}/")


# ── 4. Kiểm tra model ─────────────────────────────────────────────────────

def check_model() -> bool:
    header("4. Kiểm tra model ECAPA-TDNN")
    model_dir = SKILL_DIR / "models"
    pt_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))

    if not pt_files:
        warn("Chưa có file model trong models/")
        info("Đặt file .pt hoặc .pth vào models/ rồi cập nhật config.yaml:")
        info("  auth:")
        info("    model_path: models/your_model.pt")
        return False

    for f in pt_files:
        size_mb = f.stat().st_size / 1_048_576
        ok(f"Tìm thấy: {f.name} ({size_mb:.1f} MB)")

    return True


# ── 5. Kiểm tra cohort ────────────────────────────────────────────────────

def check_cohort() -> None:
    header("5. Kiểm tra cohort (AS-Norm)")
    cohort_dir = SKILL_DIR / "data" / "cohort"
    wav_files = list(cohort_dir.glob("*.wav")) + list(cohort_dir.glob("*.flac"))

    if not wav_files:
        warn("Chưa có file .wav trong data/cohort/")
        info("Xem data/cohort/README.txt để tải dataset miễn phí.")
        info("Daemon vẫn chạy nhưng dùng cosine thuần thay AS-Norm.")
        info("→ Khuyến nghị đặt threshold_asnorm: 0.75 trong config.yaml")
    else:
        ok(f"{len(wav_files)} file cohort — AS-Norm sẵn sàng")


# ── 6. In hướng dẫn tiếp theo ─────────────────────────────────────────────

def print_next_steps() -> None:
    header("══ Hoàn tất! Bước tiếp theo ══")
    print()
    print(f"  {BOLD}[1] Sửa config nếu cần:{RESET}")
    print(f"      notepad config.yaml      (Windows)")
    print(f"      nano config.yaml         (Linux/Mac)")
    print()
    print(f"  {BOLD}[2] Đăng ký giọng nói:{RESET}")
    print(f"      python auth/enroll.py --user alice --samples 5")
    print()
    print(f"  {BOLD}[3] Chạy daemon:{RESET}")
    print(f"      python voice_auth_daemon.py")
    print()
    print(f"  {BOLD}[Test nhanh (không cần mic thật):{RESET}")
    print(f"      python auth/enroll.py --demo --user alice")
    print(f"      python voice_auth_daemon.py --once --no-wake-word")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(f"{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}  OpenClaw Voice Auth Skill — Setup{RESET}")
    print(f"{BOLD}{'='*55}{RESET}")

    if not check_python():
        sys.exit(1)

    install_deps()
    create_dirs()
    check_model()
    check_cohort()
    print_next_steps()


if __name__ == "__main__":
    main()
