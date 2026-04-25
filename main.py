"""
main.py — Entry point cho Voice Auth Tray Application.

Chạy: python main.py

Flow:
  1. Load config từ %APPDATA%/VoiceAuthTray/settings.json
  2. Nếu chưa setup → Setup Wizard (nhập Gateway URL + Token)
  3. Nếu chưa enroll → Enrollment GUI (ghi mẫu giọng nói)
  4. Khởi động daemon thread + system tray icon
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path

# ── Thêm thư mục gốc vào sys.path ────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _setup_logging() -> None:
    """Thiết lập logging cho toàn ứng dụng."""
    log_dir = APP_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] %(levelname)-7s %(name)s - %(message)s"

    # Fix Windows cp1252 encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    try:
        handlers.append(
            logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        )
    except OSError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("main")
    logger.info("=" * 50)
    logger.info("  Voice Auth Tray — Starting")
    logger.info(f"  App root: {APP_ROOT}")
    logger.info("=" * 50)

    # ── 1. Load config ───────────────────────────────────────────────────
    from app.config_manager import ConfigManager
    config = ConfigManager()
    logger.info(f"  Config: {config}")

    # ── 2. Check setup ───────────────────────────────────────────────────
    if not config.setup_complete:
        logger.info("[Main] Setup incomplete -> opening Setup Wizard")
        _run_setup(config)

        # Sau setup, kiem tra lai
        if not config.setup_complete:
            logger.info("[Main] User cancelled setup -> exiting.")
            sys.exit(0)

    # ── 3. Check enrollment ──────────────────────────────────────────────
    if not config.enrolled:
        logger.info("[Main] Not enrolled -> opening Enrollment GUI")
        _run_enrollment(config)

        if not config.enrolled:
            logger.info("[Main] User cancelled enrollment -> exiting.")
            sys.exit(0)

    # ── 4. Start daemon + tray ───────────────────────────────────────────
    logger.info("[Main] Starting daemon + tray...")
    _run_tray_mode(config)


def _run_setup(config) -> None:
    """Mở Setup Wizard, chặn cho đến khi hoàn tất hoặc huỷ."""
    from app.setup_wizard import SetupWizard

    completed = [False]

    def on_complete():
        completed[0] = True

    def on_cancel():
        completed[0] = False

    wizard = SetupWizard(
        config=config,
        on_complete=on_complete,
        on_cancel=on_cancel,
    )
    wizard.show()  # Blocking (Tkinter mainloop)


def _run_enrollment(config) -> None:
    """Mở Enrollment GUI, chặn cho đến khi hoàn tất hoặc huỷ."""
    from app.enrollment_gui import EnrollmentGUI

    completed = [False]

    def on_complete():
        completed[0] = True

    def on_cancel():
        completed[0] = False

    gui = EnrollmentGUI(
        config=config,
        on_complete=on_complete,
        on_cancel=on_cancel,
    )
    gui.show()  # Blocking (Tkinter mainloop)


def _run_tray_mode(config) -> None:
    """Khởi động daemon thread + system tray icon."""
    from app.openclaw_client import OpenClawClient
    from app.daemon_thread import DaemonThread
    from app.tray import TrayApp

    # OpenClaw HTTP client
    client = OpenClawClient(
        gateway_url=config.gateway_url,
        token=config.gateway_token,
    )

    # Tray app instance (để daemon callback tới)
    tray = TrayApp(
        config=config,
        daemon_thread=None,   # sẽ set sau
        on_re_enroll=lambda: _re_enroll_from_tray(config, tray),
        on_settings=lambda: _settings_from_tray(config, tray, client),
    )

    # Daemon thread
    daemon = DaemonThread(
        config=config,
        openclaw_client=client,
        on_status=tray.update_status,
    )
    tray.daemon = daemon

    # Start daemon thread (background)
    daemon.start()

    # Run tray (blocking — main thread)
    # Khi user chọn Quit → tray.run() trả về
    tray.run()

    # Cleanup
    if daemon.is_alive():
        daemon.stop()
        daemon.join(timeout=5)

    logging.getLogger("main").info("[Main] Ứng dụng đã thoát.")


def _re_enroll_from_tray(config, tray) -> None:
    """Gọi khi user chọn 'Đăng ký lại' từ tray menu."""
    from app.enrollment_gui import EnrollmentGUI

    def on_complete():
        logging.getLogger("main").info("[Main] Re-enrollment hoàn tất.")
        # TODO: Restart daemon với voiceprint mới

    gui = EnrollmentGUI(
        config=config,
        on_complete=on_complete,
        on_cancel=lambda: None,
    )
    gui.show()


def _settings_from_tray(config, tray, client) -> None:
    """Gọi khi user chọn 'Cài đặt Gateway' từ tray menu."""
    from app.setup_wizard import SetupWizard

    def on_complete():
        # Cập nhật client với URL/token mới
        client.update(config.gateway_url, config.gateway_token)
        logging.getLogger("main").info("[Main] Settings updated.")

    wizard = SetupWizard(
        config=config,
        on_complete=on_complete,
        on_cancel=lambda: None,
    )
    wizard.show()


if __name__ == "__main__":
    main()
