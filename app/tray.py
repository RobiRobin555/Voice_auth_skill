"""
app/tray.py — System Tray icon + menu (pystray).

Icon thay đổi theo trạng thái:
  🟢 listening  — đang lắng nghe
  🟡 processing — đang xử lý
  🔴 error      — lỗi
  ⚪ loading    — đang khởi tạo
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _create_icon_image(color: str = "green", size: int = 64):
    """Tạo icon đơn giản bằng Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    colors = {
        "green": (0, 230, 118),
        "yellow": (255, 193, 7),
        "red": (255, 82, 82),
        "gray": (120, 144, 156),
    }
    rgb = colors.get(color, colors["gray"])

    # Circle background
    margin = 4
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=rgb,
        outline=(255, 255, 255, 200),
        width=2,
    )

    # Microphone icon text
    try:
        font = ImageFont.truetype("segoeui.ttf", size // 2)
    except Exception:
        font = ImageFont.load_default()

    draw.text(
        (size // 2, size // 2),
        "🎤",
        fill=(255, 255, 255),
        font=font,
        anchor="mm",
    )

    return img


class TrayApp:
    """
    System Tray application.

    Parameters
    ----------
    config : ConfigManager
    daemon_thread : DaemonThread
    on_re_enroll : callable
        Gọi khi user chọn "Đăng ký lại giọng nói".
    on_settings : callable
        Gọi khi user chọn "Cài đặt Gateway".
    """

    def __init__(
        self,
        config,
        daemon_thread,
        on_re_enroll: Optional[Callable] = None,
        on_settings: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.daemon = daemon_thread
        self.on_re_enroll = on_re_enroll
        self.on_settings = on_settings

        self._icon = None
        self._status = "loading"
        self._status_text_map = {
            "loading": "⏳ Đang khởi tạo...",
            "listening": "🟢 Đang lắng nghe",
            "processing": "🟡 Đang xử lý...",
            "error": "🔴 Lỗi",
        }
        self._color_map = {
            "loading": "gray",
            "listening": "green",
            "processing": "yellow",
            "error": "red",
        }

    # ── Status callback (from daemon thread) ─────────────────────────────

    def update_status(self, status: str) -> None:
        """Cập nhật trạng thái — được gọi từ daemon thread."""
        self._status = status
        if self._icon:
            try:
                color = self._color_map.get(status, "gray")
                self._icon.icon = _create_icon_image(color)
                title = self._status_text_map.get(status, status)
                self._icon.title = f"Voice Auth — {title}"
            except Exception:
                pass

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Chạy system tray icon (blocking)."""
        import pystray

        image = _create_icon_image("gray")

        menu = pystray.Menu(
            pystray.MenuItem(
                lambda item: self._status_text_map.get(self._status, "Unknown"),
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("🎙 Đăng ký lại giọng nói", self._on_re_enroll),
            pystray.MenuItem("⚙️ Cài đặt Gateway", self._on_settings),
            pystray.MenuItem("📂 Mở thư mục Logs", self._open_logs),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("❌ Thoát", self._on_quit),
        )

        self._icon = pystray.Icon(
            "VoiceAuth",
            image,
            "Voice Auth Tray — Đang khởi tạo...",
            menu,
        )

        logger.info("[Tray] System tray icon đã sẵn sàng.")
        self._icon.run()

    # ── Menu actions ─────────────────────────────────────────────────────

    def _on_re_enroll(self, icon, item) -> None:
        logger.info("[Tray] User yêu cầu đăng ký lại giọng nói.")
        if self.daemon and self.daemon.is_alive():
            self.daemon.stop()
        if self.on_re_enroll:
            # Chạy enrollment GUI trong thread riêng (vì tray blocking)
            threading.Thread(target=self.on_re_enroll, daemon=True).start()

    def _on_settings(self, icon, item) -> None:
        logger.info("[Tray] User mở cài đặt Gateway.")
        if self.on_settings:
            threading.Thread(target=self.on_settings, daemon=True).start()

    def _open_logs(self, icon, item) -> None:
        log_dir = Path(self.config.log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Tray] Mở logs: {log_dir}")
        if sys.platform == "win32":
            import os
            os.startfile(str(log_dir))
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(log_dir)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(log_dir)])

    def _on_quit(self, icon, item) -> None:
        logger.info("[Tray] Đang thoát...")
        if self.daemon and self.daemon.is_alive():
            self.daemon.stop()
        icon.stop()

    # ── Stop ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Dừng tray icon."""
        if self._icon:
            self._icon.stop()
