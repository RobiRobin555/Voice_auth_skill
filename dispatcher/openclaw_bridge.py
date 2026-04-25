"""
dispatcher/openclaw_bridge.py — Adapter gửi lệnh văn bản sang OpenClaw.

Hỗ trợ 3 mode (cấu hình trong config.yaml → openclaw_mode):
  - "import"  : import trực tiếp module openclaw (cùng virtualenv).
  - "cli"     : subprocess gọi CLI `openclaw run --prompt "..."`.
  - "http"    : POST JSON tới OpenClaw REST API (localhost:8080/run).

Mode "auto" tự thử "import" → "cli" → "http" cho đến khi thành công.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class OpenClawBridge:
    """
    Adapter gửi lệnh đã transcribe sang OpenClaw.

    Parameters
    ----------
    mode : str
        "auto" | "import" | "cli" | "http"
    api_url : str
        URL cho HTTP mode (mặc định: http://localhost:8080/run).
    timeout : int
        Timeout giây cho CLI và HTTP calls.
    """

    def __init__(
        self,
        mode: str = "auto",
        api_url: str = "http://localhost:8080/run",
        timeout: int = 30,
    ) -> None:
        self.mode = mode
        self.api_url = api_url
        self.timeout = timeout

        if mode == "auto":
            self._resolved_mode = self._detect_best_mode()
        else:
            self._resolved_mode = mode

        logger.info(f"[OpenClawBridge] Mode: '{self._resolved_mode}'")

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------

    def _detect_best_mode(self) -> str:
        """Thử các mode theo thứ tự ưu tiên, trả về mode đầu tiên khả dụng."""
        # Thử import
        try:
            import openclaw  # type: ignore  # noqa: F401
            logger.info("[OpenClawBridge] Phát hiện openclaw package → mode=import")
            return "import"
        except ImportError:
            pass

        # Thử CLI
        try:
            result = subprocess.run(
                ["openclaw", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("[OpenClawBridge] Phát hiện openclaw CLI → mode=cli")
                return "cli"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback HTTP
        logger.info("[OpenClawBridge] Fallback → mode=http")
        return "http"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, command: str, user: str = "unknown") -> Optional[str]:
        """
        Gửi lệnh văn bản sang OpenClaw.

        Parameters
        ----------
        command : str
            Văn bản lệnh đã transcribe từ giọng nói.
        user : str
            Tên user đã xác thực (dùng cho logging).

        Returns
        -------
        str | None — Phản hồi từ OpenClaw nếu có.
        """
        if not command.strip():
            logger.debug("[OpenClawBridge] Lệnh rỗng — bỏ qua.")
            return None

        logger.info(f"[OpenClawBridge] Gửi lệnh [{user}]: '{command}'")

        dispatch_fn = {
            "import": self._dispatch_import,
            "cli": self._dispatch_cli,
            "http": self._dispatch_http,
        }.get(self._resolved_mode)

        if dispatch_fn is None:
            logger.error(f"[OpenClawBridge] Mode không hợp lệ: '{self._resolved_mode}'")
            return None

        try:
            response = dispatch_fn(command)
            if response:
                logger.info(f"[OpenClawBridge] Phản hồi: '{response[:120]}'")
            return response
        except Exception as exc:
            logger.error(f"[OpenClawBridge] Lỗi dispatch ({self._resolved_mode}): {exc}")
            return None

    # ------------------------------------------------------------------
    # Dispatch implementations
    # ------------------------------------------------------------------

    def _dispatch_import(self, command: str) -> Optional[str]:
        """
        Gọi trực tiếp qua import khi OpenClaw cùng virtualenv.

        Điều chỉnh đường dẫn import theo cấu trúc thực tế của OpenClaw.
        """
        try:
            from openclaw import agent  # type: ignore
            return agent.run(command)
        except ImportError:
            # Thử alternative import path
            try:
                import openclaw  # type: ignore
                if hasattr(openclaw, "run"):
                    return openclaw.run(command)
                elif hasattr(openclaw, "process"):
                    return openclaw.process(command)
            except Exception as exc:
                raise RuntimeError(f"Không tìm thấy openclaw.agent.run: {exc}") from exc

    def _dispatch_cli(self, command: str) -> Optional[str]:
        """
        Gọi qua subprocess CLI.

        Command line: openclaw run --prompt "<command>"
        """
        try:
            result = subprocess.run(
                ["openclaw", "run", "--prompt", command],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
            )
            if result.returncode != 0:
                logger.warning(
                    f"[OpenClawBridge] CLI exit {result.returncode}: {result.stderr[:200]}"
                )
            return result.stdout.strip() or None
        except FileNotFoundError:
            raise RuntimeError(
                "Lệnh 'openclaw' không tìm thấy trong PATH. "
                "Kiểm tra cài đặt hoặc thay mode thành 'http'."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"OpenClaw CLI timeout sau {self.timeout}s.")

    def _dispatch_http(self, command: str) -> Optional[str]:
        """
        Gửi POST JSON tới OpenClaw REST API.

        Request:  POST {api_url}  Body: {"prompt": "<command>"}
        Response: {"result": "<response>"}
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({"prompt": command}).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                # Hỗ trợ cả key "result" và "response"
                return data.get("result") or data.get("response") or str(data)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Không kết nối được OpenClaw tại '{self.api_url}': {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Phản hồi không phải JSON hợp lệ: {exc}") from exc
