"""
auth/verifier.py — Speaker verification sử dụng ECAPA-TDNN + AS-Norm.

Pipeline:
  1. Tiền xử lý audio (resample 16 kHz, normalize, loại silence).
  2. Trích xuất embedding 192-dim qua ECAPA-TDNN.
  3. Tính cosine similarity với voiceprint người dùng.
  4. Chuẩn hóa điểm bằng AS-Norm (Z-score so với cohort).
  5. So sánh Z-score với ngưỡng → True / False.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

# ── Resolve thư mục skill (auth/ nằm trong skill/) ─────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample numpy audio array về target_sr nếu cần."""
    if orig_sr == target_sr:
        return audio
    waveform = torch.from_numpy(audio).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return waveform.squeeze(0).numpy()


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio về [-1, 1]."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio


def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Cắt đầu/cuối có biên độ thấp hơn threshold."""
    mask = np.abs(audio) > threshold
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return audio
    return audio[indices[0]: indices[-1] + 1]


def _load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load file .wav → numpy float32 16 kHz mono."""
    waveform, sr = torchaudio.load(path)
    # Chuyển sang mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    audio = _resample(audio, sr, target_sr)
    audio = _normalize(audio)
    audio = _trim_silence(audio)
    return audio


def _short_hash(audio: np.ndarray, length: int = 8) -> str:
    """MD5 hash ngắn của audio numpy array, dùng cho log."""
    raw = audio.tobytes()
    return hashlib.md5(raw).hexdigest()[:length]


# ---------------------------------------------------------------------------
# VoiceVerifier
# ---------------------------------------------------------------------------

class VoiceVerifier:
    """
    Speaker verifier kết hợp ECAPA-TDNN và AS-Norm.

    Parameters
    ----------
    model_path : str
        Đường dẫn file .pt / .pth ECAPA-TDNN đã train.
    voiceprint_path : str
        File .npy lưu embedding trung bình của người dùng.
    cohort_dir : str
        Thư mục chứa các file .wav người lạ để tính AS-Norm.
    threshold_asnorm : float
        Z-score tối thiểu để chấp nhận (ngưỡng an ninh).
    log_path : str
        Đường dẫn file log JSONL ghi các lượt từ chối.
    device : str | None
        "cuda" / "cpu" / None (tự phát hiện).
    """

    def __init__(
        self,
        model_path: str,
        voiceprint_path: str,
        cohort_dir: str,
        threshold_asnorm: float = 1.5,
        log_path: str = "logs/auth_log.jsonl",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold_asnorm
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Load model ──────────────────────────────────────────────────────
        logger.info(f"[VoiceVerifier] Loading ECAPA-TDNN từ '{model_path}' ...")
        self.model = self._load_model(model_path)
        self.model.eval()

        # ── Load voiceprint ─────────────────────────────────────────────────
        voiceprint_path = Path(voiceprint_path)
        if not voiceprint_path.exists():
            raise FileNotFoundError(
                f"Voiceprint không tồn tại: {voiceprint_path}. "
                "Hãy chạy enroll.py trước."
            )
        self.voiceprint: np.ndarray = np.load(str(voiceprint_path))
        logger.info(f"[VoiceVerifier] Đã load voiceprint shape={self.voiceprint.shape}")

        # ── Load cohort embeddings (AS-Norm) ────────────────────────────────
        logger.info(f"[VoiceVerifier] Chuẩn bị cohort từ '{cohort_dir}' ...")
        self.cohort_embs: np.ndarray = self._build_cohort(cohort_dir)
        logger.info(
            f"[VoiceVerifier] Cohort: {len(self.cohort_embs)} mẫu — sẵn sàng."
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load ECAPA-TDNN từ checkpoint .pt.

        Hỗ trợ 2 định dạng:
          - state_dict thuần (dict): tự khởi tạo kiến trúc rồi load_state_dict.
          - Module đã serialize (torch.save(model, ...)): load trực tiếp.
        """
        try:
            # Thử import ECAPA_TDNN từ thư mục skill (src/models/ecapa_tdnn.py)
            ecapa_path = SKILL_DIR / "src" / "models" / "ecapa_tdnn.py"
            try:
                if ecapa_path.exists():
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "ecapa_tdnn", str(ecapa_path)
                    )
                    module = importlib.util.module_from_spec(spec)  # type: ignore
                    spec.loader.exec_module(module)  # type: ignore
                    ECAPA_TDNN = module.ECAPA_TDNN
                else:
                    from src.models.ecapa_tdnn import ECAPA_TDNN  # type: ignore

                model = ECAPA_TDNN(C=512)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                # Hỗ trợ cả checkpoint dict lẫn state_dict thuần
                if isinstance(checkpoint, dict):
                    state_dict = (
                        checkpoint.get("model_state_dict")
                        or checkpoint.get("state_dict")
                        or checkpoint
                    )
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)

            except (ModuleNotFoundError, AttributeError):
                # Fallback: model đã serialize hoàn chỉnh
                logger.warning(
                    "[VoiceVerifier] Không tìm thấy src/models/ecapa_tdnn.py "
                    "— thử load module serialized."
                )
                model = torch.load(model_path, map_location=self.device, weights_only=False)  # type: ignore

            model.to(self.device)
            return model

        except Exception as exc:
            raise RuntimeError(
                f"Không thể load model từ '{model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding 192-dim từ audio numpy float32 16 kHz.

        Returns
        -------
        np.ndarray shape (192,)
        """
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)  # (1, T)

        # ECAPA-TDNN thường nhận spectrogram hoặc raw waveform tùy implement.
        # Nếu model yêu cầu spectrogram, chuyển ở đây.
        try:
            emb = self.model(waveform)  # Thử raw waveform trước
        except Exception:
            # Fallback: tính log mel-spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                hop_length=160,
                n_mels=80,
            ).to(self.device)
            spec = mel_transform(waveform)  # (1, 80, T)
            spec = (spec + 1e-6).log()
            emb = self.model(spec)

        # Đưa về numpy 1-D, normalize L2
        emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm
        return emb_np

    # ------------------------------------------------------------------
    # Cohort building
    # ------------------------------------------------------------------

    def _build_cohort(self, cohort_dir: str) -> np.ndarray:
        """
        Tính embedding cho toàn bộ file .wav trong cohort_dir.
        Cache trong RAM — chỉ gọi 1 lần lúc khởi động.

        Returns
        -------
        np.ndarray shape (N, emb_dim)
        """
        cohort_path = Path(cohort_dir)
        wav_files = sorted(cohort_path.glob("*.wav"))

        if not wav_files:
            logger.warning(
                f"[VoiceVerifier] Không tìm thấy file .wav trong '{cohort_dir}'. "
                "AS-Norm sẽ dùng cosine thuần (kém ổn định hơn)."
            )
            return np.empty((0,), dtype=np.float32)

        embs = []
        for wav in wav_files:
            try:
                audio = _load_wav(str(wav))
                emb = self._extract_embedding(audio)
                embs.append(emb)
            except Exception as exc:
                logger.warning(f"  Bỏ qua '{wav.name}': {exc}")

        return np.array(embs, dtype=np.float32)  # (N, emb_dim)

    # ------------------------------------------------------------------
    # AS-Norm
    # ------------------------------------------------------------------

    def _cosine(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity giữa hai vector đã L2-normalize → [-1, 1]."""
        return float(np.dot(emb_a, emb_b))

    def _asnorm_score(self, test_emb: np.ndarray) -> float:
        """
        Tính AS-Norm Z-score:

          Z = (raw_score - μ_cohort) / σ_cohort

        Nếu không có cohort, trả về raw cosine score (không chuẩn hóa).
        """
        raw_score = self._cosine(test_emb, self.voiceprint)

        if len(self.cohort_embs) == 0:
            return raw_score  # Fallback: cosine thuần

        # Tính tất cả similarity giữa test_emb và từng cohort mẫu
        cohort_scores = self.cohort_embs @ test_emb  # (N,)
        mu = cohort_scores.mean()
        sigma = cohort_scores.std() + 1e-8
        z = (raw_score - mu) / sigma
        return float(z)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        Xác thực giọng nói.

        Parameters
        ----------
        audio : np.ndarray
            Audio float32 (mono).
        sample_rate : int
            Sample rate của audio đầu vào.

        Returns
        -------
        (passed, score)
            passed : bool — True nếu xác thực thành công.
            score  : float — AS-Norm Z-score.
        """
        # Tiền xử lý
        audio = audio.astype(np.float32)
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)
        audio = _normalize(audio)
        audio = _trim_silence(audio)

        # Kiểm tra audio quá ngắn
        if len(audio) < 1600:  # < 0.1 giây
            logger.debug("[VoiceVerifier] Audio quá ngắn — bỏ qua.")
            return False, -999.0

        # Trích xuất embedding
        test_emb = self._extract_embedding(audio)

        # Tính AS-Norm score
        score = self._asnorm_score(test_emb)
        passed = score >= self.threshold

        if not passed:
            self._log_rejection(score, audio)

        return passed, score

    def enroll(
        self,
        audio_files: list[str],
        save_path: str,
    ) -> np.ndarray:
        """
        Đăng ký voiceprint từ danh sách file .wav.

        1. Tiền xử lý từng file.
        2. Tính embedding 192-dim.
        3. Tính trung bình → normalize L2 → lưu .npy.

        Parameters
        ----------
        audio_files : list[str]
            Danh sách đường dẫn file .wav (~3s mỗi file).
        save_path : str
            Nơi lưu file .npy kết quả.

        Returns
        -------
        np.ndarray shape (emb_dim,) — voiceprint đã normalize.
        """
        if not audio_files:
            raise ValueError("Danh sách audio_files rỗng.")

        embs = []
        for path in audio_files:
            try:
                audio = _load_wav(path)
                emb = self._extract_embedding(audio)
                embs.append(emb)
                logger.info(f"  [enroll] ✓ {Path(path).name}")
            except Exception as exc:
                logger.warning(f"  [enroll] Bỏ qua '{path}': {exc}")

        if not embs:
            raise RuntimeError("Không có file nào được enroll thành công.")

        voiceprint = np.mean(embs, axis=0).astype(np.float32)
        norm = np.linalg.norm(voiceprint)
        if norm > 0:
            voiceprint = voiceprint / norm

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), voiceprint)
        logger.info(f"[enroll] Voiceprint đã lưu → '{save_path}'")

        return voiceprint

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_rejection(self, score: float, audio: np.ndarray) -> None:
        """Ghi log từ chối — im lặng, không thông báo."""
        entry = {
            "timestamp": time.time(),
            "event": "auth_rejected",
            "score_asnorm": round(score, 4),
            "threshold": self.threshold,
            "audio_hash": _short_hash(audio),
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[VoiceVerifier] Không ghi được log: {exc}")
