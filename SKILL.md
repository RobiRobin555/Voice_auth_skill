---
name: openclaw-voice-auth
description: >
  Skill xác thực người dùng qua giọng nói và chuyển lệnh sang OpenClaw thông qua Speech-to-Text.
  Kích hoạt skill này khi người dùng đề cập đến: xác thực giọng nói, voice authentication,
  wake word, lắng nghe nền, Parakeet STT, Resemblyzer, hoặc muốn OpenClaw nhận lệnh bằng giọng nói.
  Cũng kích hoạt khi cần tích hợp audio pipeline, microphone daemon, hoặc bảo mật bằng giọng nói
  cho bất kỳ AI agent nào chạy local.
---

# OpenClaw Voice Auth Skill

Skill này triển khai một **daemon chạy nền** thực hiện pipeline 3 giai đoạn:

```
Microphone → [Wake Word] → Voice Auth (Resemblyzer) → STT (Parakeet) → OpenClaw
```

Toàn bộ pipeline chạy **local-first**, không gửi dữ liệu ra ngoài.

---

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────┐
│                  voice_auth_daemon.py                │
│                                                      │
│  [MicListener] ──► [WakeWordDetector]               │
│                           │                          │
│                    ┌──────▼──────┐                  │
│                    │ VoiceAuth   │  Resemblyzer       │
│                    │ (speaker    │  so sánh embedding │
│                    │ verificat.) │  với voiceprint    │
│                    └──────┬──────┘                  │
│                   ✓ Pass  │  ✗ Fail                  │
│                           │         └─► log + silent │
│                    ┌──────▼──────┐                  │
│                    │  STT        │  Parakeet-TDT     │
│                    │  (Parakeet) │  0.6B-v3          │
│                    └──────┬──────┘                  │
│                    ┌──────▼──────┐                  │
│                    │  OpenClaw   │  Python API call  │
│                    │  Dispatcher │                   │
│                    └─────────────┘                  │
└─────────────────────────────────────────────────────┘
```

---

## Cài đặt & Dependencies

### Yêu cầu hệ thống
- Python 3.10+
- Microphone (PyAudio hoặc sounddevice)
- RAM tối thiểu 4GB (Parakeet cần ~1.5GB, Resemblyzer ~200MB)
- GPU tùy chọn (CUDA) — pipeline hoạt động tốt trên CPU

### Cài đặt thư viện

```bash
pip install torch torchaudio sounddevice numpy scipy soundfile pyyaml
pip install nemo_toolkit[asr]   # Parakeet-TDT via NVIDIA NeMo
pip install openwakeword         # Wake word hoàn toàn offline, miễn phí
# Hoặc: pip install pvporcupine  # Nếu muốn dùng Picovoice (có free tier)
```

> Model ECAPA-TDNN (`src/models/ecapa_tdnn.py`) và file `.pt` dùng nguyên từ project của bạn — không cần cài thêm gì.

### Cấu trúc thư mục

```
openclaw-voice-auth/
├── SKILL.md
├── voice_auth_daemon.py      ← Entry point, daemon chính
├── auth/
│   ├── enroll.py             ← Đăng ký voiceprint lần đầu
│   ├── verifier.py           ← Logic xác thực Resemblyzer
│   └── voiceprints/          ← Lưu embedding người dùng (.npy)
├── stt/
│   └── parakeet_engine.py    ← Wrapper Parakeet-TDT-0.6B-v3
├── dispatcher/
│   └── openclaw_bridge.py    ← Gửi lệnh đã transcribe sang OpenClaw
├── logs/
│   └── auth_log.jsonl        ← Log từ chối (im lặng, không alert)
└── config.yaml               ← Cấu hình ngưỡng, wake word, paths
```

---

## Giai đoạn 1 — Đăng ký Voiceprint (Enrollment)

Model xác thực dùng kiến trúc **ECAPA-TDNN** (channels=512, emb_size=192) kết hợp **AS-Norm** để chuẩn hóa điểm số — đây là phương pháp speaker verification chuyên nghiệp, chính xác hơn cosine đơn thuần.

**Score AS-Norm là Z-score**, không phải probability 0–1. Ngưỡng mặc định `1.5` có nghĩa: giọng nói phải vượt 1.5 độ lệch chuẩn so với tập người lạ (cohort) mới được chấp nhận.

Trước khi chạy daemon, cần chuẩn bị 2 thứ:

### 1a. Tạo voiceprint người dùng

```python
from auth.verifier import VoiceVerifier

verifier = VoiceVerifier(
    model_path="models/your_model.pt",
    voiceprint_path="auth/voiceprints/alice.npy",  # sẽ được tạo
    cohort_dir="data/cohort",
)

# Enroll từ 5 file .wav giọng của bạn (~3 giây mỗi file)
verifier.enroll(
    audio_files=["enroll_1.wav", "enroll_2.wav", "enroll_3.wav",
                 "enroll_4.wav", "enroll_5.wav"],
    save_path="auth/voiceprints/alice.npy"
)
```

`enroll()` tự động:
1. Tiền xử lý từng file (resample 16kHz, normalize, cắt silence)
2. Tính embedding 192-dim cho từng file
3. Tính trung bình → chuẩn hóa L2 → lưu `.npy`

### 1b. Chuẩn bị cohort (AS-Norm)

Cohort là tập file `.wav` của **người lạ** (không phải người dùng), dùng để chuẩn hóa điểm số:

```
data/cohort/
├── stranger_001.wav
├── stranger_002.wav
└── ...  (50–200 file, càng nhiều AS-Norm càng ổn định)
```

Cohort được **load 1 lần lúc khởi động daemon**, cache trong RAM — không reload mỗi lần verify.

**Ngưỡng AS-Norm Z-score:**
- `1.5` — mặc định, cân bằng bảo mật/tiện dụng
- `2.0` — bảo mật cao hơn, dễ từ chối nhầm khi ồn
- `1.0` — dễ chấp nhận hơn, giảm false rejection

```yaml
auth:
  user: alice
  model_path: models/your_model.pt
  voiceprint_path: auth/voiceprints/alice.npy
  cohort_dir: data/cohort
  threshold_asnorm: 1.5
  max_auth_duration_sec: 4
```

### 1c. Cách gọi trong daemon

```python
from auth.verifier import VoiceVerifier

# Khởi tạo 1 lần khi daemon start
verifier = VoiceVerifier(
    model_path=CONFIG["auth"]["model_path"],
    voiceprint_path=CONFIG["auth"]["voiceprint_path"],
    cohort_dir=CONFIG["auth"]["cohort_dir"],
    threshold_asnorm=CONFIG["auth"]["threshold_asnorm"],
    log_path=CONFIG["logging"]["path"],
)

# Gọi mỗi khi có audio từ mic
passed, score = verifier.verify(audio_np, sample_rate=16000)
# passed = True/False
# score  = AS-Norm Z-score (ghi vào log khi từ chối)
```

---

## Giai đoạn 2 — STT với Parakeet-TDT-0.6B-v3

Parakeet-TDT là model ASR của NVIDIA, chạy local qua NeMo framework.

**parakeet_engine.py:**

```python
import nemo.collections.asr as nemo_asr
import soundfile as sf
import tempfile, numpy as np

class ParakeetSTT:
    def __init__(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v3"
        )
        self.model.eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        # Parakeet yêu cầu file path hoặc list of paths
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            result = self.model.transcribe([f.name])
        return result[0].strip()
```

**Lưu ý quan trọng về Parakeet:**
- Model tải lần đầu ~1-2 phút, sau đó cached
- Sample rate yêu cầu: **16kHz mono**
- Nếu không có GPU, thêm `map_location="cpu"` vào config NeMo
- Độ trễ transcribe trên CPU: ~1-3 giây cho câu lệnh ngắn

---

## Giai đoạn 3 — Gửi lệnh sang OpenClaw

**openclaw_bridge.py** — thiết kế dạng adapter để dễ thay đổi interface sau:

```python
import subprocess, json
from typing import Optional

class OpenClawBridge:
    """
    Adapter gửi lệnh văn bản sang OpenClaw.
    Hiện tại dùng Python import trực tiếp (nếu OpenClaw cùng môi trường)
    hoặc subprocess CLI. Dễ đổi sang REST API sau.
    """

    def __init__(self, mode: str = "auto"):
        self.mode = mode  # "import" | "cli" | "http"

    def dispatch(self, command: str, user: str) -> Optional[str]:
        if self.mode == "import":
            return self._dispatch_import(command)
        elif self.mode == "cli":
            return self._dispatch_cli(command)
        elif self.mode == "http":
            return self._dispatch_http(command)

    def _dispatch_import(self, command: str):
        # Dùng khi OpenClaw cài trong cùng virtualenv
        from openclaw import agent  # điều chỉnh theo cấu trúc thực tế
        return agent.run(command)

    def _dispatch_cli(self, command: str):
        result = subprocess.run(
            ["openclaw", "run", "--prompt", command],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout

    def _dispatch_http(self, command: str):
        import urllib.request
        payload = json.dumps({"prompt": command}).encode()
        req = urllib.request.Request(
            "http://localhost:8080/run",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["result"]
```

---

## Xử lý thất bại xác thực — Silent Reject + Logging

Khi xác thực thất bại, **không phát âm thanh, không thông báo** — ghi log ngầm:

```python
import json, time
from pathlib import Path

LOG_PATH = Path("logs/auth_log.jsonl")

def log_rejection(user: str, similarity: float, audio_hash: str):
    entry = {
        "timestamp": time.time(),
        "event": "auth_rejected",
        "user": user,
        "similarity_score": round(similarity, 4),
        "audio_hash": audio_hash   # hash ngắn của audio, không lưu raw
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    # Không raise exception, không phát âm thanh — im lặng tuyệt đối
```

---

## Daemon chính — voice_auth_daemon.py

```python
import sounddevice as sd
import numpy as np
import time
import yaml
from auth.verifier import VoiceVerifier
from stt.parakeet_engine import ParakeetSTT
from dispatcher.openclaw_bridge import OpenClawBridge

CONFIG = yaml.safe_load(open("config.yaml"))
SAMPLE_RATE = 16000

# Khởi tạo tất cả 1 lần — model không reload giữa các vòng lặp
verifier = VoiceVerifier(
    model_path=CONFIG["auth"]["model_path"],
    voiceprint_path=CONFIG["auth"]["voiceprint_path"],
    cohort_dir=CONFIG["auth"]["cohort_dir"],
    threshold_asnorm=CONFIG["auth"]["threshold_asnorm"],
    log_path=CONFIG["logging"]["path"],
)
stt = ParakeetSTT()
bridge = OpenClawBridge(mode=CONFIG.get("openclaw_mode", "cli"))

def listen_chunk(duration: float) -> np.ndarray:
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1, dtype="float32"
    )
    sd.wait()
    return audio.flatten()

def run_daemon():
    print(f"[VoiceAuth] Daemon khởi động — user: '{CONFIG['auth']['user']}'")
    while True:
        audio = listen_chunk(duration=CONFIG["auth"]["max_auth_duration_sec"])

        # Bước 1: Xác thực (ECAPA-TDNN + AS-Norm)
        passed, score = verifier.verify(audio, sample_rate=SAMPLE_RATE)

        if not passed:
            # Im lặng — log đã được ghi trong verifier.verify()
            continue

        # Bước 2: STT (Parakeet-TDT-0.6B-v3)
        command = stt.transcribe(audio)
        if not command:
            continue

        print(f"[VoiceAuth] ✅ score={score:.2f} | Lệnh: '{command}'")

        # Bước 3: Gửi sang OpenClaw
        bridge.dispatch(command, user=CONFIG["auth"]["user"])
        time.sleep(0.5)

if __name__ == "__main__":
    run_daemon()
```

---

## config.yaml mẫu

```yaml
auth:
  user: alice
  model_path: models/your_model.pt       # file .pt / .pth của bạn
  voiceprint_path: auth/voiceprints/alice.npy
  cohort_dir: data/cohort                # thư mục wav người lạ cho AS-Norm
  threshold_asnorm: 1.5                  # Z-score, không phải 0-1
  max_auth_duration_sec: 4

openclaw_mode: cli   # "import" | "cli" | "http"

wake_word:
  engine: openwakeword   # hoặc "pvporcupine"
  keyword: "hey openclaw"

logging:
  path: logs/auth_log.jsonl
  silent_reject: true
```

---

## Khởi động nhanh (Quick Start)

```bash
# 1. Đăng ký giọng nói lần đầu
python auth/enroll.py --user alice --samples 5

# 2. Chạy daemon nền
nohup python voice_auth_daemon.py > /dev/null 2>&1 &

# Hoặc chạy foreground để debug
python voice_auth_daemon.py
```

---

## Lưu ý bảo mật

- Voiceprint lưu dưới dạng numpy array — cân nhắc mã hóa file `.npy` bằng `cryptography` nếu cần
- Log từ chối không lưu raw audio, chỉ lưu MD5 hash ngắn
- Threshold thấp hơn 0.70 không được khuyến nghị trong môi trường production
- Nếu triển khai multi-user, mỗi user cần voiceprint riêng; xem `enroll.py --help`

---

## Mở rộng trong tương lai

- **Wake word offline:** Tích hợp `openwakeword` với custom keyword "hey openclaw"
- **Anti-spoofing:** Thêm liveness detection để chống phát lại bản ghi âm
- **Multi-user:** Load tất cả voiceprint → xác định người nói trước, sau đó verify
- **OpenClaw REST:** Khi OpenClaw expose HTTP API, đổi `openclaw_mode: http` trong config
