# Voice Auth Tray

> Ứng dụng desktop xác thực giọng nói chạy trên System Tray, kết nối với OpenClaw Gateway. Hoạt động tương tự **OpenClaw Tray** — chạy ngầm, lắng nghe microphone, xác thực người dùng bằng ECAPA-TDNN rồi gửi lệnh đến OpenClaw.

```
Microphone → Voice Auth (ECAPA-TDNN + AS-Norm) → STT (Parakeet) → OpenClaw Gateway
```

## Tính năng

- 🖥️ **System Tray** — chạy ngầm như OpenClaw Tray, icon thay đổi theo trạng thái
- 🔐 **Speaker verification** — ECAPA-TDNN + AS-Norm Z-score
- 📝 **STT local** — NVIDIA Parakeet-TDT-0.6B-v3 (offline)
- 🎙 **Enrollment GUI** — đăng ký giọng nói qua giao diện trực quan
- ⚙️ **Setup Wizard** — cấu hình Gateway URL + Token lần đầu
- 🤫 **Silent reject** — im lặng khi xác thực thất bại, chỉ ghi log

## Cài đặt

### 1. Tải về

```bash
git clone https://github.com/RobiRobin555/Voice_auth_skill.git
cd Voice_auth_skill
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

> NeMo (cho Parakeet STT) cài riêng:
> ```bash
> pip install nemo_toolkit[asr]
> ```

### 3. Chuẩn bị model và cohort

- Đặt file model ECAPA-TDNN `.pt` vào `models/`
- Đặt 50–200 file `.wav` người lạ vào `data/cohort/`

### 4. Chạy

```bash
python main.py
```

## Lần khởi động đầu tiên

Khi chạy `python main.py` lần đầu:

1. **Setup Wizard** hiện ra → nhập:
   - **Gateway URL**: `http://localhost:18789` (mặc định)
   - **Token**: lấy từ `openclaw config get gateway.token`
   - **Tên người dùng**
   - Nhấn "Test kết nối" để kiểm tra

2. **Enrollment GUI** hiện ra → ghi 5 mẫu giọng nói (3 giây mỗi mẫu)

3. Ứng dụng thu nhỏ vào **System Tray** → daemon lắng nghe ngầm

## Sử dụng

Sau khi setup xong:
- Ứng dụng chạy ngầm, icon trên taskbar
- **Nói vào microphone** → hệ thống tự xác thực → nếu đúng giọng → STT → gửi lệnh đến OpenClaw
- **Chuột phải vào tray icon** để:
  - Đăng ký lại giọng nói
  - Sửa cài đặt Gateway
  - Mở thư mục logs
  - Thoát

## Cấu trúc thư mục

```
Voice_auth_skill/
├── main.py                     ← Chạy ứng dụng
├── app/
│   ├── tray.py                 ← System Tray icon + menu
│   ├── setup_wizard.py         ← GUI cài đặt Gateway
│   ├── enrollment_gui.py       ← GUI đăng ký giọng nói
│   ├── daemon_thread.py        ← Background listener
│   ├── openclaw_client.py      ← HTTP client → OpenClaw
│   └── config_manager.py       ← Settings persistent
├── auth/
│   ├── verifier.py             ← ECAPA-TDNN + AS-Norm
│   └── voiceprints/            ← Voiceprint .npy
├── stt/
│   └── parakeet_engine.py      ← Parakeet-TDT wrapper
├── models/                     ← Model .pt
└── data/cohort/                ← File .wav cohort
```

## Yêu cầu hệ thống

| | Tối thiểu | Khuyến nghị |
|--|-----------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 4 GB | 8 GB |
| GPU | Không bắt buộc | CUDA (NVIDIA) |
| Microphone | Bắt buộc | USB mic |
| OS | Windows 10+ | Windows 11 |

## License

MIT
