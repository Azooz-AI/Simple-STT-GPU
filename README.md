# ⚡ Lightning-Fast Speech-to-Text Tool

> **🤯 Plot Twist:** This entire application was built by someone who had never coded before, with AI assistance! If you're reading this thinking "I could never build something like this" - think again! This is proof that modern AI has democratized software development. Anyone can build amazing tools now! 🚀

> **😅 The Real Story:** I got tired of typing long messages to ChatGPT, Claude, and Gemini. ChatGPT's desktop speech-to-text is painfully slow, Claude doesn't even have one, and Google Gemini's speech recognition is... let's just say it's not great. So I thought, "Why not build my own that's actually fast?" And here we are - a tool that transcribes 21 seconds of audio in 0.36 seconds. Take that, slow AI interfaces! 🎤⚡

A GPU-accelerated speech-to-text application with smart VRAM management and instant transcription.

## 🚀 Features

- **Lightning Fast**: GPU-accelerated transcription with RTX series GPUs
- **Smart VRAM Management**: Loads model during recording, unloads after use  
- **Distil-Whisper Powered**: 6x faster than standard models with same accuracy
- **Direct Typing**: Types directly into any application
- **System Tray**: Runs quietly in background
- **Customizable**: Easy configuration via YAML file

## 🎯 Real-World Performance

### Actual Test Results (RTX 3080 12GB):
- **Audio Duration:** 21 seconds
- **Transcription Time:** 0.359 seconds  
- **Speed:** **58.5x faster than real-time**
- **Words Processed:** 42 words in 359ms
- **Rate:** ~7,000 words per minute
- **VRAM Usage:** ~1.5GB (efficient!)

### Performance Comparison:
| Setup | 20s Audio Processing Time |
|-------|---------------------------|
| CPU (Before) | 5-6 seconds |
| GPU + Standard Model | ~2 seconds |
| **GPU + Distil-Whisper** | **~0.36 seconds** |

## 🛠️ Setup

### Prerequisites
- **GPU:** NVIDIA RTX series (tested on RTX 3080)
- **VRAM:** 2GB minimum  
- **OS:** Windows 10/11
- **Python:** 3.8+

### Installation

1. **Clone or download this repository**

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install CUDA-enabled PyTorch:**
   ```bash
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model_size: "Systran/faster-distil-whisper-large-v3"
device: "cuda"              # Use "cpu" if no GPU
compute_type: "float16"     # GPU setting
language: "auto"            # or specify "en"

# Controls
hotkey: "ctrl+shift+r"      # Recording hotkey

# Audio settings
sample_rate: 16000
channels: 1

# Sound effects
enable_sounds: true
start_sound: "start.mp3"
stop_sound: "stop.mp3"
sound_volume: 0.1
```

## 🎮 Usage

1. **Start the app:** `python main.py`
2. **Look for gray microphone icon** in system tray
3. **Recording:** Press `Ctrl+Shift+R` to start/stop recording
4. **Icon turns red** while recording
5. **Speak naturally** - model loads in background while you talk
6. **Text appears instantly** in your active window when you stop
7. **Quit:** Right-click tray icon → Quit

## 📁 Project Structure

```
STT-Windows-Tool/
├── main.py              # Main application
├── settings.py          # Configuration loader  
├── config.yaml          # User settings
├── requirements.txt     # Dependencies
├── start.mp3           # Recording start sound
├── stop.mp3            # Recording stop sound
├── .gitignore          # Git ignore file
├── README.md           # This file
└── LICENSE             # MIT License
```

## 🎵 Sound Files

The app includes notification sounds:
- `start.mp3` - Plays when recording starts
- `stop.mp3` - Plays when recording stops
- Volume configurable in `config.yaml`

## 🔧 Troubleshooting

### Common Issues:

**"CUDA not available" error:**
- Make sure you have NVIDIA GPU drivers installed
- Install CUDA-enabled PyTorch (see installation steps)
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**"Model not found" error:**
- First run downloads models automatically
- Requires internet connection
- Models cached in `~/.cache/huggingface/`

**Recording not working:**
- Check microphone permissions
- Try different hotkey in `config.yaml`
- Run as administrator if needed

**Slow performance:**
- Make sure `device: "cuda"` in config
- Check GPU memory usage
- Consider using smaller model for older GPUs

## 🏗️ Technical Details

- **Engine:** faster-whisper with CTranslate2
- **Model:** Distil-Whisper (knowledge-distilled for speed)
- **Audio Processing:** sounddevice + scipy
- **UI:** pystray (system tray)
- **Typing:** PyAutoGUI (direct text input)

## 🤝 Contributing

This project was built with AI assistance to demonstrate how modern tools democratize software development. Feel free to:

- Submit bug reports
- Suggest improvements  
- Fork and modify for your needs
- Share your performance results

## 📊 System Requirements

### Minimum:
- NVIDIA GTX 1060 or better
- 2GB VRAM
- 4GB RAM
- Windows 10

### Recommended:
- NVIDIA RTX 3060 or better  
- 6GB+ VRAM
- 8GB+ RAM
- Windows 11

## 🙏 Acknowledgments

- **OpenAI** for the original Whisper model
- **Hugging Face** for Distil-Whisper optimization
- **SYSTRAN** for faster-whisper implementation
- **AI assistance** for development guidance

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Built with ❤️ and AI - Proving that anyone can create powerful applications!**
