import sounddevice as sd
import numpy as np
import time
import platform
import keyboard
from settings import load_config
from faster_whisper import WhisperModel
import os
import pystray
from PIL import Image, ImageDraw, ImageFont
import threading
import pygame
import math
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stt_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
ICON_SIZES = [16, 32, 48, 64, 128, 256]
ICON_FILENAME = 'stt_icon_idle.ico'
AUDIO_INT16_MAX = 32767
PULSE_ANIMATION_FPS = 0.1  # 10fps animation
PULSE_SPEED = 0.3
DEFAULT_ICON_SIZE = 64

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_buffer = []
        self.recording_start = None
        self.stream = None
        self.model = None  # Don't load model immediately
    
    def load_model_if_needed(self):
        """Load model only when needed"""
        if self.model is None:
            logger.info("Loading Whisper model for GPU...")
            self.model = WhisperModel(
                config["model_size"],
                device=config["device"],
                compute_type=config["compute_type"]
            )
            logger.info("Model loaded to GPU")
    
    def unload_model(self):
        """Free up VRAM when not needed"""
        if self.model is not None:
            del self.model
            self.model = None
            # Force garbage collection to free VRAM
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Model unloaded from GPU, VRAM freed")
    
    def start_recording(self):
        """Start audio recording and load model in background"""
        self.recording = True
        self.audio_buffer = []
        self.recording_start = time.time()
        
        # Start loading model in background thread
        def load_model_background():
            self.load_model_if_needed()
        
        threading.Thread(target=load_model_background, daemon=True).start()
        logger.info("Model loading started in background...")
        
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.audio_buffer.append(indata.copy())
        
        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback)
            self.stream.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            self.recording = False
            return False
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.recording:
            return None
            
        duration = round(time.time() - self.recording_start, 2)
        self.recording = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        
        # Combine the chunks
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            logger.info(f"Recording duration: {duration}s")
            return audio_data
        return None

# Load settings from config.yaml
config = load_config()

tray_icon = None  # Global tray reference for color changes
pulse_timer = None  # Animation timer
pulse_phase = 0  # Animation state
    

def create_single_icon(size, recording_state=False, pulse_intensity=0):
    """Create simple icon - just colored circles"""
    img = Image.new('RGBA', (size, size), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    radius = int(size * 0.4)
    
    # Simple color logic
    if recording_state:
        # Red when recording
        color = (255, 0, 0)  # Bright red
    else:
        # Gray when idle  
        color = (100, 100, 100)  # Gray
    
    # Draw simple circle
    draw.ellipse([center - radius, center - radius,
                 center + radius, center + radius], fill=color)
    
    return img


# Create audio recorder
audio_recorder = AudioRecorder()

HOTKEY = config["hotkey"]
SAMPLE_RATE = config["sample_rate"]
CHANNELS = config["channels"]
ENABLE_SOUNDS = config["enable_sounds"]

# Modern minimalist tray icon creation 🎨
def create_tray_icon(recording_state: bool = False, pulse_intensity: float = 0) -> Image.Image:
    """Create icon with optional pulsing effect"""
    return create_single_icon(DEFAULT_ICON_SIZE, recording_state, pulse_intensity)

def start_pulse_animation():
    """Simple version - no pulsing for now"""
    global tray_icon
    if tray_icon:
        tray_icon.icon = create_tray_icon(recording_state=True)

def stop_pulse_animation():
    """Simple version - just change icon back"""
    global tray_icon
    if tray_icon:
        tray_icon.icon = create_tray_icon(recording_state=False)

def update_tray_icon(recording_state=False):
    """Update tray icon state - simplified"""
    global tray_icon
    if tray_icon:
        if recording_state:
            start_pulse_animation()
        else:
            stop_pulse_animation()

def play_sound(sound_file: str) -> None:
    if config["enable_sounds"]:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.set_volume(config["sound_volume"])
            pygame.mixer.music.play()
        except Exception as e:
            logger.error(f"Sound failed: {e}")

def notify(title: str, message: str, sound_type: str = "start") -> None:
    """Enhanced notification with custom sounds"""
    if sound_type == "start":
        play_sound(config["start_sound"])
    elif sound_type == "stop":
        play_sound(config["stop_sound"])

def transcribe_audio(audio_data: np.ndarray) -> None:
    """Handle the transcription process"""
    import tempfile
    import scipy.io.wavfile as wavfile
    import pyautogui
    logger.info("Got audio, transcribing...")
    
    # Convert to int16
    int_audio = (audio_data * AUDIO_INT16_MAX).astype("int16")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            wavfile.write(temp_path, SAMPLE_RATE, int_audio)
    except Exception as e:
        logger.error(f"Failed to create audio file: {e}")
        return
    
    logger.info(f"Saved temp WAV to: {temp_path}")
    
    if not os.path.exists(temp_path):
        logger.error("Temp WAV file not found.")
        return
    
    # Transcribe and handle result
    try:
        # Model should be loaded by now from background loading
        if config["language"] == "auto":
            segments, info = audio_recorder.model.transcribe(temp_path)
        else:
            segments, info = audio_recorder.model.transcribe(temp_path, language=config["language"])

        # Unload model after use to free VRAM
        audio_recorder.unload_model()
        
        raw_text = "".join([segment.text for segment in segments]).strip()
        
        if raw_text:
            logger.info(f"Transcribed Text: {raw_text}")
            
            # Type text directly
            pyautogui.write(raw_text)
        else:
            logger.warning("No speech detected. Transcription is empty.")
            
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
    
    # Cleanup temp file
    try:
        os.unlink(temp_path)
    except:
        pass

def toggle_recording():
    global tray_icon

    if not audio_recorder.recording:
        # START recording
        update_tray_icon(recording_state=True)
        notify("Speech-to-Text", "🎙️ Recording started...", "start")
        
        success = audio_recorder.start_recording()
        if not success:
            update_tray_icon(recording_state=False)
            return

    else:
        # STOP recording
        update_tray_icon(recording_state=False)
        notify("Speech-to-Text", "🛑 Recording stopped", "stop")
        
        audio_data = audio_recorder.stop_recording()
        if audio_data is not None:
            transcribe_audio(audio_data)

# Tray setup 🥷
def quit_app(icon, item):
    """Clean exit function"""
    global pulse_timer
    
    # Stop recording if active
    if audio_recorder.recording:
        audio_recorder.stop_recording()
    
    # Cancel animation timer
    if pulse_timer:
        pulse_timer.cancel()
        pulse_timer = None
    
    # Stop tray icon
    icon.stop()
    
    logger.info("Application shutting down")
    os._exit(0)

def setup_tray():
    global tray_icon
    
    # Create simple icon
    icon_image = create_tray_icon(recording_state=False)
    
    menu = pystray.Menu(pystray.MenuItem("Quit", quit_app))
    
    tray_icon = pystray.Icon("STT Tool", icon_image, menu=menu)
    tray_icon.run()

# Register the hotkey
keyboard.add_hotkey(HOTKEY, toggle_recording)

logger.info(f"STT Tool running in tray. Press {HOTKEY.upper()} to record!")
logger.info("Icon: Gray = Idle | Pulsing Red = Recording")
setup_tray()