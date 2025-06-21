import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    return {
    "model_size": config.get("model_size", "base"),
    "language": config.get("language", "auto"),
    "device": config.get("device", "cpu"),
    "compute_type": config.get("compute_type", "int8"),
    "hotkey": config.get("hotkey", "ctrl+shift+r"),
    "sample_rate": config.get("sample_rate", 16000),
    "channels": config.get("channels", 1),
    "enable_sounds": config.get("enable_sounds", True),
    "start_sound": config.get("start_sound", "start.mp3"),
    "stop_sound": config.get("stop_sound", "stop.mp3"),
    "sound_volume": config.get("sound_volume", 0.5),
    "chunk_duration": config.get("chunk_duration", 2.0),
    "silence_threshold": config.get("silence_threshold", 1.5),
    "live_typing": config.get("live_typing", True)
}
