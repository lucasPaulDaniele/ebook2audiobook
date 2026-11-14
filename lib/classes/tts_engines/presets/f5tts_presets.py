import os
from lib.conf import models_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings

# weights downloaded by tools/install_f5tts.ps1
f5tts_dir = os.path.join(models_dir, 'f5tts')

models = {
    "internal": {
        "lang": "fra",
        # community French fine-tune of F5TTS_Base (LibriVox French), CC-BY-NC-4.0
        "repo": "RASPIAUDIO/F5-French-MixedSpeakers-reduced",
        "arch": "F5TTS_Base",
        "ckpt": os.path.join(f5tts_dir, 'fr', 'model_last_reduced.pt'),
        "vocab": os.path.join(f5tts_dir, 'fr', 'vocab.txt'),
        "vocoder_dir": os.path.join(f5tts_dir, 'vocos'),
        "voice": default_engine_settings[TTS_ENGINES['F5TTS']]['voice'],
        "files": default_engine_settings[TTS_ENGINES['F5TTS']]['files'],
        "samplerate": default_engine_settings[TTS_ENGINES['F5TTS']]['samplerate']
    }
}
