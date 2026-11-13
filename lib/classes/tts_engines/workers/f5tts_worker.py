"""
F5-TTS worker, run with the python of python_env_f5 (see tools/install_f5tts.ps1).

F5-TTS pins its own torch/transformers versions, so it lives in a separate environment and
ebook2audiobook talks to it through this process: one JSON request per line on stdin, one
JSON answer per line on stdout. Everything else the libraries print goes to stderr.

  {"cmd": "load", "ckpt": ..., "vocab": ..., "vocoder_dir": ..., "arch": "F5TTS_Base", "device": "cuda"}
  {"cmd": "synth", "text": ..., "ref_audio": ..., "ref_text": ..., "out": "x.wav", "seed": 1,
   "nfe_step": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "speed": 1.0}
  {"cmd": "quit"}
"""

import json
import os
import sys
import traceback

# the protocol owns the real stdout, any library print goes to stderr
PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
sys.stdout = sys.stderr


def answer(**kwargs)->None:
    PROTOCOL.write(json.dumps(kwargs, ensure_ascii=False) + '\n')
    PROTOCOL.flush()


def load(req:dict):
    import torch
    import f5_tts.infer.utils_infer as ui
    # F5-TTS switches to fp16 by itself on any GPU with compute capability >= 7, which
    # gives silent or NaN audio on GTX 16xx cards: always load the weights in fp32
    original = ui.load_checkpoint
    ui.load_checkpoint = lambda model, ckpt_path, device, dtype=None, use_ema=True: \
        original(model, ckpt_path, device, dtype=torch.float32, use_ema=use_ema)
    from f5_tts.api import F5TTS
    device = req.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    return F5TTS(
        model=req.get('arch', 'F5TTS_Base'),
        ckpt_file=req['ckpt'],
        vocab_file=req['vocab'],
        vocoder_local_path=req.get('vocoder_dir'),
        device=device
    )


def synth(tts, req:dict)->int:
    import soundfile as sf
    wav, sr, _ = tts.infer(
        ref_file=req['ref_audio'],
        ref_text=req['ref_text'],
        gen_text=req['text'],
        nfe_step=int(req.get('nfe_step', 32)),
        cfg_strength=float(req.get('cfg_strength', 2.0)),
        sway_sampling_coef=float(req.get('sway_sampling_coef', -1.0)),
        speed=float(req.get('speed', 1.0)),
        cross_fade_duration=0.0,
        remove_silence=False,
        seed=int(req['seed']) if req.get('seed') is not None else None,
        show_info=lambda *a, **k: None,
        progress=None
    )
    sf.write(req['out'], wav, sr, subtype='FLOAT')
    return int(sr)


def main()->None:
    tts = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get('cmd')
            if cmd == 'quit':
                answer(ok=True)
                break
            if cmd == 'load':
                tts = load(req)
                answer(ok=True)
            elif cmd == 'synth':
                if tts is None:
                    raise RuntimeError('model not loaded')
                sr = synth(tts, req)
                answer(ok=True, sr=sr)
            else:
                raise ValueError(f'unknown command {cmd!r}')
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            answer(ok=False, error=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
