from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.classes.tts_engines.common.narration import NaturalNarration

import json
import tempfile


class F5TTS(NaturalNarration, TTSUtils, TTSRegistry, name='f5tts'):
    """
    F5-TTS (flow matching, non autoregressive: no babbling tail or loop by design) with a
    French checkpoint. The model runs in its own environment (python_env_f5, installed by
    tools/install_f5tts.ps1) through lib/classes/tts_engines/workers/f5tts_worker.py.

    The voice reference must be short (<= 12 s) and come with its exact transcript:
    "<voice>.f5ref.json" next to the voice file ({"wav": ..., "text": ...}) is used when it
    exists, otherwise a clip is cut from the voice at a pause and transcribed once with
    Whisper (CPU), then the result is saved next to the voice for the following runs.
    """

    REF_MAX_S = 11.5
    REF_MIN_S = 5.0

    def __init__(self, session:DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.tts_zs_key = None
            self.audio_segments = []
            self.models = load_engine_presets(self.session['tts_engine'])
            self.params = {}
            self.language = self.session.get('language')
            self.language_iso1 = self.session.get('language_iso1')
            if self.session.get('translate_enabled'):
                if self.session.get('translate'):
                    self.language = self.session['translate']
                if self.session.get('translate_iso1'):
                    self.language_iso1 = self.session['translate_iso1']
            fine_tuned = self.session.get('fine_tuned') or 'internal'
            if fine_tuned not in self.models:
                raise ValueError(f'Invalid fine_tuned model {fine_tuned}. Available models: {list(self.models.keys())}')
            self.model_cfg = self.models[fine_tuned]
            self.params['samplerate'] = self.model_cfg['samplerate']
            cfg = default_engine_settings[TTS_ENGINES['F5TTS']]
            self.synth_params = {
                key: cast(self.session.get(f'f5tts_{key}') if self.session.get(f'f5tts_{key}') is not None else cfg[key])
                for key, cast in {'nfe_step': int, 'cfg_strength': float, 'sway_sampling_coef': float, 'speed': float}.items()
            }
            self.device = 'cuda' if self.session['device'] in [devices['CUDA']['proc'], devices['ROCM']['proc'], devices['JETSON']['proc']] else 'cpu'
            self.refs = {}
            self.worker = None
            self.tmp_dir = tempfile.mkdtemp(prefix='f5tts_')
            self.engine = self.load_engine()
        except Exception as e:
            raise ValueError(f'__init__() error: {e}')

    # ---------------------------------------------------------------- worker

    def load_engine(self)->Any:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        python = os.path.join(root, 'python_env_f5', 'Scripts' if os.name == 'nt' else 'bin', 'python.exe' if os.name == 'nt' else 'python')
        if not os.path.exists(python):
            raise RuntimeError('F5-TTS is not installed: run tools/install_f5tts.ps1 (Windows) to create python_env_f5')
        for key in ('ckpt', 'vocab'):
            if not os.path.exists(self.model_cfg[key]):
                raise RuntimeError(f"F5-TTS model file missing: {self.model_cfg[key]} (run tools/install_f5tts.ps1)")
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workers', 'f5tts_worker.py')
        msg = f'Loading TTS {self.tts_key} model (F5-TTS worker), it takes a while, please be patient…'
        print(msg)
        self.worker = subprocess.Popen(
            [python, '-u', worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding='utf-8', bufsize=1, env=dict(os.environ, PYTHONIOENCODING='utf-8')
        )
        self._request(cmd='load', ckpt=self.model_cfg['ckpt'], vocab=self.model_cfg['vocab'],
                      vocoder_dir=self.model_cfg.get('vocoder_dir'), arch=self.model_cfg.get('arch', 'F5TTS_Base'),
                      device=self.device)
        print(f'TTS {self.tts_key} Loaded!')
        return self.worker

    def _request(self, **req)->dict:
        if self.worker is None or self.worker.poll() is not None:
            raise RuntimeError('F5-TTS worker is not running')
        self.worker.stdin.write(json.dumps(req, ensure_ascii=False) + '\n')
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('F5-TTS worker stopped unexpectedly (see the console for its error)')
        res = json.loads(line)
        if not res.get('ok'):
            raise RuntimeError(f"F5-TTS worker error: {res.get('error')}")
        return res

    def close(self)->None:
        if self.worker is not None:
            try:
                if self.worker.poll() is None:
                    self.worker.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                    self.worker.stdin.flush()
                    self.worker.wait(timeout=20)
            except Exception:
                self.worker.kill()
            self.worker = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- voice reference

    def _reference(self, voice:str)->tuple[str, str]:
        if voice in self.refs:
            return self.refs[voice]
        sidecar = os.path.splitext(voice)[0] + '.f5ref.json'
        if os.path.exists(sidecar):
            with open(sidecar, encoding='utf-8') as f:
                data = json.load(f)
            wav = data['wav'] if os.path.isabs(data['wav']) else os.path.join(os.path.dirname(sidecar), data['wav'])
            self.refs[voice] = (wav, data['text'])
            return self.refs[voice]
        wav, text = self._make_reference(voice)
        try:
            with open(sidecar, 'w', encoding='utf-8') as f:
                json.dump({'wav': os.path.basename(wav), 'text': text}, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
        self.refs[voice] = (wav, text)
        return self.refs[voice]

    def _make_reference(self, voice:str)->tuple[str, str]:
        """Cut a 5-11.5 s clip ending at a pause and transcribe it (Whisper turbo, CPU)."""
        import numpy as np
        import soundfile as sf
        import librosa
        sr = 24000
        y, _ = librosa.load(voice, sr=sr, mono=True)
        end = len(y)
        if len(y) / sr > self.REF_MAX_S:
            frame = int(0.02 * sr)
            n = len(y) // frame
            rms = np.sqrt((y[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
            silent = rms < rms.max() * 0.03
            lo, hi = int(self.REF_MIN_S / 0.02), int(self.REF_MAX_S / 0.02)
            # the longest pause between 5 and 11.5 s is the most natural place to stop
            best, best_len, run = hi, 0, 0
            for i in range(lo, min(hi, n)):
                run = run + 1 if silent[i] else 0
                if run > best_len:
                    best, best_len = i, run
            end = best * frame
        clip = np.concatenate([y[:end], np.zeros(int(0.8 * sr), dtype=y.dtype)])
        out = os.path.splitext(voice)[0] + '.f5ref.wav'
        sf.write(out, clip, sr, subtype='PCM_16')
        from lib.classes.asr_verifier import Whisper, models_available
        _, turbo = models_available(os.path.join(os.path.dirname(tts_dir)))
        if turbo is None:
            raise RuntimeError(f'F5-TTS needs the transcript of its voice reference: write it in {os.path.splitext(voice)[0]}.f5ref.json')
        asr = Whisper(turbo, 'cpu', self.language_iso1 or 'fr')
        text = asr.transcribe(out)
        asr.close()
        text = text.strip()
        if text and text[-1] not in '.!?…':
            text += '.'
        return out, text

    # ---------------------------------------------------------------- synthesis

    def _infer_once(self, text:str, seed:int, expected_s:float)->Any:
        import soundfile as sf
        ref_wav, ref_text = self.params['current_ref']
        out = os.path.join(self.tmp_dir, 'take.wav')
        self._request(cmd='synth', text=text, ref_audio=ref_wav, ref_text=ref_text, out=out, seed=seed, **self.synth_params)
        audio, sr = sf.read(out, dtype='float32')
        if sr != self.params['samplerate']:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.params['samplerate'])
        return audio

    def convert(self, sentence_file:str, sentence:str, **kwargs)->tuple:
        try:
            import torch
            if not self.worker:
                return False, f"TTS engine {self.session['tts_engine']} failed to load!"
            self.params['block_voice'] = kwargs.get('block_voice', self.session['voice'])
            voice = self.params.get('inline_voice') or self.params['block_voice']
            if not voice or not os.path.exists(voice):
                return False, 'F5-TTS needs a voice file to clone (choose a voice)'
            self.params['current_voice'] = voice
            self.params['current_ref'] = self._reference(voice)
            self.audio_segments = []
            for part in self._split_sentence_on_sml(sentence):
                part = part.strip()
                if not part:
                    continue
                if SML_TAG_PATTERN.fullmatch(part):
                    success, error = self._convert_sml(part)
                    if not success:
                        return False, error
                    if self.params.get('inline_voice'):
                        self.params['current_ref'] = self._reference(self.params['inline_voice'])
                    continue
                if not any(c.isalnum() for c in part):
                    continue
                part_tensor, error = self._synthesize_checked(part)
                if part_tensor is None:
                    return False, error
                self.audio_segments.append(part_tensor)
                silence_time = self._pause_after(part)
                if silence_time > 0:
                    self.audio_segments.append(torch.zeros(1, int(self.params['samplerate'] * silence_time)))
            if self.audio_segments:
                segment_tensor = torch.cat(self.audio_segments, dim=-1)
                if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                    return False, f'audio_save() error: cannot save {sentence_file}'
                self.audio_segments = []
            return True, None
        except Exception as e:
            self.audio_segments = []
            return False, self.log_exception(f'{self.__class__.__name__}.convert()', e)

    def create_vtt(self, all_sentences:list)->bool:
        return bool(self._build_vtt_file(all_sentences))
