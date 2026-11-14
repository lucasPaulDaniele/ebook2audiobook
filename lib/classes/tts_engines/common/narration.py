"""
Natural narration layer shared by the voice-cloning engines (XTTS, F5-TTS, Chatterbox).

Each chunk of text is synthesised separately; what makes the result sound like one
continuous reading instead of glued clips lives here:
- silence after a chunk chosen from how its text ends (comma < ; < sentence < paragraph)
- trimming, micro-click removal, level matching and short fades on every chunk
- duration based QA: a take whose length does not match its text (babbling tail, loop,
  skipped words) is generated again, the most plausible take is kept
- deterministic seeds derived from the text, so renders and resumed runs are reproducible

An engine only implements _infer_once(text, seed, expected_s) returning a mono waveform.
"""

import math

from typing import Any


class NaturalNarration:

    # silence appended after a synthesised chunk, by how its text ends (seconds, uniform range).
    # The chunk audio is trimmed, so these are the real gaps the listener hears; a [break]
    # (paragraph) or [pause] (heading) SML tag adds its own silence on top. Values follow
    # French read narration (comma ~0.2 s < ; ~0.35 s < sentence ~0.5 s < paragraph ~0.9 s).
    PAUSE_AFTER = {
        'sentence': (0.42, 0.60),   # .
        'question': (0.46, 0.66),   # ? !
        'ellipsis': (0.60, 0.88),   # …
        'strong': (0.28, 0.40),     # ; :
        'comma': (0.14, 0.24),      # ,
        'none': (0.06, 0.10),       # cut without punctuation (rare, last resort)
    }
    # target active-speech level of every chunk, a level jump between two chunks is the
    # most audible "glue" cue
    TARGET_LEVEL_DB = -20.0
    MAX_GAIN_DB = 6.0
    # chunk QA: a duration far from the expected one means babbling tails, loops or
    # skipped words, the chunk is generated again and the most plausible take is kept
    QA_MAX_TAKES = 3
    QA_DEFAULT_LETTERS_PER_S = 12.0

    # ---------------------------------------------------------------- engine hook

    def _infer_once(self, text:str, seed:int, expected_s:float)->Any:
        """Return a mono waveform (tensor or ndarray) at self.params['samplerate']."""
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _text_seed(text:str)->int:
        # same text -> same take: renders are reproducible, resumed conversions identical
        import zlib
        return zlib.crc32(text.encode('utf-8')) & 0x7fffffff

    def _pause_after(self, text:str)->float:
        import numpy as np
        end = text.rstrip().rstrip('»"”’)]').rstrip()
        last = end[-1:] if end else ''
        if last == '.':
            kind = 'sentence'
        elif last in '?!':
            kind = 'question'
        elif last == '…':
            kind = 'ellipsis'
        elif last in ';:':
            kind = 'strong'
        elif last == ',':
            kind = 'comma'
        else:
            kind = 'none'
        lo, hi = self.PAUSE_AFTER[kind]
        rng = np.random.default_rng(self._text_seed(text))
        return round(float(rng.uniform(lo, hi)), 3)

    def _letters_per_s(self)->float:
        import numpy as np
        history = self.params.setdefault('qa_rate_history', [])
        if len(history) < 8:
            return self.QA_DEFAULT_LETTERS_PER_S
        return float(np.median(history[-200:]))

    def _qa_score(self, text:str, audio:'torch.Tensor')->tuple:
        """0 = perfect. Returns (score, reason), reason is None when the take is plausible."""
        sr = self.params['samplerate']
        letters = sum(1 for c in text if c.isalnum())
        duration = audio.shape[-1] / sr
        expected = letters / self._letters_per_s()
        score = abs(math.log(max(duration, 0.05) / max(expected, 0.05)))
        if duration > expected * 1.45 + 0.6:
            return score, f'too long ({duration:.1f}s for ~{expected:.1f}s)'
        if letters >= 12 and duration < expected * 0.62 - 0.2:
            return score, f'too short ({duration:.1f}s for ~{expected:.1f}s)'
        frame = int(0.02 * sr)
        n = audio.shape[-1] // frame
        if n == 0:
            return score, None
        rms = audio[..., :n * frame].reshape(n, frame).pow(2).mean(dim=1).sqrt()
        silent = (rms < rms.max() * 0.02).tolist()
        # a long silence inside a chunk is a hallucinated hesitation or a skipped part
        run = longest = 0
        for flag in silent:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        if longest * 0.02 > 1.1:
            return score + 1.0, f'inner silence {longest * 0.02:.1f}s'
        # a short burst isolated after a gap at the very end is the typical tail
        # (extra syllable, breath-word) generated after the text is finished;
        # micro bursts under 0.15 s are already cut by _cut_tail_click()
        if letters >= 12:
            last_voiced = max((i for i, flag in enumerate(silent) if not flag), default=-1)
            gap_end = last_voiced
            while gap_end >= 0 and not silent[gap_end]:
                gap_end -= 1
            burst = (last_voiced - gap_end) * 0.02
            gap = 0
            i = gap_end
            while i >= 0 and silent[i]:
                gap += 1
                i -= 1
            if 0.15 <= burst < 0.45 and gap * 0.02 >= 0.25 and i >= 0:
                return score + 0.5, f'tail burst {burst:.2f}s after {gap * 0.02:.2f}s gap'
        return score, None

    def _cut_tail_click(self, audio:'torch.Tensor')->'torch.Tensor':
        """Remove a micro burst (breath, click) isolated after the last word."""
        sr = self.params['samplerate']
        frame = int(0.02 * sr)
        n = audio.shape[-1] // frame
        if n < 10:
            return audio
        rms = audio[:n * frame].reshape(n, frame).pow(2).mean(dim=1).sqrt()
        silent = (rms < rms.max() * 0.02).tolist()
        last = max((i for i, flag in enumerate(silent) if not flag), default=-1)
        start = last
        while start >= 0 and not silent[start]:
            start -= 1
        burst = (last - start) * 0.02
        gap_end = start
        while start >= 0 and silent[start]:
            start -= 1
        gap = (gap_end - start) * 0.02
        if start >= 0 and 0 < burst < 0.15 and gap >= 0.3:
            keep = min(audio.shape[-1], (start + 1) * frame + int(0.06 * sr))
            return audio[:keep]
        return audio

    def _postprocess(self, audio:'torch.Tensor')->'torch.Tensor':
        """Trim, remove a trailing click, level-match and fade one chunk."""
        import torch
        from lib.classes.tts_engines.common.audio import trim_audio
        sr = self.params['samplerate']
        audio = trim_audio(audio.squeeze(), sr, 0.001, 0.006)
        if audio.numel() == 0:
            return audio.unsqueeze(0)
        audio = self._cut_tail_click(audio)
        frame = int(0.02 * sr)
        n = audio.shape[-1] // frame
        if n > 0:
            rms = audio[:n * frame].reshape(n, frame).pow(2).mean(dim=1).sqrt() + 1e-9
            db = 20 * torch.log10(rms)
            active = db[db > db.max() - 30]
            level = float(active.mean()) if active.numel() else float(db.mean())
            gain_db = max(-self.MAX_GAIN_DB, min(self.MAX_GAIN_DB, self.TARGET_LEVEL_DB - level))
            audio = audio * (10 ** (gain_db / 20))
            peak = float(audio.abs().max())
            if peak > 0.97:
                audio = audio * (0.97 / peak)
        audio = audio.clone()
        fade_in = min(int(0.008 * sr), audio.shape[-1] // 4)
        fade_out = min(int(0.025 * sr), audio.shape[-1] // 4)
        if fade_in > 0:
            audio[:fade_in] *= torch.linspace(0.0, 1.0, fade_in)
        if fade_out > 0:
            audio[-fade_out:] *= torch.linspace(1.0, 0.0, fade_out)
        return audio.unsqueeze(0)

    def _synthesize_checked(self, text_chunk:str)->tuple:
        """Synthesise one chunk with QA and retries. Returns (tensor [1, n], error)."""
        import numpy as np
        import torch
        from lib.classes.tts_engines.common.audio import is_audio_data_valid
        letters = sum(1 for c in text_chunk if c.isalnum())
        expected = letters / self._letters_per_s()
        # seed_offset lets a repair pass draw takes different from the first conversion
        seed = self._text_seed(text_chunk) + int(self.params.get('seed_offset', 0))
        best = None
        for take in range(1, self.QA_MAX_TAKES + 1):
            take_seed = seed + 1000 * (take - 1)
            torch.manual_seed(take_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(take_seed)
            audio_part = self._infer_once(text_chunk, take_seed, expected)
            if audio_part is None or len(audio_part) == 0 or not is_audio_data_valid(audio_part):
                return None, 'audio_part not valid'
            if isinstance(audio_part, np.ndarray):
                audio_part = torch.from_numpy(audio_part.astype(np.float32))
            if torch.is_tensor(audio_part):
                audio_part = audio_part.detach().float().cpu()
            part_tensor = self._postprocess(audio_part)
            if part_tensor.numel() == 0:
                return None, 'part_tensor not valid'
            score, reason = self._qa_score(text_chunk, part_tensor)
            if best is None or score < best[0]:
                best = (score, part_tensor)
            if reason is None:
                break
            print(f'\n[QA] take {take}/{self.QA_MAX_TAKES} rejected: {reason} -> "{text_chunk[:60]}"')
        score, part_tensor = best
        if letters >= 40 and score < 0.35:
            self.params.setdefault('qa_rate_history', []).append(letters / (part_tensor.shape[-1] / self.params['samplerate']))
        return part_tensor, None
