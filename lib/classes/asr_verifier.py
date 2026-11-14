"""
Whisper verification of every synthesised chunk, with automatic repair.

XTTS can drop words, garble a syllable or add a babbling tail with a plausible duration,
which the in-engine duration QA cannot see. A transcription can.

1. During the conversion a separate CPU process (Whisper small, 2 threads so XTTS keeps the
   machine) transcribes each finished chapter and flags suspicious chunks (watch mode).
2. When all chapters are synthesised, XTTS is unloaded and Whisper large-v3-turbo on the GPU
   checks what the watcher did not reach and confirms the flagged chunks.
3. Each confirmed chunk is generated again (3 takes with new seeds), every take is scored by
   Whisper and the best one replaces the original if it is better.
4. The caller recombines the touched chapters and writes the report.

GPU memory is never shared: XTTS and Whisper turbo are loaded one after the other.
"""

import json
import os
import subprocess
import sys
import time
import unicodedata

import regex as re

SR = 16000
# Whisper always encodes a 30 s window: short chunks are packed together (1 s of silence
# between them) and the text is given back to each chunk from the segment timestamps
PACK_MAX_S = 27.0
PACK_GAP_S = 1.0
WATCH_THREADS = 4
WATCH_FLAG_CER = 0.15     # Whisper small: flag for a closer look
# real hallucinations (tail babble, inserted or swapped words) often score 3-6 %: a repair
# only replaces a chunk when a new take scores better, so a low threshold is safe
CONFIRM_CER = float(os.environ.get('E2A_ASR_CONFIRM_CER', 0.04))
REPAIR_TAKES = 3
# automatic second chance: a chunk whose best repair take is still above this CER gets
# SECOND_CHANCE_TAKES more takes with new seeds
SECOND_CHANCE_CER = float(os.environ.get('E2A_SECOND_CHANCE_CER', 0.15))
SECOND_CHANCE_TAKES = int(os.environ.get('E2A_SECOND_CHANCE_TAKES', 8))


def same_transcription(hyps:list[str])->bool:
    """All the takes heard the same way: when several different takes of a chunk get exactly
    the same transcription, Whisper is the one that is wrong (rare passé simple, foreign
    names, "16" for "seize"), not the voice."""
    texts = {norm_text(h) for h in hyps}
    return len(hyps) >= 2 and len(texts) == 1


# ---------------------------------------------------------------- text

def norm_text(s:str)->str:
    s = unicodedata.normalize('NFKC', s).lower()
    s = s.replace('’', "'").replace('œ', 'oe').replace('æ', 'ae')
    s = re.sub(r"[^\p{L}\p{N}' ]+", ' ', s)
    return ' '.join(s.split())


def spell_numbers(s:str, lang_iso1:str)->str:
    # Whisper writes "192e" / "2012" where the TTS text says the words
    from num2words2 import num2words

    def _ord(m):
        try:
            n = int(m.group(1))
            if lang_iso1 == 'fr' and n == 1:
                return 'premier'
            return num2words(n, to='ordinal', lang=lang_iso1)
        except Exception:
            return m.group(0)

    def _card(m):
        try:
            return num2words(int(m.group(0)), lang=lang_iso1)
        except Exception:
            return m.group(0)

    if lang_iso1 == 'fr':
        s = re.sub(r'\b(\d+)\s?(?:e|ème|er|re)\b', _ord, s)
    return re.sub(r'\d+', _card, s)


def edit_distance(a:str, b:str)->int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref:str, hyp:str, lang_iso1:str='fr')->float:
    r = norm_text(ref).replace('-', ' ')
    h = norm_text(spell_numbers(hyp, lang_iso1)).replace('-', ' ')
    r, h = ' '.join(r.split()), ' '.join(h.split())
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ---------------------------------------------------------------- whisper

class Whisper:
    """Whisper without transformers.pipeline (its audio stack needs torchcodec)."""

    def __init__(self, model_dir:str, device:str, lang_iso1:str, threads:int|None=None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.lang = lang_iso1
        # fp16 Whisper outputs garbage on GTX 16xx cards, fp32 is used everywhere
        self.proc = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()

    def transcribe(self, wav_path:str)->str:
        return self.transcribe_array(load_16k(wav_path))

    def transcribe_array(self, y, timestamps:bool=False):
        feats = self.proc(y, sampling_rate=SR, return_tensors='pt').input_features.to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(feats, language=self.lang, task='transcribe', return_timestamps=timestamps)
        tok = self.proc.tokenizer
        if not timestamps:
            return tok.decode(out[0], skip_special_tokens=True).strip()
        dec = tok.decode(out[0], output_offsets=True, skip_special_tokens=True)
        return [(o['timestamp'][0], o['timestamp'][1], o['text']) for o in dec.get('offsets', [])]

    def transcribe_packed(self, chunks:list[dict])->dict:
        """chunks: [{'wav', ...}] in reading order -> {wav: transcription}."""
        import numpy as np
        audios = [(c['wav'], load_16k(c['wav'])) for c in chunks]
        result = {}
        packs, cur, cur_len = [], [], 0.0
        for wav, y in audios:
            dur = len(y) / SR
            if cur and cur_len + PACK_GAP_S + dur > PACK_MAX_S:
                packs.append(cur)
                cur, cur_len = [], 0.0
            cur.append((wav, y))
            cur_len += dur + (PACK_GAP_S if len(cur) > 1 else 0.0)
        if cur:
            packs.append(cur)
        gap = np.zeros(int(PACK_GAP_S * SR), dtype=np.float32)
        for pack in packs:
            if len(pack) == 1:
                result[pack[0][0]] = self.transcribe_array(pack[0][1])
                continue
            spans, parts, t = [], [], 0.0
            for k, (wav, y) in enumerate(pack):
                if k:
                    parts.append(gap)
                    t += PACK_GAP_S
                spans.append((wav, t, t + len(y) / SR))
                parts.append(y)
                t += len(y) / SR
            texts = {wav: [] for wav, _, _ in spans}
            for start, end, text in self.transcribe_array(np.concatenate(parts), timestamps=True):
                end = end if end is not None else t
                mid = (start + end) / 2
                owner = min(spans, key=lambda sp: 0 if sp[1] <= mid <= sp[2] else min(abs(mid - sp[1]), abs(mid - sp[2])))
                texts[owner[0]].append(text.strip())
            for wav, words in texts.items():
                result[wav] = ' '.join(words).strip()
        return result

    def close(self)->None:
        self.model = None
        import gc
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def load_16k(wav_path:str):
    import librosa
    import numpy as np
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    return y.astype(np.float32)


def models_available(models_dir:str)->tuple[str|None, str|None]:
    small = os.path.join(models_dir, 'whisper-small')
    turbo = os.path.join(models_dir, 'whisper-large-v3-turbo')
    ok = lambda d: os.path.exists(os.path.join(d, 'model.safetensors')) and os.path.exists(os.path.join(d, 'config.json'))
    return (small if ok(small) else None), (turbo if ok(turbo) else None)


# ---------------------------------------------------------------- babbling tails

TRIM_MIN_GAP_S = 0.02
TRIM_GAP_DB = -25
TRIM_MAX_CUTS = 5


def has_extra_tail(ref:str, hyp:str)->bool:
    """The transcription is the text followed by words that are not in it: "Il y a un
    pirate dans la cave d'Helto" for "Il y a un pirate dans la cave"."""
    r = norm_text(ref).split()
    h = norm_text(hyp).split()
    if len(h) <= len(r) or not r:
        return False
    # the last words of the text are found before the end of the transcription
    tail = r[-2:] if len(r) >= 2 else r
    for k in range(len(h) - len(tail) - 1, -1, -1):
        if h[k:k + len(tail)] == tail or (len(tail) == 2 and h[k] == tail[-1]):
            return k + len(tail) < len(h)
    return False


def trim_tail(asr:'Whisper', take:dict, lang_iso1:str)->bool:
    """XTTS often goes on after a short sentence (a made-up word or two). The take is cut
    at each inner pause, from the last one, every cut is transcribed and the best one is
    kept when it is clearly better; the original end silence is put back after the cut.
    Updates take['wav'] in place, and its 'asr' / 'cer'. Returns True when cut."""
    import numpy as np
    import soundfile as sf
    try:
        y, sr = sf.read(take['wav'], dtype='float32')
        mono = y if y.ndim == 1 else y.mean(axis=1)
        frame = int(0.01 * sr)
        n = len(mono) // frame
        if n < 20:
            return False
        rms = np.sqrt(np.mean(mono[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
        voiced = 20 * np.log10(rms / (rms.max() + 1e-12)) > TRIM_GAP_DB
        idx = np.where(voiced)[0]
        if len(idx) == 0:
            return False
        last = idx[-1]
        end_silence = len(mono) - (last + 1) * frame
        # ends of voiced runs followed by a pause: candidate cut points, latest first
        cuts = []
        gap = 0
        for i in range(last, -1, -1):
            if voiced[i]:
                if gap * 0.01 >= TRIM_MIN_GAP_S:
                    cuts.append(i + 1)
                gap = 0
            else:
                gap += 1
        cuts = [c for c in cuts if c * 0.01 >= 0.4][:TRIM_MAX_CUTS]
        best = (take['cer'], None, take['asr'])
        fade = int(0.025 * sr)
        tmp = take['wav'][:-5] + '.trim.flac' if take['wav'].endswith('.flac') else take['wav'] + '.trim.flac'
        for c in cuts:
            cut = y[:c * frame].copy()
            if len(cut) > fade:
                ramp = np.linspace(1.0, 0.0, fade, dtype='float32')
                cut[-fade:] = (cut[-fade:].T * ramp).T
            pad = np.zeros((end_silence,) + y.shape[1:], dtype='float32')
            audio = np.concatenate([cut, pad])
            sf.write(tmp, audio, sr, subtype=sf.info(take['wav']).subtype)
            hyp = asr.transcribe(tmp)
            score = round(cer(take['text'], hyp, lang_iso1), 4)
            if score < best[0] - 0.02:
                best = (score, audio, hyp)
        if os.path.exists(tmp):
            os.remove(tmp)
        if best[1] is None:
            return False
        sf.write(take['wav'], best[1], sr, subtype=sf.info(take['wav']).subtype)
        take['cer'], take['asr'] = best[0], best[2]
        take['trimmed'] = True
        return True
    except Exception as e:
        print(f'[verif] coupe de queue impossible ({take.get("wav")}) : {e}')
        return False


# ---------------------------------------------------------------- shared files

def _read_jsonl(path:str)->list[dict]:
    items = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except ValueError:
                        pass
    return items


def _append_jsonl(path:str, item:dict)->None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------- watch mode (CPU process)

def _parent_gone(parent_pid:int|None)->bool:
    # the conversion died (crash, GPU reset): the watcher must not stay behind as an orphan
    if not parent_pid:
        return False
    try:
        import psutil
        return not psutil.pid_exists(parent_pid)
    except Exception:
        return False


def watch(queue_path:str, results_path:str, stop_path:str, model_dir:str, lang_iso1:str, parent_pid:str|None=None)->None:
    parent_pid = int(parent_pid) if parent_pid else None
    asr = Whisper(model_dir, 'cpu', lang_iso1, threads=WATCH_THREADS)
    print(f'[verif] Whisper small pret (CPU), surveillance de {queue_path}', flush=True)
    while True:
        done = {r['wav'] for r in _read_jsonl(results_path)}
        chapters = _read_jsonl(queue_path)
        todo_chapters = [[c for c in ch['chunks'] if c['wav'] not in done and os.path.exists(c['wav'])] for ch in chapters]
        todo_chapters = [t for t in todo_chapters if t]
        if _parent_gone(parent_pid):
            print('[verif] conversion arretee, fin du surveillant', flush=True)
            return
        if not todo_chapters:
            if os.path.exists(stop_path):
                break
            time.sleep(5)
            continue
        for todo in todo_chapters:
            # one chapter at a time, by groups of ~8 chunks so a stop request is honoured quickly
            for i in range(0, len(todo), 8):
                if os.path.exists(stop_path) or _parent_gone(parent_pid):
                    return
                group = todo[i:i + 8]
                hyps = asr.transcribe_packed(group)
                for c in group:
                    hyp = hyps.get(c['wav'], '')
                    score = cer(c['text'], hyp, lang_iso1)
                    _append_jsonl(results_path, {'wav': c['wav'], 'model': 'small', 'cer': round(score, 4), 'asr': hyp})
                    if score > WATCH_FLAG_CER:
                        print(f"[verif] suspect (CER {score:.2f}) : {c['text'][:70]} => {hyp[:70]}", flush=True)


# ---------------------------------------------------------------- orchestration (in the conversion process)

class AsrVerifier:

    REPAIR_TAKES = REPAIR_TAKES
    SECOND_CHANCE_CER = SECOND_CHANCE_CER
    SECOND_CHANCE_TAKES = SECOND_CHANCE_TAKES

    def __init__(self, session:dict, models_dir:str):
        self.session = session
        self.lang = session.get('language_iso1') or 'fr'
        self.small, self.turbo = models_available(models_dir)
        work = session['process_dir']
        self.queue_path = os.path.join(work, 'asr_queue.jsonl')
        self.results_path = os.path.join(work, 'asr_results.jsonl')
        self.stop_path = os.path.join(work, 'asr_stop')
        self.repairs_path = os.path.join(work, 'asr_repairs.jsonl')
        self.proc = None

    @property
    def enabled(self)->bool:
        return self.turbo is not None

    def start_watcher(self)->None:
        if not self.small:
            print('[verif] Whisper small absent : verification seulement en fin de conversion')
            return
        if os.path.exists(self.stop_path):
            os.remove(self.stop_path)
        env = dict(os.environ, OMP_NUM_THREADS=str(WATCH_THREADS), MKL_NUM_THREADS=str(WATCH_THREADS), CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
        self.proc = subprocess.Popen(
            [sys.executable, '-u', '-m', 'lib.classes.asr_verifier', 'watch',
             self.queue_path, self.results_path, self.stop_path, self.small, self.lang, str(os.getpid())],
            env=env, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

    def enqueue_chapter(self, block_id:str, ch_num:int, chunks:list[dict])->None:
        """chunks: [{'j', 'wav', 'text'}] — text is what the TTS had to say."""
        _append_jsonl(self.queue_path, {'block_id': block_id, 'ch_num': ch_num, 'chunks': chunks})

    def stop_watcher(self)->None:
        open(self.stop_path, 'w').close()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def _latest_results(self)->dict:
        res = {}
        for r in _read_jsonl(self.results_path):
            prev = res.get(r['wav'])
            # precision order: turbo alone > turbo packed > small packed
            rank = {'small': 0, 'turbo-packed': 1, 'turbo': 2}
            if prev is None or rank.get(r['model'], 0) >= rank.get(prev['model'], 0):
                res[r['wav']] = r
        return res

    def verify_on_gpu(self, progress=None, block_ids:set|None=None)->list[dict]:
        """Checks the chunks not seen yet and confirms the flagged ones. Returns confirmed bad chunks.
        block_ids limits the check to the chapters of one part of the book."""
        chunks = {c['wav']: dict(c, block_id=ch['block_id'], ch_num=ch['ch_num'])
                  for ch in _read_jsonl(self.queue_path) if block_ids is None or ch['block_id'] in block_ids
                  for c in ch['chunks'] if os.path.exists(c['wav'])}
        results = self._latest_results()
        unseen = [c for w, c in chunks.items() if w not in results]
        print(f'[verif] {len(chunks)} troncons, {len(results)} deja verifies par le surveillant, {len(unseen)} restants')
        asr = None
        if unseen:
            # fast pass: packed windows
            asr = Whisper(self.turbo, 'cuda', self.lang)
            t0 = time.time()
            for i in range(0, len(unseen), 8):
                group = unseen[i:i + 8]
                hyps = asr.transcribe_packed(group)
                for c in group:
                    hyp = hyps.get(c['wav'], '')
                    _append_jsonl(self.results_path, {'wav': c['wav'], 'model': 'turbo-packed', 'cer': round(cer(c['text'], hyp, self.lang), 4), 'asr': hyp})
                done = min(i + 8, len(unseen))
                eta = (time.time() - t0) / done * (len(unseen) - done)
                print(f'[verif] passage rapide {done}/{len(unseen)} reste ~{eta / 60:.1f} min', flush=True)
                if progress:
                    progress(done / len(unseen) / 2)
        # precise pass: every suspect is transcribed alone by turbo
        results = self._latest_results()
        suspects = [c for w, c in chunks.items() if w in results and results[w]['model'] != 'turbo' and results[w]['cer'] > CONFIRM_CER]
        print(f'[verif] {len(suspects)} troncon(s) suspect(s) a confirmer')
        if suspects:
            asr = asr or Whisper(self.turbo, 'cuda', self.lang)
            for i, c in enumerate(suspects, 1):
                hyp = asr.transcribe(c['wav'])
                score = cer(c['text'], hyp, self.lang)
                _append_jsonl(self.results_path, {'wav': c['wav'], 'model': 'turbo', 'cer': round(score, 4), 'asr': hyp})
                print(f'[verif] confirmation {i}/{len(suspects)} CER {score:.3f}' + ('  <-- a reparer' if score > CONFIRM_CER else '  (faux positif)'), flush=True)
        if asr:
            asr.close()
        results = self._latest_results()
        bad = []
        for w, c in chunks.items():
            r = results.get(w)
            if r and r['model'] == 'turbo' and r['cer'] > CONFIRM_CER:
                bad.append(dict(c, cer=r['cer'], asr=r['asr']))
        return bad

    def score_takes(self, takes:list[dict], trim:bool=True)->None:
        """takes: [{'wav', 'text'}] -> adds 'cer' and 'asr' to each (GPU turbo).
        With trim, a take that says the text and then babbles is cut (see trim_tail)."""
        if not takes:
            return
        asr = Whisper(self.turbo, 'cuda', self.lang)
        for i, t in enumerate(takes, 1):
            t['asr'] = asr.transcribe(t['wav'])
            t['cer'] = round(cer(t['text'], t['asr'], self.lang), 4)
            note = ''
            if trim and t['cer'] > CONFIRM_CER and has_extra_tail(t['text'], t['asr']):
                before = t['cer']
                if trim_tail(asr, t, self.lang):
                    note = f' (queue coupee, CER {before:.3f} -> {t["cer"]:.3f})'
            print(f"[verif] essai {i}/{len(takes)} CER {t['cer']:.3f}{note}", flush=True)
        asr.close()

    def record_result(self, wav:str, score:float, hyp:str)->None:
        _append_jsonl(self.results_path, {'wav': wav, 'model': 'turbo', 'cer': round(score, 4), 'asr': hyp})

    def record_repair(self, item:dict)->None:
        _append_jsonl(self.repairs_path, item)

    def repairs(self)->list[dict]:
        return _read_jsonl(self.repairs_path)


if __name__ == '__main__':
    if len(sys.argv) >= 7 and sys.argv[1] == 'watch':
        watch(*sys.argv[2:8])
    else:
        print(__doc__)
