"""
Final intonation (cadence) of a chunk: does the reading end on a falling pitch?

XTTS often ends a sentence flat or rising. For chapter titles and the last chunk of a
chapter, a non falling end sounds like the reading is not finished, so these chunks get
extra takes and the most falling one is kept (see core.convert_chapters2audio).

Measure, on the last 3 s of speech, F0 with pyin:
  fall_st = median F0 of the last 180 ms of voicing - median F0 of the 600..180 ms before
  end_st  = median F0 of the last 180 ms - median F0 of the analysed speech
(semitones). A chunk falls when fall_st <= -1.0 and end_st <= -1.5.
"""

import regex as re

FALL_ST = -1.0
END_ST = -1.5
MAX_TAKES = 6

_SML = re.compile(r'\[[a-z]+\]')
_CLOSERS = ' \t\n»"”’)]'


def plain(text:str)->str:
    return ' '.join(_SML.sub(' ', text).split())


def needs_falling_end(text:str)->bool:
    """Questions, exclamations and unfinished fragments (ending with a comma) keep their tune."""
    end = plain(text).rstrip(_CLOSERS)
    return bool(end) and end[-1] not in '?!,'


def is_heading_chunk(text:str)->bool:
    return '[pause]' in text


def cadence_targets(sentences:list[str])->dict[int, str]:
    """{chunk index: 'titre' | 'fin'} of the chunks whose reading must end falling."""
    # an SML-only chunk ("[break]") is a silence, not the last spoken line
    spoken = [j for j, s in enumerate(sentences) if any(c.isalnum() for c in plain(s))]
    targets = {}
    for j in spoken[:-1]:
        # titles: the leading chunks marked by the heading pause (the last chunk also ends
        # with a [pause], the one closing the chapter, it is never a title)
        if not is_heading_chunk(sentences[j]):
            break
        targets[j] = 'titre'
    if spoken and spoken[-1] not in targets:
        targets[spoken[-1]] = 'fin'
    return {j: kind for j, kind in targets.items() if needs_falling_end(sentences[j])}


def falls(c:dict|None)->bool:
    return bool(c) and c.get('fall_st') is not None and c['fall_st'] <= FALL_ST and c['end_st'] <= END_ST


def fall_score(c:dict|None)->float:
    """Lower is more falling; an unmeasurable take ranks last."""
    return 99.0 if not c or c.get('fall_st') is None else c['fall_st'] + c['end_st']


def rank(c:dict|None)->tuple:
    """Sort key, best first: takes that meet the falling criterion, then the deepest fall."""
    return (0 if falls(c) else 1, fall_score(c))


def _speech_bounds(y, sr:int, thr_db:float=-40.0)->tuple[int, int]:
    import numpy as np
    frame = int(0.01 * sr)
    n = len(y) // frame
    if n == 0:
        return 0, len(y)
    rms = np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.max(rms) + 1e-12))
    idx = np.where(db > thr_db)[0]
    return (idx[0] * frame, (idx[-1] + 1) * frame) if len(idx) else (0, len(y))


def measure(path:str)->dict|None:
    """{'fall_st', 'end_st', 'end_hz'} or None when there is too little voicing to tell."""
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        y, sr = sf.read(path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        a, b = _speech_bounds(y, sr)
        seg = y[max(a, b - 3 * sr):b]
        y16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
        hop = 160
        f0, voiced, _ = librosa.pyin(y16, fmin=60, fmax=300, sr=16000, frame_length=1024, hop_length=hop)
        t = np.arange(len(f0)) * hop / 16000
        ok = voiced & ~np.isnan(f0)
        if ok.sum() < 8:
            return None
        st = 12 * np.log2(f0[ok] / 100.0)
        tv = t[ok]
        t_end = tv[-1]
        last = st[tv >= t_end - 0.18]
        before = st[(tv < t_end - 0.18) & (tv >= t_end - 0.6)]
        if len(last) < 3 or len(before) < 3:
            return None
        return {
            'fall_st': round(float(np.median(last) - np.median(before)), 2),
            'end_st': round(float(np.median(last) - np.median(st)), 2),
            'end_hz': round(float(100 * 2 ** (np.median(last) / 12)), 1),
        }
    except Exception as e:
        print(f'[cadence] mesure impossible ({path}) : {e}')
        return None


CER_TIE = 0.005


def choose_take(cands:list[dict], max_cer:float)->dict|None:
    """Choice for a title / chapter end: among the takes Whisper accepts (CER <= max_cer),
    the most falling one; when none is accepted, the most falling of the takes at the lowest
    CER. Whisper often hears every take of a short title or a name the same wrong way
    ("Addy Domarus" -> "Adi Domarus", 17 % for all), the tie must then go to the cadence.
    cands: [{'cer', 'cadence'?}]."""
    scored = [c for c in cands if 'cer' in c]
    if not scored:
        return None
    limit = max(max_cer, min(c['cer'] for c in scored) + CER_TIE)
    ok = [c for c in scored if c['cer'] <= limit]
    return min(ok, key=lambda c: (rank(c.get('cadence')), c['cer']))
