"""
Rebuilds the chapter files of a session from its existing sentence files, without any
synthesis: chapter silences (lead-in/tail), progressive "(en cours)" export. A resume of the
session then produces the final audiobook from these chapters.

  python tools/rebuild_chapters.py <session_id>
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def log(msg:str)->None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main()->None:
    from lib.conf import tmp_dir, default_audio_proc_format
    from lib.core import chapter_silence_files, chapter_title_of
    from lib.classes.progressive_export import ProgressiveExporter
    session_id = sys.argv[1]
    proc = glob.glob(os.path.join(tmp_dir, f'proc-{session_id}', '*', ''))[0]
    saved = glob.glob(os.path.join(proc, '__saved_*.json'))[0]
    data = json.load(open(saved, encoding='utf-8'))
    blocks = [b for b in data['blocks'] if b.get('keep') and (b.get('text') or '').strip()]
    chapters_dir = os.path.join(proc, 'chapters')
    sentences_dir = os.path.join(chapters_dir, 'sentences')
    # repairs already attempted must not be redone by the resume
    repairs = os.path.join(proc, 'asr_repairs.jsonl')
    results = os.path.join(proc, 'asr_results.jsonl')
    if os.path.exists(repairs):
        done = []
        for line in open(repairs, encoding='utf-8'):
            r = json.loads(line)
            if 'wav' not in r:
                r['wav'] = os.path.join(sentences_dir, blocks[r['ch_num'] - 1]['id'], f"{r['j']}.{default_audio_proc_format}")
            done.append(r)
        with open(repairs, 'w', encoding='utf-8') as f:
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(results, 'a', encoding='utf-8') as f:
            for r in done:
                if r.get('fixed') and r.get('after_cer') is not None:
                    f.write(json.dumps({'wav': r['wav'], 'model': 'turbo', 'cer': r['after_cer'], 'asr': r['after_asr']}, ensure_ascii=False) + '\n')
        log(f'{len(done)} reparation(s) Whisper deja faites, conservees')
    session_like = {'audiobooks_dir': os.path.join(ROOT, 'audiobooks', 'Piranese'), 'final_name': None, 'metadata': data.get('metadata') or {}, 'cover': None}
    out_dir = None
    for f in glob.glob(os.path.join(ROOT, 'audiobooks', '*', '*(en cours)')):
        if os.path.exists(os.path.join(f, '_chapitres.json')):
            out_dir = f
    if out_dir:
        session_like['audiobooks_dir'] = os.path.dirname(out_dir)
        session_like['final_name'] = os.path.basename(out_dir).replace(' (en cours)', '') + '.m4b'
        session_like['metadata'] = {'title': 'Piranèse', 'creator': 'Susanna Clarke'}
    exporter = ProgressiveExporter(session_like) if out_dir else None
    t0 = time.time()
    for n, b in enumerate(blocks, start=1):
        files = [os.path.join(sentences_dir, b['id'], f'{j}.{default_audio_proc_format}') for j, s in enumerate(b['sentences']) if any(c.isalnum() for c in s)]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            log(f'chapitre {n} : {len(missing)} troncon(s) manquant(s), laisse tel quel')
            continue
        lead_in, tail = chapter_silence_files(proc, files[0])
        lst = os.path.join(proc, 'concat_rebuild.txt')
        with open(lst, 'w', encoding='utf-8') as f:
            for p in [lead_in] + files + [tail]:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")
        out = os.path.join(chapters_dir, f"{b['id']}.{default_audio_proc_format}")
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c:a', default_audio_proc_format, out], check=True)
        if exporter:
            exporter.add_chapter(n, out, chapter_title_of(b['sentences'], n))
        eta = (time.time() - t0) / n * (len(blocks) - n)
        print(f'  [{n}/{len(blocks)}] chapitre reconstruit, reste ~{eta / 60:.1f} min', flush=True)
    log('Chapitres reconstruits.')


if __name__ == '__main__':
    main()
