"""
Progressive export: every finished chapter becomes listenable right away, long before the
whole book is converted.

In "<audiobooks_dir>/<book> (en cours)/":
- "NN - <chapter title>.m4a"  one file per finished chapter (final mastering, tagged)
- "<book> (en cours).m3u8"    playlist of the chapters ready so far
- "<book> (en cours).m4b"     the chapters ready so far as one audiobook with chapter marks,
                              rebuilt by stream copy (no re-encoding) after each chapter

Nothing here may stop a conversion: every error is printed and swallowed.
"""

import json
import os
import shutil
import subprocess

from pathlib import Path

# same mastering as the final export (core.combine_audio_chapters._export_audio)
MASTERING = 'dynaudnorm=f=150:g=15,afftdn=nf=-70'
FORBIDDEN = '<>:"/\\|?*'


def _safe(name:str, max_len:int=90)->str:
    name = ''.join('_' if c in FORBIDDEN or ord(c) < 32 else c for c in name).strip(' .')
    return name[:max_len].rstrip(' .') or 'chapitre'


def _stream_duration(path:str)->float:
    """Duration of the audio stream as ffprobe reports it, i.e. without the AAC priming
    frame. Chained by stream copy, a chapter starts exactly there; a decoded length is
    1024 samples longer and makes every later chapter mark drift (~43 ms per chapter)."""
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], text=True).strip()
        return float(out)
    except Exception:
        from lib.classes.tts_engines.common.audio import get_audio_duration
        return get_audio_duration(path)


def _ffmeta_escape(s:str)->str:
    for ch in ('\\', '=', ';', '#', '\n'):
        s = s.replace(ch, '\\' + ch)
    return s


class ProgressiveExporter:

    def __init__(self, session:dict):
        self.ffmpeg = shutil.which('ffmpeg')
        self.metadata = dict(session.get('metadata') or {})
        final_name = session.get('final_name') or 'audiobook'
        self.stem = Path(final_name).stem
        self.book_title = self.metadata.get('title') or self.stem
        self.out_dir = os.path.join(session['audiobooks_dir'], f'{self.stem} (en cours)')
        self.cover = session.get('cover')
        self.chapters:dict[int, tuple[str, str]] = {}
        os.makedirs(self.out_dir, exist_ok=True)
        self.index = os.path.join(self.out_dir, '_chapitres.json')
        # chapters exported by a previous (interrupted) run are kept
        try:
            with open(self.index, encoding='utf-8') as f:
                for num, (path, title) in json.load(f).items():
                    if os.path.exists(path):
                        self.chapters[int(num)] = (path, title)
        except (OSError, ValueError):
            pass

    def has(self, ch_num:int)->bool:
        return ch_num in self.chapters and os.path.exists(self.chapters[ch_num][0])

    def add_chapter(self, ch_num:int, chapter_file:str, title:str)->None:
        try:
            if not self.ffmpeg or not os.path.exists(chapter_file):
                return
            for old_num, (old_path, _) in list(self.chapters.items()):
                if old_num == ch_num and os.path.exists(old_path):
                    os.remove(old_path)
            target = os.path.join(self.out_dir, f'{ch_num:02d} - {_safe(title)}.m4a')
            tmp = target + '.tmp.m4a'
            cmd = [
                self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', chapter_file,
                '-af', MASTERING, '-ac', '1', '-c:a', 'aac', '-b:a', '128k',
                '-metadata', f'title={title}', '-metadata', f'track={ch_num}',
                '-metadata', f'album={self.book_title}',
                '-metadata', f"artist={self.metadata.get('creator') or ''}",
                '-movflags', '+faststart', tmp
            ]
            subprocess.run(cmd, check=True)
            os.replace(tmp, target)
            self.chapters[ch_num] = (target, title)
            with open(self.index, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in self.chapters.items()}, f, ensure_ascii=False, indent=1)
            self._write_playlist()
            self._rebuild_m4b()
            print(f'[en cours] chapitre {ch_num} pret a l\'ecoute : {target}')
        except Exception as e:
            print(f'[en cours] export du chapitre {ch_num} impossible : {e}')

    def _ready(self)->list[tuple[int, str, str]]:
        # only the uninterrupted run of chapters from the first one, so the partial
        # audiobook never has holes
        ready = []
        expected = min(self.chapters) if self.chapters else 1
        for num in sorted(self.chapters):
            if num != expected:
                break
            path, title = self.chapters[num]
            if os.path.exists(path):
                ready.append((num, path, title))
            expected += 1
        return ready

    def _write_playlist(self)->None:
        lines = ['#EXTM3U']
        for num, path, title in sorted((n, p, t) for n, (p, t) in self.chapters.items()):
            lines.append(f'#EXTINF:-1,{num:02d} - {title}')
            lines.append(os.path.basename(path))
        with open(os.path.join(self.out_dir, f'{self.stem} (en cours).m3u8'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def _rebuild_m4b(self)->None:
        ready = self._ready()
        if not ready:
            return
        concat = os.path.join(self.out_dir, '_concat.txt')
        meta = os.path.join(self.out_dir, '_chapters.txt')
        with open(concat, 'w', encoding='utf-8') as f:
            for _, path, _ in ready:
                f.write(f"file '{path.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")
        start = 0
        with open(meta, 'w', encoding='utf-8') as f:
            f.write(';FFMETADATA1\n')
            f.write(f'title={_ffmeta_escape(self.book_title)} (en cours, {len(ready)} chapitres)\n')
            if self.metadata.get('creator'):
                f.write(f"artist={_ffmeta_escape(self.metadata['creator'])}\n")
            for _, path, title in ready:
                dur = round(_stream_duration(path) * 1000)
                f.write(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={start + dur}\ntitle={_ffmeta_escape(title)}\n')
                start += dur
        target = os.path.join(self.out_dir, f'{self.stem} (en cours).m4b')
        tmp = target + '.tmp.m4b'
        cmd = [
            self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'concat', '-safe', '0', '-i', concat, '-f', 'ffmetadata', '-i', meta,
            '-map', '0:a', '-map_metadata', '1', '-map_chapters', '1', '-c', 'copy',
            '-movflags', '+faststart', '-f', 'mp4', tmp
        ]
        subprocess.run(cmd, check=True)
        self._add_cover(tmp)
        try:
            os.replace(tmp, target)
        except PermissionError:
            # the partial book is open in a player: keep it, the next chapter will retry
            os.remove(tmp)
            print('[en cours] livre partiel ouvert dans un lecteur, mise a jour au prochain chapitre')
        for f in (concat, meta):
            if os.path.exists(f):
                os.remove(f)

    def export_range(self, first:int, last:int, target:str, title:str)->str|None:
        """One chaptered m4b of chapters first..last (stream copy of the chapter files)."""
        try:
            chapters = [(n, *self.chapters[n]) for n in range(first, last + 1) if self.has(n)]
            missing = [n for n in range(first, last + 1) if not self.has(n)]
            if missing or not chapters:
                print(f'[partie] chapitres absents de l\'export : {missing}')
                return None
            os.makedirs(os.path.dirname(target), exist_ok=True)
            concat = target + '.concat.txt'
            meta = target + '.meta.txt'
            with open(concat, 'w', encoding='utf-8') as f:
                for _, path, _ in chapters:
                    f.write(f"file '{path.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")
            start = 0
            with open(meta, 'w', encoding='utf-8') as f:
                f.write(';FFMETADATA1\n')
                f.write(f'title={_ffmeta_escape(title)}\n')
                f.write(f'album={_ffmeta_escape(self.book_title)}\n')
                f.write('genre=Audiobook\n')
                if self.metadata.get('creator'):
                    f.write(f"artist={_ffmeta_escape(self.metadata['creator'])}\n")
                for _, path, ch_title in chapters:
                    dur = round(_stream_duration(path) * 1000)
                    f.write(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={start + dur}\ntitle={_ffmeta_escape(ch_title)}\n')
                    start += dur
            tmp = target + '.tmp.m4b'
            subprocess.run([
                self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'concat', '-safe', '0', '-i', concat, '-f', 'ffmetadata', '-i', meta,
                '-map', '0:a', '-map_metadata', '1', '-map_chapters', '1', '-c', 'copy',
                '-movflags', '+faststart', '-f', 'mp4', tmp
            ], check=True)
            self._add_cover(tmp)
            os.replace(tmp, target)
            for f in (concat, meta):
                if os.path.exists(f):
                    os.remove(f)
            print(f'[partie] {target} : {len(chapters)} chapitres, {start / 3600000:.2f} h')
            return target
        except Exception as e:
            print(f'[partie] export impossible : {e}')
            return None

    def _add_cover(self, path:str)->None:
        if not self.cover or not os.path.exists(self.cover):
            return
        try:
            from mutagen.mp4 import MP4, MP4Cover
            audio = MP4(path)
            with open(self.cover, 'rb') as img:
                fmt = MP4Cover.FORMAT_PNG if self.cover.lower().endswith('.png') else MP4Cover.FORMAT_JPEG
                audio['covr'] = [MP4Cover(img.read(), imageformat=fmt)]
            audio.save()
        except Exception as e:
            print(f'[en cours] couverture non ajoutee : {e}')
