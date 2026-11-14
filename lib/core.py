# NOTE!!NOTE!!!NOTE!!NOTE!!!NOTE!!NOTE!!!NOTE!!NOTE!!!
# THE WORD "CHAPTER" IN THE CODE DOES NOT MEAN
# IT'S THE REAL CHAPTER OF THE EBOOK SINCE NO STANDARDS
# ARE DEFINING A CHAPTER ON .EPUB FORMAT. THE WORD "BLOCK"
# IS USED TO PRINT IT OUT TO THE TERMINAL, AND "CHAPTER" TO THE CODE
# WHICH IS LESS GENERIC FOR THE DEVELOPERS

import argparse, asyncio, csv, difflib, fnmatch, sqlite3, hashlib, io, json, math, os, pytesseract, gc
import random, shutil, subprocess, sys, tempfile, threading, time, uvicorn, copy, base64
import traceback, socket, unicodedata, urllib.request, uuid, zipfile, pymupdf, multiprocessing
import ebooklib, psutil, requests, stanza, importlib, queue, pykakasi
import regex as re, gradio as gr

from typing import Any, Generator, Dict
from PIL import Image, ImageSequence
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString, Tag
from collections import Counter
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from ebooklib import epub
from ebooklib.epub import EpubBook
from ebooklib.epub import EpubHtml
from glob import glob
from iso639 import Lang
from markdown import markdown
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager, Event
from multiprocessing.managers import DictProxy, ListProxy, SyncManager
from stanza.pipeline.core import Pipeline, DownloadMethod
from num2words2 import num2words
from pathlib import Path
from PIL import Image
from pydub import AudioSegment
from pydub.utils import mediainfo
from queue import Queue, Empty
from types import MappingProxyType
from langdetect import detect
from unidecode import unidecode
from phonemizer import phonemize
from pypinyin import pinyin, Style

from lib.classes.subprocess_pipe import SubprocessPipe
from lib.classes.vram_detector import VRAMDetector
from lib.classes.voice_extractor import VoiceExtractor
from lib.classes.non_text_filter import NonTextFilter
from lib.classes.epub_sections import split_spine, is_excluded as is_excluded_doc
from lib.classes.html_tts_prep import prepare_html_for_tts, speak_inline_ordinals
from lib.classes.natural_chunker import NaturalChunker
from lib.classes.progressive_export import ProgressiveExporter
from lib.classes.asr_verifier import AsrVerifier, CONFIRM_CER as ASR_CONFIRM_CER, same_transcription
from lib.classes.asr_verifier import _append_jsonl as asr_jsonl_append
from lib.classes import cadence
#from lib.classes.redirect_console import RedirectConsole
from lib.classes.argos_translator import ArgosTranslator
from lib.classes.tts_manager import TTSManager
from lib.classes.tts_engines.common.audio import get_audiolist_duration, get_audio_duration
from lib.classes.tts_engines.common.utils import build_vtt_file

from lib import *

#import logging
#logging.basicConfig(
#    level=logging.INFO, # DEBUG for more verbosity
#    format="%(asctime)s [%(levelname)s] %(message)s"
#)

context = None
context_tracker = None
active_sessions = None
progress_bar = None

status_tags = {
    "OVERRIDE": "override",
    "DELETION": "deletion",
    "READY": "ready",
    "EDIT": "edit",
    "SKIP": "skip",
    "SWITCH": "switch",
    "CONVERTING": "converting",
    "END": "end",
    "DISCONNECTED": "disconnected"
}

ebook_modes = {
    "SINGLE": "single",
    "DIRECTORY": "directory",
    "TEXT": "text"
}

save_session_keys_except = [
    'blocks_orig',
    'blocks_saved',
    'blocks_current'
]

file_prefixes = {
    "clone": "__",
    "saved": "__saved_",
    'current': "__current_"
}

########### Classes

class DependencyError(Exception):
    def __init__(self, message:str|None):
        super().__init__(message)
        print(message)
        # Automatically handle the exception when it's raised
        self.handle_exception()

    def handle_exception(self)->None:
        # Print the full traceback of the exception
        traceback.print_exc()      
        # Print the exception message
        error = f'Caught DependencyError: {self}'
        print(error)

class SessionTracker:
    def __init__(self):
        self.lock = threading.Lock()
        #self.blocks_autosave = AppAutosave()
        #self.blocks_autosave.start()

    def start_session(self, session_id:str)->bool:
        with self.lock:
            if session_id not in context.sessions:
                return False
            session = context.get_session(session_id)
            session['status'] = status_tags['READY']
            return True

    def end_session(self, session_id:str, socket_hash:str)->None:
        #self.blocks_autosave.unregister(session_id)
        active_sessions.discard(socket_hash)
        with self.lock:
            context.sessions.pop(session_id, None)

class SessionContext:
    def __init__(self):
        self.manager:Manager = Manager()
        self.sessions:DictProxy[str, DictProxy[str, Any]] = self.manager.dict()
        self.cancellation_events = {}

    def _recursive_proxy(self, data:Any, manager:SyncManager|None)->Any:
        if manager is None:
            manager = self.manager
        if isinstance(data, dict):
            proxy_dict = manager.dict()
            for key, value in data.items():
                proxy_dict[key] = self._recursive_proxy(value, manager)
            return proxy_dict
        elif isinstance(data, list):
            proxy_list = manager.list()
            for item in data:
                proxy_list.append(self._recursive_proxy(item, manager))
            return proxy_list
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            error = f'Unsupported data type: {type(data)}'
            print(error)
            return None

    def set_session(self, session_id:str)->Any:
        self.sessions[session_id] = self._recursive_proxy({
            ####### Global settings
            "id": session_id,
            "script_mode": NATIVE,
            "tab_id": None,
            "socket_hash": None,
            "session_dir": None,
            "is_gui_process": False,
            "free_vram_gb": 0,
            "status": None,
            "ticker": 0,
            "cancellation_requested": False,
            "ebook_mode": ebook_modes['SINGLE'],
            "blocks_preview": False,
            "device": default_device,
            "tts_engine": default_tts_engine,
            "fine_tuned": default_fine_tuned,
            "model_cache": None,
            "model_zs_cache": None,
            "stanza_cache": None,
            "system": None,
            "client": None,
            "language": default_language_code,
            "language_iso1": None,
            "translate_enabled": False,
            "translate": None,
            "translate_iso1": None,
            "voice": None,
            "voice_dir": None,
            "voice_map": {},
            "ebook_selected": None,
            "custom_model": None,
            "custom_model_dir": None,
            "output_dir": None,
            "output_format": default_output_format,
            "output_channel": default_output_channel,
            "output_split": default_output_split,
            "output_split_hours": default_output_split_hours,
            "abs_url": default_abs_url,
            "abs_api_token": default_abs_api_token,
            "abs_library": default_abs_library,
            ####### Xtts settings
            "xtts_temperature": default_engine_settings[TTS_ENGINES['XTTS']]['temperature'],
            #"xtts_codec_temperature": default_engine_settings[TTS_ENGINES['XTTS']]['codec_temperature'],
            "xtts_length_penalty": default_engine_settings[TTS_ENGINES['XTTS']]['length_penalty'],
            "xtts_num_beams": default_engine_settings[TTS_ENGINES['XTTS']]['num_beams'],
            "xtts_repetition_penalty": default_engine_settings[TTS_ENGINES['XTTS']]['repetition_penalty'],
            #"xtts_cvvp_weight": default_engine_settings[TTS_ENGINES['XTTS']]['cvvp_weight'],
            "xtts_top_k": default_engine_settings[TTS_ENGINES['XTTS']]['top_k'],
            "xtts_top_p": default_engine_settings[TTS_ENGINES['XTTS']]['top_p'],
            "xtts_speed": default_engine_settings[TTS_ENGINES['XTTS']]['speed'],
            #"xtts_gpt_cond_len": default_engine_settings[TTS_ENGINES['XTTS']]['gpt_cond_len'],
            #"xtts_gpt_batch_size": default_engine_settings[TTS_ENGINES['XTTS']]['gpt_batch_size'],
            "xtts_enable_text_splitting": default_engine_settings[TTS_ENGINES['XTTS']]['enable_text_splitting'],
            ####### Bark settings
            "bark_text_temp": default_engine_settings[TTS_ENGINES['BARK']]['text_temp'],
            "bark_waveform_temp": default_engine_settings[TTS_ENGINES['BARK']]['waveform_temp'],
            ####### Audiobook editor
            "audiobook": None,
            "audiobooks_dir": None,
            ####### Ebook conversion
            "ebook": None,
            "ebook_src": None,
            "ebook_list": None,
            "ebook_loaded": None,
            "ebook_textarea": None,
            "ebook_textarea_src": None,
            "audiobook_overridden": None,
            "process_dir": None,
            "chapters_dir": None,
            "sentences_dir": None,
            "epub_path": None,
            "final_name": None,
            "filename_noext": None,
            "cover": None,
            "blocks_orig": {},
            "blocks_saved": {},
            "blocks_current": {},
            "blocks_orig_json": None,
            "blocks_saved_json": None,
            "blocks_current_db": None,
            "duration": 0,
            "playback_time": 0,
            "playback_volume": 0,
            "metadata": {
                "title": None, 
                "creator": None,
                "contributor": None,
                "language": None,
                "identifier": None,
                "publisher": None,
                "date": None,
                "description": None,
                "subject": None,
                "rights": None,
                "format": None,
                "type": None,
                "coverage": None,
                "relation": None,
                "Source": None,
                "Modified": None,
            }
        }, manager=self.manager)
        return self.sessions[session_id]

    def get_session(self, session_id:str)->Any:
        if session_id in self.sessions:
            return self.sessions[session_id]
        return {}

    def find_id_by_hash(self, socket_hash:str)->str|None:
        for session_id, session in list(self.sessions.items()):
            if socket_hash in session:
                return session_id
        return None
        
class JSONDictProxyEncoder(json.JSONEncoder):
    def default(self, o:Any)->Any:
        if isinstance(o, DictProxy):
            return dict(o)
        elif isinstance(o, ListProxy):
            return list(o)
        return super().default(o)

"""
class AppAutosave:
    def __init__(self, interval:float=15.0):
        self._interval = interval
        self._sessions: set[str] = set()
        self._lock = threading.Lock()
        self._started = False

    def start(self)->None:
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._timer, daemon=True)
        t.start()

    def register(self, session_id:str)->None:
        with self._lock:
            self._sessions.add(session_id)

    def unregister(self, session_id:str)->None:
        with self._lock:
            self._sessions.discard(session_id)

    def _timer(self)->None:
        while True:
            time.sleep(self._interval)
            with self._lock:
                session_ids = set(self._sessions)
            for session_id in session_ids:
                try:
                    session = context.get_session(session_id)
                    if not session or not session.get('id', False):
                        with self._lock:
                            self._sessions.discard(session_id)
                        continue
                except Exception as e:
                    logger.error(f'AppAutosave._timer({session_id}): {e}!')
"""
        
############# End classes

def prepare_dirs(session_id:str)->bool:
    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            resume = False
            os.makedirs(os.path.join(models_dir,'tts'), exist_ok=True)
            os.makedirs(session['session_dir'], exist_ok=True)
            os.makedirs(session['process_dir'], exist_ok=True)
            os.makedirs(session['custom_model_dir'], exist_ok=True)
            os.makedirs(session['voice_dir'], exist_ok=True)
            os.makedirs(session['audiobooks_dir'], exist_ok=True)
            os.makedirs(session['chapters_dir'], exist_ok=True)
            os.makedirs(session['sentences_dir'], exist_ok=True)
            return True
    except Exception as e:
        DependencyError(e)
        return False

def check_programs(prog_name:str, command:str, options:str)->bool:
    try:
        subprocess.run(
            [command, options],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            encoding='utf-8'
        )
        return True
    except FileNotFoundError:
        e = f'''********** Error: {prog_name} is not installed! if your OS calibre package version 
        is not compatible you still can run ebook2audiobook.sh (linux/mac) or ebook2audiobook.cmd (windows) **********'''
        DependencyError(e)
    except subprocess.CalledProcessError:
        e = f'Error: There was an issue running {prog_name}.'
        DependencyError(e)
    return False

def analyze_uploaded_file(zip_path:str, required_files:list[str])->bool:
    try:
        if not os.path.exists(zip_path):
            error = f'The file does not exist: {os.path.basename(zip_path)}'
            print(error)
            return False
        files_in_zip = {}
        empty_files = set()
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for file_info in zf.infolist():
                file_name = file_info.filename
                if file_info.is_dir():
                    continue
                base_name = os.path.basename(file_name)
                files_in_zip[base_name] = file_info.file_size
                if file_info.file_size == 0:
                    empty_files.add(base_name)
        required_files = [file for file in required_files]
        missing_files = [f for f in required_files if f not in files_in_zip]
        required_empty_files = [f for f in required_files if f in empty_files]
        if missing_files:
            msg = f'Missing required files: {missing_files}'
            print(msg)
        if required_empty_files:
            msg = f'Required files with 0 KB: {required_empty_files}'
            print(msg)
        return not missing_files and not required_empty_files
    except zipfile.BadZipFile:
        error = 'The file is not a valid ZIP archive.'
        print(error)
        return False
    except Exception as e:
        error = f'An error occurred: {e}'
        print(error)
        return False

def extract_custom_model(session_id)->str|None:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        file_src = session['custom_model']
        required_files = default_engine_settings[session['tts_engine']]['files']
        model_path = None
        model_name = re.sub('.zip', '', os.path.basename(file_src), flags=re.IGNORECASE)
        model_name = get_sanitized(model_name)
        try:
            with zipfile.ZipFile(file_src, 'r') as zip_ref:
                files = zip_ref.namelist()
                files_length = len(files)
                tts_dir = session['tts_engine']
                model_path = os.path.join(session['custom_model_dir'], tts_dir, model_name)
                os.makedirs(model_path, exist_ok=True)
                msg = f'Extracting files to {model_path}…'
                with tqdm(total=files_length, unit='files') as t:
                    for f in files:
                        base_f = os.path.basename(f)
                        if base_f in required_files:
                            out_path = os.path.join(model_path, base_f)
                            with zip_ref.open(f) as src, open(out_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                        t.update(1)
                        if session['is_gui_process']:
                            progress_bar((t.n + 1) / files_length, desc=msg)
            if model_path is not None:
                msg = f'Normalizing ref.wav…'
                print(msg)
                voice_ref = os.path.join(model_path, 'ref.wav')
                voice_name = model_name
                final_voice_file = os.path.join(model_path, f'{voice_name}.wav')
                extractor = VoiceExtractor(session, voice_ref, voice_name, final_voice_file)
                status, msg = extractor.extract_voice()
                if status:
                    session['voice'] = final_voice_file
                    if os.path.exists(file_src):
                        os.remove(file_src)
                    if os.path.exists(voice_ref):
                        os.remove(voice_ref)
                    return model_path
                else:
                    error = f'extract_custom_model() VoiceExtractor.extract_voice() error! {msg}'
                    print(error)
            else:
                error = f'An error occurred     when unzip {file_src}'
                print(error)
        except asyncio.exceptions.CancelledError as e:
            DependencyError(e)
            error = f'extract_custom_model asyncio.exceptions.CancelledError: {e}'
            print(error)
        except Exception as e:
            DependencyError(e)
            error = f'extract_custom_model Exception: {e}'
            print(error)
        if session['is_gui_process']:
            if os.path.exists(file_src):
                os.remove(file_src)
        session['custom_model'] = None
    return None
        
def hash_proxy_dict(proxy_dict:Any)->str:
    try:
        data = {k: v for k, v in dict(proxy_dict).items() if k not in save_session_keys_except}
    except Exception:
        data = {}
    data_str = json.dumps(data, default=str, sort_keys=True)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()

def compare_checksums(session_id:str)->tuple[bool, str|None]:
    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            hash_algorithm:str = 'sha256'
            checksum_path = os.path.join(session['process_dir'], 'checksum')
            hash_func = hashlib.new(hash_algorithm)
            with open(session['ebook'], 'rb') as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)
            new_checksum = hash_func.hexdigest()
            if not os.path.exists(checksum_path):
                with open(checksum_path, 'w', encoding='utf-8') as f:
                    f.write(new_checksum)
                return False, None
            else:
                with open(checksum_path, 'r', encoding='utf-8') as f:
                    saved_checksum = f.read().strip()
                if saved_checksum == new_checksum:
                    return True, None
                else:
                    with open(checksum_path, 'w', encoding='utf-8') as f:
                        f.write(new_checksum)
                        return False, None
        error = f'compare_checksums() error: session does not exist'
        return False, error
    except Exception as e:
        error = f'compare_checksums() error: {e}'
        return False, error

def compare_dict_keys(d1, d2):
    if not isinstance(d1, Mapping) or not isinstance(d2, Mapping):
        return d1 == d2
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    missing_in_d2 = d1_keys - d2_keys
    missing_in_d1 = d2_keys - d1_keys
    if missing_in_d2 or missing_in_d1:
        return {
            "missing_in_d2": missing_in_d2,
            "missing_in_d1": missing_in_d1,
        }
    for key in d1_keys.intersection(d2_keys):
        nested_result = compare_keys(d1[key], d2[key])
        if nested_result:
            return {key: nested_result}
    return None

def ocr2xhtml(img: Image.Image, lang:str)->tuple[str|bool, str|None]:
    try:
        debug = True
        try:
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME)
            # Handle silent OCR failures (empty or None result)
            if data is None or data.empty:
                error = f'Tesseract returned empty OCR data for language "{lang}".'
                return False, error
        except (pytesseract.TesseractError, Exception) as e:
            print(f'The OCR {lang} trained model must be downloaded.')
            try:
                tessdata_dir = os.environ['TESSDATA_PREFIX']
                os.makedirs(tessdata_dir, exist_ok=True)
                url = f'https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata'
                dest_path = os.path.join(tessdata_dir, f'{lang}.traineddata')
                msg = f'Downloading {lang}.traineddata into {tessdata_dir}…'
                print(msg)
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(response.content)
                    msg = f'Downloaded and installed {lang}.traineddata successfully.'
                    print(msg)
                    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME)
                    if data is None or data.empty:
                        error = f'Tesseract returned empty OCR data even after downloading {lang}.traineddata.'
                        return False, error
                else:
                    error = f'Failed to download traineddata for {lang} (HTTP {response.status_code})'
                    return False, error
            except Exception as e:
                error = f'Automatic download failed: {e}'
                return False, error
        data = data.dropna(subset=['text'])
        lines = []
        last_block = None
        for _, row in data.iterrows():
            text = row['text'].strip()
            if not text:
                continue
            block = row['block_num']
            if last_block is not None and block != last_block:
                lines.append('')  # blank line between blocks
            lines.append(text)
            last_block = block
        joined = '\n'.join(lines)
        raw_lines = [l.strip() for l in joined.split('\n')]
        # Normalize line breaks
        merged_lines = []
        buffer = ''
        for i, line in enumerate(raw_lines):
            if not line:
                if buffer:
                    merged_lines.append(buffer.strip())
                    buffer = ''
                continue
            if buffer and not buffer.endswith(('.', '?', '!', ':')) and not line[0].isupper():
                buffer += ' ' + line
            else:
                if buffer:
                    merged_lines.append(buffer.strip())
                buffer = line
        if buffer:
            merged_lines.append(buffer.strip())
        # Detect heading-like lines
        xhtml_parts = []
        debug_dump = []
        for i, p in enumerate(merged_lines):
            is_heading = False
            if p.isupper() and len(p.split()) <= 8:
                is_heading = True
            elif len(p.split()) <= 5 and p.istitle():
                is_heading = True
            elif (i == 0 or (i > 0 and merged_lines[i-1] == '')) and len(p.split()) <= 10:
                is_heading = True
            if is_heading:
                xhtml_parts.append(f'<h2>{p}</h2>')
                debug_dump.append(f'[H2] {p}')
            else:
                xhtml_parts.append(f'<p>{p}</p>')
                debug_dump.append(f'[P ] {p}')
        if debug:
            print('=== OCR DEBUG OUTPUT ===')
            for line in debug_dump:
                print(line)
            print('========================')
        return '\n'.join(xhtml_parts), None
    except Exception as e:
        DependencyError(e)
        error = f'ocr2xhtml error: {e}'
        return False, error

def create_db_blocks(db_path:str)->None:
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA foreign_keys=ON')
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS stamp (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                page INTEGER,
                block_resume INTEGER,
                sentence_resume INTEGER,
                voice TEXT,
                tts_engine TEXT,
                fine_tuned TEXT
            );
            CREATE TABLE IF NOT EXISTS blocks (
                id TEXT PRIMARY KEY,
                idx INTEGER NOT NULL,
                expand INTEGER NOT NULL,
                keep INTEGER NOT NULL,
                text TEXT NOT NULL,
                voice TEXT,
                tts_engine TEXT,
                fine_tuned TEXT
            );
            CREATE TABLE IF NOT EXISTS sentences (
                block_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (block_id, idx),
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_idx ON blocks(idx);
            INSERT OR IGNORE INTO stamp (id, page, block_resume, sentence_resume, voice, tts_engine, fine_tuned)
            VALUES (1, 0, 0, 0, NULL, NULL, NULL);
        ''')

def load_db_blocks(db_path:str)->dict:
    try:
        if not os.path.exists(db_path):
            return {}
        with sqlite3.connect(db_path) as conn:
            conn.execute('PRAGMA foreign_keys=ON')
            stamp_row = conn.execute(
                'SELECT page, block_resume, sentence_resume, voice, tts_engine, fine_tuned FROM stamp WHERE id=1'
            ).fetchone()
            if stamp_row is None:
                return {}
            page, block_resume, sentence_resume, voice, tts_engine, fine_tuned = stamp_row
            sentences_by_block = {}
            for block_id, text in conn.execute('SELECT block_id, text FROM sentences ORDER BY block_id, idx'):
                sentences_by_block.setdefault(block_id, []).append(text)
            blocks = []
            for row in conn.execute('SELECT id, expand, keep, text, voice, tts_engine, fine_tuned FROM blocks ORDER BY idx'):
                bid, expand, keep, text, b_voice, b_tts_engine, b_fine_tuned = row
                blocks.append({
                    'id': bid,
                    'expand': bool(expand),
                    'keep': bool(keep),
                    'text': text,
                    'voice': b_voice,
                    'tts_engine': b_tts_engine,
                    'fine_tuned': b_fine_tuned,
                    'sentences': sentences_by_block.get(bid, []),
                })
            return {
                'page': page,
                'block_resume': block_resume,
                'sentence_resume': sentence_resume,
                'voice': voice,
                'tts_engine': tts_engine,
                'fine_tuned': fine_tuned,
                'blocks': blocks,
            }
    except Exception as e:
        error = f'load_db_blocks() error: {e}'
        print(error)
        return {}

def voice_name_of(voice:str|None)->str|None:
    if not voice:
        return None
    name = os.path.splitext(os.path.basename(voice))[0].replace('&', 'And')
    return get_sanitized(name)

def read_stamp_voice(db_path:str)->tuple:
    try:
        if not os.path.exists(db_path):
            return False, None
        with sqlite3.connect(db_path) as conn:
            row = conn.execute('SELECT voice FROM stamp WHERE id=1').fetchone()
            if row is None:
                return False, None
            return True, row[0]
    except Exception as e:
        print(f'read_stamp_voice() error: {e}')
        return False, None

def count_blocks_global_voice(db_path:str)->tuple:
    try:
        if not os.path.exists(db_path):
            return False, 0, 0
        with sqlite3.connect(db_path) as conn:
            row = conn.execute('SELECT voice FROM stamp WHERE id=1').fetchone()
            if row is None:
                return False, 0, 0
            stamp_voice = row[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE keep=1 AND TRIM(COALESCE(text,'')) <> ''"
            ).fetchone()[0]
            following = conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE keep=1 AND TRIM(COALESCE(text,'')) <> '' AND voice IS ?",
                (stamp_voice,)
            ).fetchone()[0]
            return True, total, following
    except Exception as e:
        print(f'count_blocks_global_voice() error: {e}')
        return False, 0, 0

def build_voice_change_note(process_dir:str, current_voice:str|None, html:bool=True)->str|None:
    try:
        db_matches = glob(os.path.join(process_dir, f"{file_prefixes['current']}*.db"))
        if not db_matches:
            return None
        db_path = db_matches[0]
        has_prev, prev_voice = read_stamp_voice(db_path)
        if not has_prev or voice_name_of(prev_voice) == voice_name_of(current_voice):
            return None
        prev_label = voice_name_of(prev_voice) or 'default'
        curr_label = voice_name_of(current_voice) or 'default'
        b0, b1, br = ('<b>', '</b>', '<br/><br/>') if html else ('', '', '\n')
        note = (f'{br}NOTE: the previous global voice was {b0}{prev_label}{b1} but the current '
                f'global voice is {b0}{curr_label}{b1}.')
        ok, total, following = count_blocks_global_voice(db_path)
        if not ok or total == 0:
            return note + ' If you keep this current voice the whole ebook will be converted again.'
        own_voice = total - following
        if following == 0:
            note += (f' All {total} blocks use their own voice, so none of them will be '
                     f'reconverted because of this change.')
        elif own_voice == 0:
            note += f' If you keep this current voice all {total} blocks will be converted again.'
        else:
            note += (f' If you keep this current voice {b0}{following}{b1} of {total} blocks will be '
                     f'converted again; the other {b0}{own_voice}{b1} keep their own block voice '
                     f'and will not be reconverted.')
        return note
    except Exception as e:
        print(f'build_voice_change_note() error: {e}')
        return None

def save_db_stamp(session_id:str)->None:
    try:
        session = context.get_session(session_id)
        if not (session and session.get('id', False)):
            return
        data = session['blocks_current']
        if not data:
            return
        db_path = session['blocks_current_db']
        create_db_blocks(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                'UPDATE stamp SET page=?, block_resume=?, sentence_resume=?, voice=?, tts_engine=?, fine_tuned=? WHERE id=1',
                (
                    data.get('page'),
                    data.get('block_resume'),
                    data.get('sentence_resume'),
                    data.get('voice'),
                    data.get('tts_engine'),
                    data.get('fine_tuned'),
                )
            )
            conn.commit()
    except Exception as e:
        error = f'save_db_stamp() error: {e}'
        print(error)

def save_db_blocks(session_id:str)->None:
    try:
        session = context.get_session(session_id)
        if not (session and session.get('id', False)):
            return
        data = session['blocks_current']
        if not data:
            return
        db_path = session['blocks_current_db']
        create_db_blocks(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute(
                'UPDATE stamp SET page=?, block_resume=?, sentence_resume=?, voice=?, tts_engine=?, fine_tuned=? WHERE id=1',
                (
                    data.get('page'),
                    data.get('block_resume'),
                    data.get('sentence_resume'),
                    data.get('voice'),
                    data.get('tts_engine'),
                    data.get('fine_tuned'),
                )
            )
            new_blocks = data.get('blocks', [])
            new_ids = {b['id'] for b in new_blocks}
            existing_ids = {row[0] for row in conn.execute('SELECT id FROM blocks')}
            removed = existing_ids - new_ids
            if removed:
                conn.executemany('DELETE FROM blocks WHERE id=?', [(rid,) for rid in removed])
            for idx, block in enumerate(new_blocks):
                block_id = block['id']
                conn.execute(
                    'INSERT INTO blocks (id, idx, expand, keep, text, voice, tts_engine, fine_tuned) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?) '
                    'ON CONFLICT(id) DO UPDATE SET '
                    'idx=excluded.idx, expand=excluded.expand, keep=excluded.keep, text=excluded.text, '
                    'voice=excluded.voice, tts_engine=excluded.tts_engine, fine_tuned=excluded.fine_tuned',
                    (
                        block_id,
                        idx,
                        1 if block.get('expand') else 0,
                        1 if block.get('keep') else 0,
                        block.get('text', ''),
                        block.get('voice'),
                        block.get('tts_engine'),
                        block.get('fine_tuned'),
                    )
                )
                conn.execute('DELETE FROM sentences WHERE block_id=?', (block_id,))
                sentences = block.get('sentences', [])
                if sentences:
                    conn.executemany(
                        'INSERT INTO sentences (block_id, idx, text) VALUES (?, ?, ?)',
                        [(block_id, i, s) for i, s in enumerate(sentences)]
                    )
            conn.commit()
    except Exception as e:
        error = f'save_db_blocks() error: {e}'
        print(error)

def load_json_blocks(filepath:str)->dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        error = f"load_json_blocks() error: {e}"
        print(error)
        return {}

def save_json_blocks(session_id:str, key:str)->None:
    try:
        session = context.get_session(session_id)
        if (session and session.get('id', False)):
            json_data = session[key]
            json_path = session[f'{key}_json']
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'save_json_blocks() error: {e}')

def sync_globals_to_blocks(session_id:str)->None:
    try:
        session = context.get_session(session_id)
        if not (session and session.get('id', False)):
            return
        blocks_current = session.get('blocks_current') or {}
        anchor_voice = blocks_current.get('voice')
        current_voice = session.get('voice')
        if anchor_voice == current_voice:
            return
        changed = False
        for block in blocks_current.get('blocks', []):
            if block.get('voice') == anchor_voice:
                block['voice'] = current_voice
                changed = True
        blocks_current['voice'] = current_voice
        session['blocks_current'] = blocks_current
    except Exception as e:
        exception_alert(session_id, f'sync_globals_to_blocks(): {e}')

def normalize_epub_zip(session_id:str, file_input:str)->str|None:
    try:
        session = context.get_session(session_id)
        if not (session and session.get('id', False)):
            return None
        with zipfile.ZipFile(file_input, 'r') as zf:
            names = [n for n in zf.namelist() if n and not n.endswith('/')]
            epubs = [n for n in names if n.lower().endswith('.epub')]
            if len(epubs) > 1:
                msg = f'Unsupported ZIP ebook wrapper: expected one nested .epub file, found {len(epubs)}'
                print(msg)
                return None
            nested_epub = epubs[0] if epubs else None
            if nested_epub:
                # case 1 - a real .epub FILE sits inside the zip, just extract it verbatim
                target_name = f'{get_sanitized(Path(nested_epub).stem)}.epub'
                target_path = os.path.join(os.path.dirname(file_input), target_name)
                with open(target_path, 'wb') as out:
                    out.write(zf.read(nested_epub))
            else:
                # case 2/3 - epub contents are at root or under a single top dir
                root_mimetype = 'mimetype' in names
                prefix = ''
                if not root_mimetype:
                    dirs = {n.split('/', 1)[0] for n in names if '/' in n}
                    cands = [d for d in dirs if f'{d}/mimetype' in names and f'{d}/META-INF/container.xml' in names]
                    if len(cands) != 1:
                        msg = f'Unsupported ZIP ebook wrapper: expected one EPUB root, found {len(cands)}'
                        print(msg)
                        return None
                    prefix = cands[0]
                stem = Path(prefix).stem if prefix else Path(file_input).stem
                target_name = f'{get_sanitized(stem)}.epub'
                target_path = os.path.join(os.path.dirname(file_input), target_name)
                strip = f'{prefix}/' if prefix else ''
                members = [n for n in names if n.startswith(strip)]
                mimetype_name = f'{strip}mimetype'
                if mimetype_name not in members:
                    msg = 'Unsupported ZIP ebook wrapper: no mimetype entry'
                    print(msg)
                    return None
                with zipfile.ZipFile(target_path, 'w') as out:
                    mt = zipfile.ZipInfo('mimetype')
                    mt.compress_type = zipfile.ZIP_STORED
                    out.writestr(mt, zf.read(mimetype_name))
                    for name in members:
                        if name == mimetype_name:
                            continue
                        arcname = name[len(strip):]
                        if not arcname:
                            continue
                        out.writestr(arcname, zf.read(name), zipfile.ZIP_DEFLATED)
        session = context.get_session(session_id)
        session['ebook'] = target_path
        session['filename_noext'] = os.path.splitext(os.path.basename(target_path))[0]
        session['epub_path'] = os.path.join(session['process_dir'], f"__{session['filename_noext']}.epub")
        msg = f'Normalized EPUB package ZIP: {Path(file_input).name} -> {Path(target_path).name}'
        print(msg)
        return target_path
    except zipfile.BadZipFile:
        error = f'Unsupported ZIP ebook wrapper: bad ZIP file {file_input}'
        print(error)
    except Exception as e:
        error = f'normalize_epub_zip(): {e}'
        exception_alert(session_id, error)
    return None

def convert2epub(session_id:str)->bool:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        if session['cancellation_requested']:
            return False
        try:
            title = False
            author = False
            ebook_convert = shutil.which('ebook-convert')
            if not ebook_convert:
                error = 'ebook-convert utility is not installed or not found.'
                print(error)
                return False
            file_input = session['ebook']
            if os.path.getsize(file_input) == 0:
                error = f'Input file is empty: {file_input}'
                print(error)
                return False
            file_ext = os.path.splitext(file_input)[1].lower()
            if file_ext not in ebook_formats:
                error = f'Unsupported file format: {file_ext}'
                print(error)
                return False
            if file_ext == '.zip':
                file_input = normalize_epub_zip(session_id, file_input)
                if file_input is None:
                    return False
                file_ext = '.epub'
            if file_ext == '.txt':
                with open(file_input, 'r', encoding='utf-8') as f:
                    text = f.read()
                text = text.replace('\r\n', '\n')
                text = re.sub(r'\n{2,}', f".{TTS_SML['pause']['static']}", text)
                with open(file_input, 'w', encoding='utf-8') as f:
                    f.write(text)
            elif file_ext == '.pdf':
                msg = 'File input is a PDF. flatten it in XHTML…'
                print(msg)
                doc = pymupdf.open(file_input)
                file_meta = doc.metadata
                filename_noext = os.path.splitext(os.path.basename(session['ebook']))[0]
                title = file_meta.get('title') or filename_noext
                author = file_meta.get('author') or False
                xhtml_pages = []
                for i, page in enumerate(doc):
                    has_text = page.get_text('text').strip()
                    if has_text:
                        try:
                            xhtml_content = page.get_text('xhtml').strip()
                        except Exception as e:
                            print(f'Error extracting text from page {i+1}: {e}')
                            xhtml_content = ''
                        error = None
                    else:
                        xhtml_content = ''
                        error = None
                    if not xhtml_content:
                        msg = f'The page {i+1} seems to be image-based. Using OCR…'
                        show_alert(session_id, {"type": "warning", "msg": msg})
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes('png')))
                        xhtml_content, error = ocr2xhtml(img, session['language'])
                    if xhtml_content:
                        xhtml_pages.append(xhtml_content)
                    else:
                        show_alert(session_id, {"type": "warning", "msg": error})
                if xhtml_pages:
                    xhtml_body = '\n'.join(xhtml_pages)
                    xhtml_text = (
                        '<?xml version="1.0" encoding="utf-8"?>\n'
                        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
                        '<head>\n'
                        f'<meta charset="utf-8"/>\n<title>{title}</title>\n'
                        '</head>\n'
                        '<body>\n'
                        f'{xhtml_body}\n'
                        '</body>\n'
                        '</html>\n'
                    )
                    file_input = os.path.join(session['process_dir'], f'{filename_noext}.xhtml')
                    with open(file_input, 'w', encoding='utf-8') as html_file:
                        html_file.write(xhtml_text)
                else:
                    return False
            elif file_ext == '.pptx':
                from html import escape as html_escape
                from pptx import Presentation as PptxPresentation
                filename_noext = os.path.splitext(os.path.basename(session['ebook']))[0]
                msg = f'File input is a presentation ({file_ext}). Extracting content…'
                print(msg)
                prs = PptxPresentation(file_input)
                title = prs.core_properties.title or filename_noext
                author = prs.core_properties.author or False
                xhtml_pages = []
                for i, slide in enumerate(prs.slides):
                    slide_texts = []
                    slide_images = []
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for para in shape.text_frame.paragraphs:
                                text = para.text.strip()
                                if text:
                                    slide_texts.append(text)
                        if shape.has_table:
                            for row in shape.table.rows:
                                row_texts = [c.text.strip() for c in row.cells if c.text.strip()]
                                if row_texts:
                                    slide_texts.append(' | '.join(row_texts))
                        try:
                            slide_images.append(shape.image.blob)
                        except (AttributeError, ValueError):
                            pass
                    if slide_texts:
                        xhtml_content = '\n'.join(f'<p>{html_escape(t)}</p>' for t in slide_texts)
                    elif slide_images:
                        msg = f'Slide {i+1} seems to be image-based. Using OCR…'
                        show_alert(session_id, {"type": "warning", "msg": msg})
                        xhtml_parts = []
                        for blob in slide_images:
                            img = Image.open(io.BytesIO(blob))
                            xhtml_content, error = ocr2xhtml(img, session['language'])
                            if xhtml_content:
                                xhtml_parts.append(xhtml_content)
                            else:
                                show_alert(session_id, {"type": "warning", "msg": error})
                        xhtml_content = '\n'.join(xhtml_parts) if xhtml_parts else ''
                    else:
                        xhtml_content = ''
                    if xhtml_content:
                        xhtml_pages.append(xhtml_content)
                if xhtml_pages:
                    xhtml_body = '\n'.join(xhtml_pages)
                    xhtml_text = (
                        '<?xml version="1.0" encoding="utf-8"?>\n'
                        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
                        '<head>\n'
                        f'<meta charset="utf-8"/>\n<title>{title}</title>\n'
                        '</head>\n'
                        '<body>\n'
                        f'{xhtml_body}\n'
                        '</body>\n'
                        '</html>\n'
                    )
                    file_input = os.path.join(session['process_dir'], f'{filename_noext}.xhtml')
                    with open(file_input, 'w', encoding='utf-8') as html_file:
                        html_file.write(xhtml_text)
                else:
                    return False
            elif file_ext == '.docx':
                from docx import Document as DocxDocument
                filename_noext = os.path.splitext(os.path.basename(session['ebook']))[0]
                docx_doc = DocxDocument(file_input)
                all_text = ''.join(p.text.strip() for p in docx_doc.paragraphs)
                if not all_text:
                    for table in docx_doc.tables:
                        for row in table.rows:
                            for cell in row.cells:
                                all_text += cell.text.strip()
                                if all_text:
                                    break
                            if all_text:
                                break
                        if all_text:
                            break
                if not all_text:
                    msg = f'File input is a DOCX with no extractable text. Extracting images for OCR…'
                    print(msg)
                    title = docx_doc.core_properties.title or filename_noext
                    author = docx_doc.core_properties.author or False
                    xhtml_pages = []
                    for rel in docx_doc.part.rels.values():
                        if 'image' in rel.reltype:
                            try:
                                img = Image.open(io.BytesIO(rel.target_part.blob))
                                xhtml_content, error = ocr2xhtml(img, session['language'])
                                if xhtml_content:
                                    xhtml_pages.append(xhtml_content)
                                else:
                                    show_alert(session_id, {"type": "warning", "msg": error})
                            except Exception as e:
                                print(f'Error processing embedded image: {e}')
                    if xhtml_pages:
                        xhtml_body = '\n'.join(xhtml_pages)
                        xhtml_text = (
                            '<?xml version="1.0" encoding="utf-8"?>\n'
                            '<html xmlns="http://www.w3.org/1999/xhtml">\n'
                            '<head>\n'
                            f'<meta charset="utf-8"/>\n<title>{title}</title>\n'
                            '</head>\n'
                            '<body>\n'
                            f'{xhtml_body}\n'
                            '</body>\n'
                            '</html>\n'
                        )
                        file_input = os.path.join(session['process_dir'], f'{filename_noext}.xhtml')
                        with open(file_input, 'w', encoding='utf-8') as html_file:
                            html_file.write(xhtml_text)
                    else:
                        return False
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                filename_noext = os.path.splitext(os.path.basename(session['ebook']))[0]
                msg = f'File input is an image ({file_ext}). Running OCR…'
                print(msg)
                img = Image.open(file_input)
                xhtml_pages = []
                page_count = 0
                for i, frame in enumerate(ImageSequence.Iterator(img)):
                    page_count += 1
                    frame = frame.convert('RGB')
                    xhtml_content, error = ocr2xhtml(frame, session['language'])
                    if xhtml_content:
                        xhtml_pages.append(xhtml_content)
                    else:
                        show_alert(session_id, {"type": "warning", "msg": error})
                if xhtml_pages:
                    xhtml_body = '\n'.join(xhtml_pages)
                    xhtml_text = (
                        '<?xml version="1.0" encoding="utf-8"?>\n'
                        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
                        '<head>\n'
                        f'<meta charset="utf-8"/>\n<title>{filename_noext}</title>\n'
                        '</head>\n'
                        '<body>\n'
                        f'{xhtml_body}\n'
                        '</body>\n'
                        '</html>\n'
                    )
                    file_input = os.path.join(session['process_dir'], f'{filename_noext}.xhtml')
                    with open(file_input, 'w', encoding='utf-8') as html_file:
                        html_file.write(xhtml_text)
                    print(f'OCR completed for {page_count} image page(s).')
                else:
                    return False
            env = dict(os.environ)
            for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE', 'PYTHONNOUSERSITE', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'DYLD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES'):
                env.pop(key, None)
            prefixes = [os.path.normcase(os.path.normpath(p)) for p in (os.environ.get('CONDA_PREFIX'), os.environ.get('VIRTUAL_ENV'), sys.prefix if sys.prefix != sys.base_prefix else None) if p]
            env['PATH'] = os.pathsep.join([p for p in env.get('PATH', '').split(os.pathsep) if p and not any(os.path.normcase(os.path.normpath(p)) == pre or os.path.normcase(os.path.normpath(p)).startswith(pre + os.sep) for pre in prefixes)])
            env.setdefault('LANG', 'C.UTF-8')
            env.setdefault('LC_ALL', 'C.UTF-8')
            if sys.platform == systems['LINUX'] and not env.get('DISPLAY') and not env.get('WAYLAND_DISPLAY'):
                env.setdefault('QT_QPA_PLATFORM', 'offscreen')
            cmd_prefix = [ebook_convert]
            if sys.platform != systems['WINDOWS']:
                try:
                    with open(ebook_convert, 'rb') as f:
                        head = f.readline(256)
                    if head.startswith(b'#!'):
                        shebang = head[2:].decode('utf-8', 'replace').strip()
                        if shebang.startswith('/usr/bin/env') and 'python' in shebang:
                            interpreter = shutil.which(shebang.split()[-1], path=env['PATH'])
                            if interpreter:
                                cmd_prefix = [interpreter, ebook_convert]
                except OSError:
                    pass
            msg = f"Running command: {' '.join(cmd_prefix)} {file_input} {session['epub_path']}"
            print(msg)
            cmd = cmd_prefix + [
                    file_input, session['epub_path'],
                    '--input-encoding=utf-8',
                    '--output-profile=generic_eink',
                    '--flow-size=0',
                    '--chapter-mark=pagebreak',
                    '--page-breaks-before',
                    "//*[name()='h1' or name()='h2' or name()='h3' or name()='h4' or name()='h5']",
                    '--disable-font-rescaling',
                    '--pretty-print',
                    '--smarten-punctuation',
                    '--verbose'
                ]
            if title:
                cmd += ['--title', title]
            if author:
                cmd += ['--authors', author]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                env=env
            )
            if result.returncode != 0:
                error = f'ebook-convert exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
                print(error)
                show_alert(session_id, {"type": "error", "msg": error})
                return False
            if not os.path.exists(session['epub_path']) or os.path.getsize(session['epub_path']) == 0:
                error = f"ebook-convert produced no output: {session['epub_path']}"
                print(error)
                return False
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            DependencyError(e)
            error = f'convert2epub subprocess.CalledProcessError: {e.stderr}'
            print(error)
            return False
        except FileNotFoundError as e:
            DependencyError(e)
            error = f'convert2epub FileNotFoundError: {e}'
            print(error)
            return False
        except Exception as e:
            DependencyError(e)
            error = f'convert2epub error: {e}'
            print(error)
            return False

def get_ebook_title(epubBook:EpubBook, all_docs:list[Any])->str|None:
    # 1. Try metadata (official EPUB title)
    meta_title = epubBook.get_metadata('DC','title')
    if meta_title and meta_title[0][0].strip():
        return meta_title[0][0].strip()
    # 2. Try <title> in the head of the first XHTML document
    if all_docs:
        html = all_docs[0].get_content().decode('utf-8')
        soup = BeautifulSoup(html,'html.parser')
        title_tag = soup.select_one('head > title')
        if title_tag and title_tag.text.strip():
            return title_tag.text.strip()
        # 3. Try <img alt = '…'> if no visible <title>
        img = soup.find('img',alt = True)
        if img:
            alt = img['alt'].strip()
            if alt and 'cover' not in alt.lower():
                return alt
    return None

def get_cover(epubBook:EpubBook, session_id:str)->bool|str:
    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            if session['cancellation_requested']:
                return False
            cover_image = None
            cover_path = os.path.join(session['process_dir'], session['filename_noext'] + '.jpg')
            for item in epubBook.get_items_of_type(ebooklib.ITEM_COVER):
                cover_image = item.get_content()
                break
            if not cover_image:
                for item in epubBook.get_items_of_type(ebooklib.ITEM_IMAGE):
                    if 'cover' in item.file_name.lower() or 'cover' in item.get_id().lower():
                        cover_image = item.get_content()
                        break
            if cover_image:
                # Open the image from bytes
                image = Image.open(io.BytesIO(cover_image))
                # Convert to RGB if needed (JPEG doesn't support alpha)
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                image.save(cover_path, format = 'JPEG')
                return cover_path
            return True
    except Exception as e:
        DependencyError(e)
        return False

class SpinePiece:
    """Chapter cut out of a spine document, quacks like the EpubHtml used by filter_blocks()."""

    def __init__(self, doc:EpubHtml, html:str):
        self.doc = doc
        self.id = doc.id
        self.html = html

    def get_body_content(self)->bytes:
        return self.html.encode('utf-8')

    def get_name(self)->str:
        return self.doc.get_name()

def is_scanned_book(all_docs:list[Any])->bool:
    # OCR is for scanned books. In a text book the image-only pages are the cover, the title
    # page or illustrations: reading their OCR aloud gives garbage ("péi À Q ami").
    with_text = image_only = 0
    for doc in all_docs:
        try:
            soup = BeautifulSoup(doc.get_body_content(), 'html.parser')
        except Exception:
            continue
        if soup.get_text(strip=True):
            with_text += 1
        elif soup.find(['img', 'image']):
            image_only += 1
    scanned = image_only > with_text
    if image_only and not scanned:
        print(f'{image_only} image-only page(s) (cover, title page…) skipped, no OCR for a text book')
    return scanned

def split_spine_docs(all_docs:list[Any])->list[Any]:
    # one block per chapter: split documents bundling several chapters and glue
    # title-only pages (part titles, calibre heading page breaks) to their chapter
    try:
        by_id = {doc.id: doc for doc in all_docs}
        originals = {doc.id: doc.get_content() for doc in all_docs}
        pieces = split_spine([(doc.id, originals[doc.id]) for doc in all_docs])
        result = []
        for name, html, _title in pieces:
            doc_id = name.split('#', 1)[0]
            original = originals[doc_id]
            original = original.decode('utf-8', errors='ignore') if isinstance(original, bytes) else original
            if '#' not in name and html == original:
                result.append(by_id[doc_id])
            else:
                result.append(SpinePiece(by_id[doc_id], html))
        if len(result) != len(all_docs):
            print(f'Chapters detection: {len(all_docs)} spine documents -> {len(result)} chapters')
        return result
    except Exception as e:
        print(f'split_spine_docs() error: {e}, keeping the original spine')
        return all_docs

def get_blocks(session_id:str, epubBook:EpubBook)->list:
    try:
        msg = r'''
*******************************************************************************
NOTE:
The warning "Character xx not found in the vocabulary."
MEANS THE MODEL CANNOT INTERPRET THE CHARACTER AND WILL MAYBE GENERATE
(AS WELL AS WRONG PUNCTUATION POSITION) AN HALLUCINATION. THE BEST SOLUTION IS
TO MANUALLY REMOVE ALL UNRECOGNIZED CHARS AND WRONG PUNCTUATIONS FROM YOUR EBOOK
AND RESTART THE CONVERSION. TO IMPROVE THIS MODEL, IT NEEDS TO ADD THIS CHARACTER
INTO A NEW TRAINING MODEL. YOU CAN IMPROVE IT OR ASK TO A TRAINING MODEL EXPERT.
*******************************************************************************
        '''
        print(msg)
        session = context.get_session(session_id)
        if session and session.get('id', False):
            if session['cancellation_requested']:
                return []
            # Step 1: Extract TOC (Table of Contents)
            try:
                toc = epubBook.toc
                toc_list = [
                        nt for item in toc if hasattr(item, 'title')
                        if (nt := normalize_text(str(item.title), session['language'], session['language_iso1'], session['tts_engine'])) is not None
                ]
            except Exception as toc_error:
                error = f'Error extracting Table of Content: {toc_error}'
                show_alert(session_id, {"type": "warning", "msg": error})
            # Get spine item IDs
            spine_ids = [item[0] for item in epubBook.spine]
            # Filter only spine documents (i.e., reading order)
            all_docs = [
                item for item in epubBook.get_items_of_type(ebooklib.ITEM_DOCUMENT)
                if item.id in spine_ids
            ]
            if not all_docs:
                error = 'No document body found!'
                print(error)
                return []
            title = get_ebook_title(epubBook, all_docs)
            all_docs = split_spine_docs(all_docs)
            allow_ocr = is_scanned_book(all_docs)
            blocks = []
            stanza_nlp = False
            if session['language'] in year_to_decades_languages:
                try:
                    stanza_model = f"stanza-{session['language_iso1']}"
                    stanza_nlp = loaded_tts.get(stanza_model, False)
                    if stanza_nlp:
                        msg = f"NLP model {stanza_model} loaded."
                        print(msg)
                    else:
                        use_gpu = True if (
                            (session['device'] == devices['CUDA']['proc'] and devices['CUDA']['found']) or
                            (session['device'] == devices['ROCM']['proc'] and devices['ROCM']['found']) or
                            (session['device'] == devices['XPU']['proc'] and devices['XPU']['found']) or
                            (session['device'] == devices['JETSON']['proc'] and devices['JETSON']['found'])
                        ) else False
                        # only use mwt if the language supports it
                        stanza_lang = session['language_iso1']
                        stanza_has_mwt = False
                        try:
                            stanza_resources = stanza.resources.common.load_resources_json(os.getenv('STANZA_RESOURCES_DIR', stanza.resources.common.DEFAULT_MODEL_DIR))
                            stanza_has_mwt = 'mwt' in stanza_resources.get(stanza_lang, {})
                        except Exception:
                            pass
                        stanza_processors = 'tokenize,mwt,ner' if stanza_has_mwt else 'tokenize,ner'
                        stanza_nlp = stanza.Pipeline(stanza_lang, processors=stanza_processors, use_gpu=use_gpu, download_method=DownloadMethod.REUSE_RESOURCES, dir=os.getenv('STANZA_RESOURCES_DIR'))
                        if stanza_nlp:
                            session['stanza_cache'] = stanza_model
                            loaded_tts[stanza_model] = stanza_nlp
                            msg = f"NLP model {stanza_model} loaded!"
                            print(msg)
                except (ConnectionError, TimeoutError) as e:
                    error = f'Stanza model download connection error: {e}. Retry later'
                    print(error)
                    return []
                except Exception as e:
                    error = f'Stanza model initialization error: {e}'
                    print(error)
                    return []
            is_num2words_compat = get_num2words_compat(session['language_iso1'])
            non_text_filter = NonTextFilter(sml_pattern=SML_TAG_PATTERN, lang=session['language'])
            try:
                with zipfile.ZipFile(session['epub_path'], 'r') as zf:
                    zip_names = set(zf.namelist())
                    zip_basenames = {os.path.basename(n): n for n in zip_names}
                    for doc_idx, doc in enumerate(all_docs):
                        text = filter_blocks(session_id, doc_idx, doc, stanza_nlp, is_num2words_compat, non_text_filter, zf, zip_names, zip_basenames, allow_ocr)
                        if text is None:
                            error = f'Error extracting content from document #{doc_idx + 1}; aborting conversion to avoid partial output.'
                            show_alert(session_id, {"type": "warning", "msg": error})
                            return []
                        blocks.append(text)
            finally:
                if stanza_nlp:
                    import gc, torch
                    try:
                        cache_key = session.get('stanza_cache')
                        if cache_key:
                            loaded_tts.pop(cache_key, None)
                        session['stanza_cache'] = None
                    except Exception:
                        pass
                    stanza_nlp = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            if len(blocks) == 0:
                error = 'No blocks found! possible reason: file corrupted or need to convert images to text with OCR'
                print(error)
                return []
            return blocks
        return []
    except Exception as e:
        error = f'Error extracting main content pages: {e}'
        DependencyError(error)
        return []

def filter_blocks(session_id:str, idx:int, doc:EpubHtml, stanza_nlp:Pipeline, is_num2words_compat:bool, non_text_filter:NonTextFilter, zf:zipfile.ZipFile=None, zip_names:set=None, zip_basenames:dict=None, allow_ocr:bool=True)->str|None:

    def _tuple_row(node:Any, last_text_char:str|None=None, in_heading:bool=False)->Generator[tuple[str, Any], None, None]|None:
        try:
            prev_child_had_data = False
            for idx, child in enumerate(node.children):
                current_child_had_data = False
                if isinstance(child, NavigableString):
                    text = child.strip()
                    if text:
                        # Only add break if not inside a heading and previous had data
                        if prev_child_had_data and not in_heading:
                            yield ('break', sml_token("break"))
                        # Yield as heading if inside heading, else text
                        yield ('heading' if in_heading else 'text', text)
                        last_text_char = text[-1]
                        current_child_had_data = True
                elif isinstance(child, Tag):
                    name = child.name.lower()
                    if name in heading_tags:
                        # Process heading tag and all its children as heading content
                        title = child.get_text(strip=True)
                        if title:
                            # Add break before heading if needed (only if not already in heading)
                            if prev_child_had_data and not in_heading:
                                yield ('break', sml_token("break"))
                            # Always process heading children recursively to handle nested elements
                            for inner in _tuple_row(child, last_text_char, in_heading=True):
                                yield inner
                            # After finishing the heading, yield a pause marker
                            yield ('pause', sml_token("pause"))
                            last_text_char = title[-1]
                            current_child_had_data = True
                    elif name == 'table':
                        if prev_child_had_data and not in_heading:
                            yield ('break', sml_token("break"))
                        yield ('table', child)
                        current_child_had_data = True
                    else:
                        return_data = False
                        if name in proc_tags:
                            is_header = False
                            # Add break before processing tag content if needed
                            if prev_child_had_data and name in break_tags and not in_heading:
                                yield ('break', sml_token("break"))
                            for inner in _tuple_row(child, last_text_char, in_heading=in_heading):
                                return_data = True
                                yield inner
                                if len(inner) > 1 and isinstance(inner[1], str) and inner[1]:
                                    last_text_char = inner[1][-1]
                                current_child_had_data = True
                                if inner[0] in ('text', 'heading') and isinstance(inner[1], str) and inner[1]:
                                    is_header = True
                            if return_data:
                                if name in break_tags and name != 'span' and not in_heading:
                                    if is_header or (last_text_char and not last_text_char.isalnum() and not last_text_char.isspace()):
                                        yield ('break', sml_token("break"))
                                elif name in pause_tags and not in_heading:
                                    yield ('pause', sml_token("pause"))
                        else:
                            yield from _tuple_row(child, last_text_char, in_heading=in_heading)
                            current_child_had_data = True
                if current_child_had_data:
                    prev_child_had_data = True
        except Exception as e:
            error = f'filter_blocks() _tuple_row() error: {e}'
            DependencyError(error)
            return None

    def _num_repl(m):
        s = m.group(0)
        # leave years alone (already handled above)
        if re.fullmatch(r"\d{4}", s):
            return s
        n = float(s) if '.' in s else int(s)
        if is_num2words_compat:
            return num2words(n, lang=(lang_iso1 or 'en'))
        else:
            return math2words(m, lang, lang_iso1, tts_engine, is_num2words_compat)

    try:
        msg = f'----------\nParsing doc {idx}'
        print(msg)
        session = context.get_session(session_id)
        if session and session.get('id', False):
            lang, lang_iso1, tts_engine = session['language'], session['language_iso1'], session['tts_engine']
            heading_tags = [f'h{i}' for i in range(1, 5)]
            break_tags = ['br', 'p', 'span']
            pause_tags = ['div']
            proc_tags = heading_tags + break_tags + pause_tags
            doc_body = doc.get_body_content()
            raw_html = doc_body.decode('utf-8') if isinstance(doc_body, bytes) else doc_body
            soup = BeautifulSoup(raw_html, 'html.parser')
            body = soup.body
            if not body:
                msg = 'No body found. Skip to next doc…'
                print(msg)
                return ''
            # Skip known non-chapter types (front/back matter, navigation, epigraphs)
            if is_excluded_doc(soup):
                msg = 'No body part. Skip to next doc…'
                print(msg)
                return ''
            # remove scripts/styles
            for tag in soup(['script', 'style']):
                tag.decompose()
            # notes, page markers, inline formatting, superscripts, all-caps blocks, bullets
            # and missing final punctuation are fixed on the markup, before flattening
            prepare_html_for_tts(soup, lang)
            if not body.get_text(strip=True):
                images = body.find_all('img') + body.find_all('image')
                if images and zf and allow_ocr:
                    msg = f'Doc {idx}: no text but {len(images)} image(s) detected. Running OCR…'
                    show_alert(session_id, {"type": "info", "msg": msg})
                    ocr_parts = []
                    doc_dir = os.path.dirname(doc.get_name())
                    for img_tag in images:
                        img_ref = (
                            img_tag.get('src')
                            or img_tag.get('href')
                            or img_tag.get('{http://www.w3.org/1999/xlink}href')
                            or img_tag.get('xlink:href')
                        )
                        if not img_ref:
                            continue
                        img_zip_path = os.path.normpath(os.path.join(doc_dir, img_ref)).replace('\\', '/')
                        if img_zip_path not in zip_names:
                            img_zip_path = zip_basenames.get(os.path.basename(img_ref))
                        if not img_zip_path:
                            print(f'Could not resolve image in EPUB: {img_ref}')
                            continue
                        try:
                            img_data = zf.read(img_zip_path)
                            img = Image.open(io.BytesIO(img_data))
                            img = img.convert('RGB')
                            xhtml_content, error = ocr2xhtml(img, lang)
                            if xhtml_content:
                                ocr_parts.append(xhtml_content)
                            else:
                                show_alert(session_id, {"type": "warning", "msg": error})
                        except Exception as ocr_err:
                            print(f'OCR error on {img_zip_path}: {ocr_err}')
            tuples_list = list(_tuple_row(body))
            if not tuples_list:
                msg = 'No body text and no images found. Skip to next doc…'
                print(msg)
                return ''
            msg = f'Parsing xhtml markers…'
            print(msg)
            text_list = []
            handled_tables = set()
            prev_typ = None
            for typ, payload in tuples_list:
                if typ == 'heading':
                    text_list.append(payload.strip())
                elif typ in ('break', 'pause'):
                    # Original logic: add token if different from previous type
                    if prev_typ != typ:
                        token = sml_token(typ)
                        if text_list and text_list[-1] not in {v['static'] for v in TTS_SML.values() if 'static' in v}:
                            # a heading closes its intonation like a sentence before the pause
                            sep = '. ' if prev_typ == 'heading' and text_list[-1].rstrip()[-1:] not in punctuation_list_set else ''
                            text_list[-1] = text_list[-1] + sep + token
                        else:
                            text_list.append(token)
                elif typ == 'table':
                    table = payload
                    if table in handled_tables:
                        prev_typ = typ
                        continue
                    handled_tables.add(table)
                    rows = table.find_all('tr')
                    if not rows:
                        prev_typ = typ
                        continue
                    headers = [c.get_text(strip=True) for c in rows[0].find_all(['td', 'th'])]
                    for row in rows[1:]:
                        cells = [c.get_text(strip=True).replace('\xa0', ' ') for c in row.find_all('td')]
                        if not cells:
                            continue
                        if len(cells) == len(headers) and headers:
                            line = ' — '.join(f'{h}: {c}' for h, c in zip(headers, cells))
                        else:
                            line = ' — '.join(cells)
                        if line:
                            text_list.append(line.strip())
                else:
                    text = payload.strip()
                    if text:
                        text_list.append(text)
                prev_typ = typ
            msg = f'Flattening as raw text…'
            print(msg)
            max_chars = int(language_mapping[lang]['max_chars'] / 1.5)
            clean_list = []
            i = 0
            while i < len(text_list):
                current = text_list[i]
                if current in {v['static'] for v in TTS_SML.values() if "static" in v}:
                    if clean_list:
                        prev = clean_list[-1]
                        if prev in {v['static'] for v in TTS_SML.values() if "static" in v}:
                            i += 1
                            continue
                    clean_list.append(current)
                    i += 1
                    continue
                clean_list.append(current)
                i += 1
            text = ' '.join(clean_list)
            if not re.search(r"[^\W_]", text):
                error = 'No valid text found!'
                print(error)
                return None
            # clean SML tags badly coded
            res, text = normalize_sml_tags(text)
            if res is False:
                show_alert(session_id, {"type": "warning", "msg": text})
                return None
            # remove any [break] between words or cutting words
            break_token = re.escape(sml_token('break'))
            strip_break_spaces_re = re.compile(rf'\s*{break_token}\s*')
            break_between_alnum_re = re.compile(rf'(?<=[\w]){break_token}(?=[\w])', flags=re.UNICODE)
            text = strip_break_spaces_re.sub(sml_token('break'), text)
            text = break_between_alnum_re.sub(' ', text)
            # strip non-prose content; preserves math signs for math2words
            text = non_text_filter(text)
            # escape all SML tags to not be touched by any text treatment
            text, sml_blocks = escape_sml(text)
            if stanza_nlp:
                msg = 'Converting dates and years to words…'
                print(msg)
                re_ordinal = re.compile(
                    r'(?<!\w)(0?[1-9]|[12][0-9]|3[01])(?:\s|\u00A0)*(?:st|nd|rd|th)(?!\w)',
                    re.IGNORECASE
                )
                re_num = re.compile(r'(?<!\w)[-+]?\d+(?:\.\d+)?(?!\w)')
                text = unicodedata.normalize('NFKC', text).replace('\u00A0', ' ')
                re_year = re.compile(r'\b(?:1[0-9]|20)\d{2}\b')
                if re_num.search(text) and (re_ordinal.search(text) or re_year.search(text)):
                    date_spans = get_date_entities(text, stanza_nlp)
                    if date_spans:
                        result = []
                        last_pos = 0
                        for start, end, date_text in date_spans:
                            result.append(text[last_pos:start])
                            # 1) convert 4-digit years (your original behavior)
                            processed = re.sub(
                                r"\b\d{4}\b",
                                lambda m: year2words(m.group(), lang, lang_iso1, is_num2words_compat),
                                date_text
                            )
                            # 2) convert ordinal days like "16th"/"16 th"->"sixteenth"
                            if is_num2words_compat:
                                processed = re_ordinal.sub(
                                    lambda m: num2words(int(m.group(1)), to='ordinal', lang=(lang_iso1 or 'en')),
                                    processed
                                )
                            else:
                                processed = re_ordinal.sub(
                                    lambda m: math2words(m.group(), lang, lang_iso1, tts_engine, is_num2words_compat),
                                    processed
                                )
                            # 3) convert other numbers (skip 4-digit years)
                            processed = re_num.sub(_num_repl, processed)
                            result.append(processed)
                            last_pos = end
                        result.append(text[last_pos:])
                        text = ' '.join(result)
                    else:
                        if is_num2words_compat:
                            text = re_ordinal.sub(
                                lambda m: num2words(int(m.group(1)), to='ordinal', lang=(lang_iso1 or 'en')),
                                text
                            )
                        else:
                            text = re_ordinal.sub(
                                lambda m: math2words(int(m.group(1)), lang, lang_iso1, tts_engine, is_num2words_compat),
                                text
                            )
                        text = re.sub(
                            r"\b\d{4}\b",
                            lambda m: year2words(m.group(), lang, lang_iso1, is_num2words_compat),
                            text
                        )
            msg = 'Convert romans to numbers…'
            print(msg)
            text = roman2number(text)
            msg = 'Convert time to words…'
            print(msg)
            text = clock2words(text, lang, lang_iso1, tts_engine, is_num2words_compat)
            msg = 'Convert numbers, maths signs to words…'
            print(msg)
            text = math2words(text, lang, lang_iso1, tts_engine, is_num2words_compat)
            msg = 'Normalize text…'
            print(msg)
            text = normalize_text(text, lang, lang_iso1, tts_engine)
            text = restore_sml(text, sml_blocks)
            return text
        return None
    except Exception as e:
        error = f'filter_blocks() error: {e}'
        DependencyError(error)
        return None

def get_sentences(session_id:str, text:str)->list|None:

    def _split_inclusive(text:str, pattern:re.Pattern[str])->list[str]:
        result = []
        last_end = 0
        for match in pattern.finditer(text):
            result.append(text[last_end:match.end()].strip())
            last_end = match.end()
        if last_end < len(text):
            tail = text[last_end:].strip()
            if tail:
                result.append(tail)
        return result

    def _split_sentence_on_sml(sentence:str)->list[str]:
        parts:list[str] = []
        last = 0
        i = 0
        n = len(sentence)
        while i < n:
            m = SML_TAG_PATTERN.match(sentence, i)
            if m:
                if m.start() > last:
                    parts.append(sentence[last:m.start()])
                parts.append(m.group(0))
                i = m.end()
                last = i
                continue
            if ord(sentence[i]) >= sml_escape_tag:
                if i > last:
                    parts.append(sentence[last:i])
                j = i
                while j < n and ord(sentence[j]) >= sml_escape_tag:
                    j += 1
                parts.append(sentence[i:j])
                i = j
                last = i
                continue
            i += 1
        if last < n:
            parts.append(sentence[last:])
        return parts

    def _strip_escaped_sml(s:str)->str:
        return ''.join(c for c in s if ord(c) < sml_escape_tag)

    def _clean_len(s:str)->int:
        return len(_strip_escaped_sml(s))

    def _is_latin_only(s:str)->bool:
        s = _strip_escaped_sml(s)
        s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)
        has_latin = bool(re.search(r'[A-Za-z]', s))
        has_nonlatin = bool(re.search(r'[^\x00-\x7F]', s))
        return has_latin and not has_nonlatin

    def _segment_ideogramms(text:str)->list[str]:
        result = []
        try:
            if lang in ['yue','yue-Hant','yue-Hans','zh-yue','cantonese']:
                import pycantonese as pc
                result.extend([t for t in pc.segment(text) if t.strip()])
            elif lang == 'zho':
                import jieba
                jieba.dt.cache_file = os.path.join(models_dir, 'jieba.cache')
                result.extend([t for t in jieba.cut(text) if t.strip()])
            elif lang == 'jpn':
                import nagisa
                result.extend(nagisa.tagging(text).words)
            elif lang == 'kor':
                from soynlp.tokenizer import LTokenizer
                ltokenizer = LTokenizer()
                result.extend([t for t in ltokenizer.tokenize(text) if t.strip()])
            elif lang in ['tha','lao','mya','khm']:
                from pythainlp.tokenize import word_tokenize
                result.extend([t for t in word_tokenize(text, engine='newmm') if t.strip()])
            else:
                result.append(text.strip())
            return result
        except Exception as e:
            DependencyError(e)
            return [text]

    def _join_ideogramms(idg_list:list[str])->str:
        try:
            buffer = ''
            prev_latin = False
            prev_nonlatin = False
            for token in idg_list:
                cur_starts_latin = bool(re.match(r'[A-Za-z0-9]', token))
                cur_starts_nonlatin = bool(re.match(r'[^\x00-\x7F]', token))
                if buffer:
                    if (prev_latin and (cur_starts_latin or cur_starts_nonlatin)) or (prev_nonlatin and cur_starts_latin):
                        buffer += ' '
                    elif len(buffer) + len(token) > max_chars:
                        yield buffer
                        buffer = ''
                buffer += token
                prev_latin = bool(re.search(r'[A-Za-z0-9]$', token))
                prev_nonlatin = bool(re.search(r'[^\x00-\x7F]$', token))
            if buffer:
                yield buffer
        except Exception as e:
            DependencyError(e)
            if buffer:
                yield buffer

    def _is_pure_escaped_sml(s:str)->bool:
        return bool(s) and all(ord(c) >= sml_escape_tag for c in s)

    def _strip_leading_noise(s:str)->str:
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c.isalnum() or c == '_' or c.isspace() or ord(c) >= sml_escape_tag:
                break
            i += 1
        return s[i:].lstrip()

    def _force_split_segment(segment:str)->list[str]:
        results = []
        rest = segment.strip()
        hard_set = tuple(punctuation_split_hard_set)
        soft_set = tuple(punctuation_split_soft_set)

        if not rest:
            return []

        clause_words = language_clause_split_words.get(lang, default_clause_split_words)
        clause_pattern = re.compile(
            r'(?i)\b(?:' + '|'.join(map(re.escape, clause_words)) + r')\b'
        )

        # If there is no punctuation at all, prefer a natural clause boundary instead of
        # slicing in the middle of the phrase. This keeps long unpunctuated clauses readable.
        if not any(ch in hard_set or ch in soft_set for ch in rest):
            clause_positions = [m.end() for m in clause_pattern.finditer(rest) if m.end() > 0.35 * len(rest)]
            if clause_positions:
                best_idx = max(clause_positions)
                left = rest[:best_idx].strip()
                right = rest[best_idx:].strip()
                if left and right:
                    results.extend(_force_split_segment(left))
                    results.extend(_force_split_segment(right))
                    return results
            return [rest]

        while rest:
            if _clean_len(rest) <= max_chars:
                sentence_end = -1
                for i, char in enumerate(rest[:-1]):
                    if char not in hard_set:
                        continue
                    j = i + 1
                    while j < len(rest) and rest[j].isspace():
                        j += 1
                    if j < len(rest) and (rest[j].isupper() or rest[j].isdigit()):
                        sentence_end = i + 1
                        break
                if sentence_end > 0:
                    results.append(rest[:sentence_end].strip())
                    rest = rest[sentence_end:].strip()
                    continue
                results.append(rest.strip())
                break

            target = min(len(rest), max_chars)
            best_idx = -1

            # 1) prefer a real sentence-ending punctuation before the target length
            for i in range(min(target, len(rest) - 1), -1, -1):
                if rest[i] in hard_set:
                    best_idx = i + 1
                    break

            # 2) if no hard end was found nearby, allow a hard sentence end a little beyond the
            # target when it truly closes the sentence (next token starts a new sentence)
            if best_idx == -1:
                scan_end = min(len(rest) - 1, target + max_chars * 2)
                for i in range(target, scan_end + 1):
                    if rest[i] in hard_set:
                        j = i + 1
                        while j < len(rest) and rest[j].isspace():
                            j += 1
                        next_text = rest[j:j + 20]
                        if next_text and (next_text[0].isupper() or next_text[0].isdigit()):
                            best_idx = i + 1
                            break

            # 3) soft punctuation is the preferred natural split before clause connectors.
            # In practice, users prefer a later comma/semicolon over an earlier conjunction
            # when the sentence is long, because it preserves the narration rhythm. Never
            # search beyond the next hard sentence boundary.
            if best_idx == -1:
                next_hard_end = next(
                    (i for i in range(target, len(rest)) if rest[i] in hard_set),
                    len(rest)
                )
                for i in range(next_hard_end - 1, -1, -1):
                    if rest[i] in soft_set:
                        if i >= max_chars * 0.35 or i >= target:
                            best_idx = i + 1
                            break

            # 4) if there is still no safe break, prefer a clause connector before the target
            if best_idx == -1:
                clause_positions = [m.end() for m in clause_pattern.finditer(rest[:target]) if m.end() > 0.35 * target]
                if clause_positions:
                    best_idx = max(clause_positions)

            # 5) final fallback: whitespace near the max length
            if best_idx == -1:
                idx = rest.rfind(' ', 0, target)
                if idx > 0:
                    best_idx = idx + 1

            if best_idx <= 0 or best_idx >= len(rest):
                best_idx = max(1, min(len(rest) - 1, target))

            # Safety: never cut inside an SML group
            while best_idx < len(rest) and ord(rest[best_idx]) >= sml_escape_tag:
                best_idx += 1

            left = rest[:best_idx].strip()
            right = rest[best_idx:].strip()
            if not left or right == rest:
                results.append(rest.strip())
                break
            results.append(left)
            rest = right
        return results

    try:
        session = context.get_session(session_id)
        if not session:
            return None
        lang = session['language']
        if session.get('translate_enabled') and session.get('translate'):
            lang = session['translate']
        tts_engine = session['tts_engine']
        max_chars = int(language_mapping[lang]['max_chars'] / 2)

        if lang not in ['zho', 'jpn', 'kor', 'tha', 'lao', 'mya', 'khm']:
            # XTTS keeps a natural intonation on whole sentences up to its per-language
            # limit, cutting at max_chars/2 forced splits in the middle of most sentences
            hard_max = int(language_mapping[lang]['max_chars'] * 0.9) if tts_engine == TTS_ENGINES['XTTS'] else max_chars
            # engines with their own generation length limit (F5-TTS ~12 s per call)
            hard_max = min(hard_max, default_engine_settings.get(tts_engine, {}).get('max_chars') or hard_max)
            chunker = NaturalChunker(
                hard_max=hard_max,
                clause_words=language_clause_split_words.get(lang, default_clause_split_words),
                sml_pattern=SML_TAG_PATTERN,
                pack_sentences=bool(session.get('pack_sentences', True))
            )
            chunks = [_strip_leading_noise(c) for c in chunker.chunk(text)]
            return [c for c in chunks if c and any(ch.isalnum() for ch in c)]

        text, sml_blocks = escape_sml(text)
        assert not SML_TAG_PATTERN.search(text)

        # Tokenize into content and SML runs
        segments = []
        idx = 0
        tlen = len(text)
        current_text = []
        while idx < tlen:
            c = text[idx]
            if ord(c) >= sml_escape_tag:
                if current_text:
                    segments.append(('text', ''.join(current_text)))
                    current_text = []
                start = idx
                while idx < tlen and ord(text[idx]) >= sml_escape_tag:
                    idx += 1
                segments.append(('sml', text[start:idx]))
            else:
                current_text.append(c)
                idx += 1
        if current_text:
            segments.append(('text', ''.join(current_text)))

        # SINGLE inline buffer — SML stays in position next to its surrounding text.
        # On overflow, cut at the LAST SML position in the buffer (a natural pause point).
        final_list = []
        buffer = []
        current_len = 0

        for seg_type, seg_content in segments:
            if seg_type == 'sml':
                buffer.append(seg_content)
                continue

            seg_clean_len = _clean_len(seg_content)
            potential_len = current_len + seg_clean_len

            if potential_len <= max_chars:
                buffer.append(seg_content)
                current_len = potential_len
                continue

            # Doesn't fit. Try to cut at the rightmost SML run in the buffer.
            combined = ''.join(buffer)
            cut_idx = -1
            j = len(combined) - 1
            while j >= 0:
                if ord(combined[j]) >= sml_escape_tag:
                    cut_idx = j + 1
                    break
                j -= 1

            cut_done = False
            if 0 < cut_idx <= len(combined):
                part1 = combined[:cut_idx]
                part2 = combined[cut_idx:]
                if seg_clean_len <= max_chars and _clean_len(part2) + seg_clean_len <= max_chars:
                    if _strip_escaped_sml(part1).strip():
                        final_list.append(part1.strip())
                    elif part1.strip() and final_list:
                        # part1 is pure SML — attach to previous sentence, no standalone tag
                        final_list[-1] = final_list[-1].rstrip() + ' ' + part1.strip()
                    buffer = [part2, seg_content] if part2 else [seg_content]
                    current_len = _clean_len(part2) + seg_clean_len
                    cut_done = True

            if not cut_done:
                # No usable SML cut. Flush buffer wholesale (force-split if too long).
                if _strip_escaped_sml(combined).strip():
                    if current_len <= max_chars:
                        final_list.append(combined.strip())
                    else:
                        final_list.extend(_force_split_segment(combined.strip()))
                    if seg_clean_len > max_chars:
                        parts = _force_split_segment(seg_content)
                        for p in parts[:-1]:
                            if p:
                                final_list.append(p)
                        buffer = [parts[-1]]
                        current_len = _clean_len(parts[-1])
                    else:
                        buffer = [seg_content]
                        current_len = seg_clean_len
                else:
                    # Buffer is pure SML — carry forward, never emit a standalone [break]
                    pending = combined
                    if seg_clean_len > max_chars:
                        parts = _force_split_segment(seg_content)
                        if parts and pending.strip():
                            parts[0] = pending + parts[0]
                        for p in parts[:-1]:
                            if p:
                                final_list.append(p)
                        buffer = [parts[-1]]
                        current_len = _clean_len(parts[-1])
                    else:
                        buffer = [pending, seg_content] if pending.strip() else [seg_content]
                        current_len = seg_clean_len

        # Final flush
        if buffer:
            combined = ''.join(buffer).strip()
            if combined:
                if _strip_escaped_sml(combined).strip():
                    if _clean_len(combined) <= max_chars:
                        final_list.extend(_force_split_segment(combined))
                    else:
                        final_list.extend(_force_split_segment(combined))
                elif final_list:
                    # Trailing pure-SML — attach to last sentence, no standalone
                    final_list[-1] = final_list[-1].rstrip() + ' ' + combined

        normalized_list = []
        for item in final_list:
            if _clean_len(item) <= max_chars:
                normalized_list.extend(_force_split_segment(item))
            else:
                normalized_list.append(item)
        final_list = [_strip_leading_noise(s) for s in normalized_list if s.strip()]
        final_list = [s for s in final_list if s]

        if lang in ['zho', 'jpn', 'kor', 'tha', 'lao', 'mya', 'khm']:
            result = []
            for s in final_list:
                parts = _split_sentence_on_sml(s)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if _is_pure_escaped_sml(part) or SML_TAG_PATTERN.fullmatch(part):
                        result.append(part)
                        continue
                    tokens = _segment_ideogramms(part)
                    if isinstance(tokens, list):
                        result.extend([t for t in tokens if t.strip()])
                    else:
                        tokens = tokens.strip()
                        if tokens:
                            result.append(tokens)
            ideogram_list = []
            for s in _join_ideogramms(result):
                if not _is_latin_only(s):
                    ideogram_list.append(s)
            if ideogram_list:
                ideogram_list = [restore_sml(s, sml_blocks) for s in ideogram_list]
            return ideogram_list

        if final_list:
            final_list = [restore_sml(s, sml_blocks) for s in final_list]
        return final_list
    except Exception as e:
        print(f'get_sentences() error: {e}')
        return None

def natural_sort_key(path:str)->list:
    """Natural sort key on the file BASENAME only.
    Full paths must NOT be sorted: gradio uploaded files live in per-file
    hashed cache dirs (<cache>/<hash>/<name>.<ext>) whose hash would
    dominate the order. Digit runs compare numerically so
    'volume-2' < 'volume-10'."""
    name = os.path.basename(str(path)).casefold()
    return [int(chunk) if chunk.isdigit() else chunk
            for chunk in re.split(r'(\d+)', name)]

def get_sanitized(str:str, replacement:str='_')->str:
    str = str.replace('&', 'And')
    forbidden_chars = r'[<>:"/\\|?*\x00-\x1F ()]'
    sanitized = re.sub(r'\s+', replacement, str)
    sanitized = re.sub(forbidden_chars, replacement, sanitized)
    sanitized = sanitized.strip('_')
    return sanitized
    
def get_date_entities(text:str, stanza_nlp:Pipeline)->list[tuple[int,int,str]]|bool:
    try:
        doc = stanza_nlp(text)
        date_spans = []
        for ent in doc.ents:
            if ent.type == 'DATE':
                date_spans.append((ent.start_char, ent.end_char, ent.text))
        return date_spans
    except Exception as e:
        error = f'get_date_entities() error: {e}'
        print(error)
        return False

def get_num2words_compat(lang_iso1:str)->bool:
    try:
        test = num2words(1, lang=lang_iso1.replace('zh', 'zh_CN'))
        return True
    except NotImplementedError:
        return False
    except Exception as e:
        return False

def set_formatted_number(text:str, lang:str, lang_iso1:str, is_num2words_compat:bool, max_single_value:int=999_999_999_999_999_999)->str:
    # match up to 18 digits, optional “,…” groups (allowing spaces or NBSP after comma), optional decimal of up to 12 digits
    # handle optional range with dash/en dash/em dash between numbers, and allow trailing punctuation
    number_re = re.compile(
        r'(?<!\w)'
        r'(\d{1,18}(?:,\d{3})*(?:\.\d{1,12})?)'            # first number
        r'(?:\s*([-–—])\s*'                                # dash type
        r'(\d{1,18}(?:,\d{3})*(?:\.\d{1,12})?))?'          # optional second number
        r'([^\w\s]*)',                                     # optional trailing punctuation
        re.UNICODE
    )
    if lang in decimal_comma_languages:
        # "1,83 mètre": the comma is the decimal separator, "1,830" is not a thousand
        text = re.sub(r'(?<![\d,.])(\d{1,15}),(\d{1,12})(?![\d,])', r'\1.\2', text)

    def _normalize_commas(num_str:str)->str:
        # ormalize number string to standard comma format: 1,234,567
        tok = num_str.replace('\u00A0', '').replace(' ', '')
        if '.' in tok:
            integer_part, decimal_part = tok.split('.', 1)
            integer_part = integer_part.replace(',', '')
            integer_part = "{:,}".format(int(integer_part))
            return f'{integer_part}.{decimal_part}'
        else:
            integer_part = tok.replace(',', '')
            return "{:,}".format(int(integer_part))

    def _clean_single_num(num_str:str)->str:
        tok = unicodedata.normalize('NFKC', num_str)
        if tok.lower() in ('inf', 'infinity', 'nan'):
            return tok
        clean = tok.replace(',', '').replace('\u00A0', '').replace(' ', '')
        try:
            num = float(clean) if '.' in clean else int(clean)
        except (ValueError, OverflowError):
            return tok
        if not math.isfinite(num) or abs(num) > max_single_value:
            return tok

        # Normalize commas before final output
        tok = _normalize_commas(tok)

        if is_num2words_compat:
            new_lang_iso1 = lang_iso1.replace('zh', 'zh_CN')
            if isinstance(num, float) and lang in decimal_comma_languages and '.' in clean:
                # "1,83" is read "un virgule quatre-vingt-trois", not digit by digit
                int_part, dec_part = clean.lstrip('+-').split('.', 1)
                comma_word = language_math_phonemes.get(lang, {}).get(',', ',')
                zeros = len(dec_part) - len(dec_part.lstrip('0'))
                zero_word = num2words(0, lang=new_lang_iso1)
                dec_words = ' '.join([zero_word] * zeros + ([num2words(int(dec_part), lang=new_lang_iso1)] if dec_part.strip('0') else []))
                sign = num2words(-1, lang=new_lang_iso1).split()[0] + ' ' if clean.startswith('-') else ''
                return f'{sign}{num2words(int(int_part or 0), lang=new_lang_iso1)} {comma_word} {dec_words}'.strip()
            return num2words(num, lang=new_lang_iso1)
        else:
            phoneme_map = language_math_phonemes.get(
                lang,
                language_math_phonemes.get(default_language_code, language_math_phonemes['eng'])
            )
            return ' '.join(phoneme_map.get(ch, ch) for ch in str(num))

    def clean_match(match:re.Match)->str:
        first_num = _clean_single_num(match.group(1))
        dash_char = match.group(2) or ''
        second_num = _clean_single_num(match.group(3)) if match.group(3) else ''
        trailing = match.group(4) or ''
        if second_num:
            range_word = number_range_words.get(lang)
            if range_word:
                return f'{first_num} {range_word} {second_num}{trailing}'
            return f'{first_num}{dash_char}{second_num}{trailing}'
        else:
            return f'{first_num}{trailing}'

    return number_re.sub(clean_match, text)

def year2words(year_str:str, lang:str, lang_iso1:str, is_num2words_compat:bool)->str|bool:
    try:
        year = int(year_str)
        first_two = int(year_str[:2])
        last_two = int(year_str[2:])
        lang_iso1 = lang_iso1 if lang in language_math_phonemes.keys() else default_language_code
        lang_iso1 = lang_iso1.replace('zh', 'zh_CN')
        if not year_str.isdigit() or len(year_str) != 4 or last_two < 10:
            if is_num2words_compat:
                return num2words(year, lang=lang_iso1)
            else:
                return ' '.join(language_math_phonemes[lang].get(ch, ch) for ch in year_str)
        if is_num2words_compat:
            return f'{num2words(first_two, lang=lang_iso1)} {num2words(last_two, lang=lang_iso1)}' 
        else:
            return ' '.join(language_math_phonemes[lang].get(ch, ch) for ch in first_two) + ' ' + ' '.join(language_math_phonemes[lang].get(ch, ch) for ch in last_two)
    except Exception as e:
        error = f'year2words() error: {e}'
        print(error)
        return False

def clock2words(text:str, lang:str, lang_iso1:str, tts_engine:str, is_num2words_compat:bool)->str:

    def _n2w(n:int)->str:
        key = (n, lang, is_num2words_compat)
        if key in _n2w_cache:
            return _n2w_cache[key]
        if is_num2words_compat:
            word = num2words(n, lang=lang_iso1)
        else:
            word = math2words(n, lang, lang_iso1, tts_engine, is_num2words_compat)
        if not isinstance(word, str):
            word = str(word)
        _n2w_cache[key] = word
        return word

    def _repl_num(m:re.Match)->str:
        # Reject enumeration patterns like "(1.2)"
        start, end = m.start(), m.end()
        if (
            start > 0 and end < len(text)
            and text[start - 1] == '('
            and text[end] == ')'
        ):
            return m.group(0)
        # Parse hh[:mm[:ss]]
        try:
            h = int(m.group(1))
            mnt = int(m.group(2))
            sec = m.group(3)
            sec = int(sec) if sec is not None else None
        except Exception:
            return m.group(0)
        # basic validation; if out of range, keep original
        if not (0 <= h <= 23 and 0 <= mnt <= 59 and (sec is None or 0 <= sec <= 59)):
            return m.group(0)
        # If no language clock rules, just say numbers plainly
        if not lc:
            parts = [_n2w(h)]
            if mnt != 0:
                parts.append(_n2w(mnt))
            if sec is not None and sec > 0:
                parts.append(_n2w(sec))
            return ' '.join(parts)
        next_hour = (h + 1) % 24
        special_hours = lc.get('special_hours', {})
        if mnt == 0 and (sec is None or sec == 0):
            if h in special_hours:
                phrase = special_hours[h]
            else:
                phrase = lc['oclock'].format(hour=_n2w(h))
        elif mnt == 15:
            phrase = lc['quarter_past'].format(hour=_n2w(h))
        elif mnt == 30:
            phrase = lc['half_past'].format(hour=_n2w(h), next_hour=_n2w(next_hour))
        elif mnt == 45:
            phrase = lc['quarter_to'].format(next_hour=_n2w(next_hour))
        elif mnt < 30:
            phrase = lc['past'].format(hour=_n2w(h), minute=_n2w(mnt)) if mnt != 0 else lc['oclock'].format(hour=_n2w(h))
        else:
            minute_to_hour = 60 - mnt
            phrase = lc['to'].format(
                next_hour=_n2w(next_hour),
                minute=_n2w(minute_to_hour),
                minute_to_hour=_n2w(minute_to_hour)
            )
        if sec is not None and sec > 0:
            second_phrase = lc['second'].format(second=_n2w(sec))
            phrase = lc['full'].format(phrase=phrase, second_phrase=second_phrase)
        return phrase

    time_rx = re.compile(
        r'\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b'
    )
    lc = language_clock.get(lang) if 'language_clock' in globals() else None
    _n2w_cache = {}
    return time_rx.sub(_repl_num, text)

def math2words(text:str, lang:str, lang_iso1:str, tts_engine:str, is_num2words_compat:bool)->str:

    def _repl_ambiguous(match:re.Match)->str:
        # handles "num SYMBOL num" and "SYMBOL num"
        if match.group(2) == '-' and range_word and not re.search(r'\s', match.group(0)):
            # a dash glued between two numbers is a range: pages 19-32, 2011-2012
            return f'{match.group(1)} {range_word} {match.group(3)}'
        if match.group(2) and match.group(2) in ambiguous_replacements:
            return f'{match.group(1)} {ambiguous_replacements[match.group(2)]} {match.group(3)}'
        if match.group(3) and match.group(3) in ambiguous_replacements:
            return f'{ambiguous_replacements[match.group(3)]} {match.group(4)}'
        return match.group(0)

    def _ordinal_to_words(m:re.Match)->str:
        n = int(m.group(1))
        if is_num2words_compat:
            try:
                return num2words(n, to='ordinal', lang=(lang_iso1 or 'en'))
            except Exception:
                pass
        # If num2words isn't available/compatible, keep original token as-is.
        return m.group(0)

    range_word = number_range_words.get(lang)
    # Matches any digits + optional space/NBSP + st/nd/rd/th, not glued into words.
    re_ordinal = re.compile(r'(?<!\w)(\d+)(?:\s|\u00A0)*(?:st|nd|rd|th)(?!\w)')
    # list numbering "1) item" only at the start of an item, not "(numéro 10) a commencé"
    text = re.sub(r'(?m)(^|[.:;]\s+)(\d{1,3})\)\s', r'\1\2 : ', text)
    text = re_ordinal.sub(_ordinal_to_words, text)
    # Symbol phonemes
    ambiguous_symbols = {"-", "/", "*", "x"}
    phonemes_list = language_math_phonemes.get(lang, language_math_phonemes[default_language_code])
    replacements = {k: v for k, v in phonemes_list.items() if not k.isdigit() and k not in [',', '.']}
    normal_replacements  = {k: v for k, v in replacements.items() if k not in ambiguous_symbols}
    ambiguous_replacements = {k: v for k, v in replacements.items() if k in ambiguous_symbols}
    # Replace unambiguous symbols everywhere
    if normal_replacements:
        sym_pat = r'(' + '|'.join(re.escape(k) for k in sorted(normal_replacements, key=len, reverse=True)) + r')'
        text = re.sub(sym_pat, lambda m: f' {normal_replacements[m.group(1)]} ', text)
    # Replace ambiguous symbols only in valid equation contexts
    if ambiguous_replacements:
        ambiguous_pattern = (
            r'(?<!\S)'                   # no non-space before
            r'(\d+)\s*([-/*x])\s*(\d+)'  # num SYMBOL num
            r'(?!\S)'                    # no non-space after
            r'|'                         # or
            r'(?<!\S)([-/*x])\s*(\d+)(?!\S)'  # SYMBOL num
        )
        text = re.sub(ambiguous_pattern, _repl_ambiguous, text)
    text = set_formatted_number(text, lang, lang_iso1, is_num2words_compat)
    return text

def roman2number(text:str)->str:

    def _is_valid_roman(s:str)->bool:
        return bool(valid_roman.fullmatch(s))

    def _to_int(s:str)->str:
        s = s.upper()
        i = 0
        result = 0
        while i < len(s):
            for roman, value in roman_numbers_tuples:
                if s[i:i + len(roman)] == roman:
                    result += value
                    i += len(roman)
                    break
            else:
                return s
        return str(result)

    def _repl_heading(m:re.Match)->str:
        roman = m.group(1)
        if not _is_valid_roman(roman):
            return m.group(0)
        return f"{_to_int(roman)}{m.group(2)}{m.group(3)}"

    def _repl_standalone(m:re.Match)->str:
        roman = m.group(1)
        if not _is_valid_roman(roman):
            return m.group(0)
        return f"{_to_int(roman)}{m.group(2)}"

    def _repl_word(m:re.Match)->str:
        roman = m.group(1)
        if not _is_valid_roman(roman):
            return m.group(0)
        return _to_int(roman)

    def _repl_chapter_single(m:re.Match)->str:
        word = m.group(1)
        roman = m.group(2)
        if not _is_valid_roman(roman):
            return m.group(0)
        return f'{word} {_to_int(roman)}'

    valid_roman = re.compile(
        r'^(?=.)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',
        re.IGNORECASE
    )
    chapter_words = sorted(
        {w for words in chapter_word_mapping.values() for w in words},
        key=len,
        reverse=True
    )
    # the chapter word is case-insensitive, the numeral is not: "partie d'une" must not
    # become "partie cinq cents'une"; a lowercase numeral ("interlude ii") is only read as
    # one with the small letters i, v, x and right before a punctuation or the end
    chapter_words_re = re.compile(
        rf'\b({"|".join(map(re.escape, chapter_words))})\s+((?-i:[IVXLCDM]+|[ivx]+(?=[\s.,;:!?…\[]|$)))(?![\p{{L}}\p{{N}}\'’])',
        re.IGNORECASE | re.UNICODE
    )
    text = re.sub(
        r'^(?:\s*)([IVXLCDM]+)([.-])(\s+)',
        _repl_heading,
        text,
        flags=re.MULTILINE
    )
    text = re.sub(
        r'^(?:\s*)([IVXLCDM]+)([.-])(?:\s*)$',
        _repl_standalone,
        text,
        flags=re.MULTILINE
    )
    text = chapter_words_re.sub(_repl_chapter_single, text)
    text = re.sub(
        r'(?<!\S)([IVXLCDM]{2,})(?!\S)',
        _repl_word,
        text
    )
    return text
    
def is_latin(s:str)->bool:
    return all((u'a' <= ch.lower() <= 'z') or ch.isdigit() or not ch.isalpha() for ch in s)

def foreign2latin(text:str, base_lang:str)->str:

    def _script_of(word:str)->str:
        for ch in word:
            if ch.isalpha():
                name = unicodedata.name(ch, '')
                if 'CYRILLIC' in name:
                    return 'cyrillic'
                if 'LATIN' in name:
                    return 'latin'
                if 'ARABIC' in name:
                    return 'arabic'
                if 'HANGUL' in name:
                    return 'hangul'
                if 'HIRAGANA' in name or 'KATAKANA' in name:
                    return 'japanese'
                if 'CJK' in name or 'IDEOGRAPH' in name:
                    return 'chinese'
        return 'unknown'

    def _romanize(word:str)->str:
        scr = _script_of(word)
        if scr == 'latin':
            return word
        try:
            if scr == 'chinese':
                return ''.join(x[0] for x in pinyin(word, style=Style.NORMAL))
            if scr == 'japanese':
                k = pykakasi.kakasi()
                k.setMode('H', 'a')
                k.setMode('K', 'a')
                k.setMode('J', 'a')
                k.setMode('r', 'Hepburn')
                return k.getConverter().do(word)
            if scr == 'hangul':
                return unidecode(word)
            if scr == 'arabic':
                return unidecode(phonemize(word, language='ar', backend='espeak'))
            if scr == 'cyrillic':
                return unidecode(phonemize(word, language='ru', backend='espeak'))
            return unidecode(word)
        except Exception:
            return unidecode(word)

    # Protect ALL SML tags using the global grammar
    protected:dict[str, str] = {}
    for i, m in enumerate(SML_TAG_PATTERN.finditer(text)):
        key:str = f'__TTS_MARKER_{i}__'
        protected[key] = m.group(0)
        text = text.replace(m.group(0), key)
    matches = list(re.finditer(r"\w+|[^\w\s]", text, re.UNICODE))
    tokens:list[str] = [m.group(0) for m in matches]
    # whether the source had no whitespace right before each token
    glued:list[bool] = [i > 0 and m.start() == matches[i - 1].end() for i, m in enumerate(matches)]
    buf:list[str] = []
    for t in tokens:
        if t in protected:
            buf.append(t)
        elif re.match(r"^\w+$", t):
            buf.append(_romanize(t))
        else:
            buf.append(t)
    out: str = ''
    joiners = ("'", "\u2018", "\u2019")  # stay glued to the next word (contractions)
    # a hyphen stays glued to the next word only when it was in the source
    # ("dit-il", "peut-\u00eatre"), so a spaced dialogue dash keeps its space
    hyphens = ('-', '\u2010', '\u2011')
    for i, t in enumerate(buf):
        if i == 0:
            out += t
        else:
            prev = buf[i - 1]
            prev_is_word = bool(re.match(r"^\w+$", prev))
            curr_is_word = bool(re.match(r"^\w+$", t))
            prev_is_protected = prev in protected
            curr_is_protected = t in protected
            if prev_is_word and curr_is_word:
                out += ' ' + t
            elif prev_is_protected or curr_is_protected:
                out += ' ' + t
            elif prev in hyphens and glued[i]:
                out += t
            elif not prev_is_word and not prev_is_protected and curr_is_word and prev not in joiners:
                out += ' ' + t
            else:
                out += t
    for k, v in protected.items():
        out = out.replace(k, v)
    return out

def normalize_sml_tags(text:str)->tuple[bool, str]:
    out = []
    stack = []
    last = 0
    for m in SML_TAG_PATTERN.finditer(text):
        start, end = m.span()
        out.append(text[last:start])
        tag = m.group("tag")
        close = bool(m.group("close"))
        value = m.group("value")
        info = TTS_SML.get(tag)
        if not info:
            out.append(m.group(0))
            last = end
            continue
        if info.get("paired"):
            if close:
                if not stack or stack[-1] != tag:
                    error = f'normalize_sml_tags() error: unmatched closing tag [/{tag}]'
                    return False, error
                stack.pop()
                out.append(f"[/{tag}]")
            else:
                stack.append(tag)
                if value is not None:
                    out.append(f"[{tag}:{value.strip()}]")
                else:
                    error = f'normalize_sml_tags() error: paired tag [{tag}] requires a value'
                    return False, error
        else:
            if close:
                error = f'normalize_sml_tags() error: non-paired tag [/{tag}] is invalid'
                return False, error
            out.append(info['static'])
        last = end
    out.append(text[last:])
    if stack:
        error = f"normalize_sml_tags() error: unclosed tag(s): {', '.join(stack)}"
        return False, error
    return True, ''.join(out)

def escape_sml(text:str)->tuple[str, list[str]]:
    sml_blocks:list[str] = []

    def _replace(m:re.Match[str])->str:
        sml_blocks.append(m.group(0))
        return chr(sml_escape_tag + len(sml_blocks) - 1)

    return SML_TAG_PATTERN.sub(_replace, text), sml_blocks

def restore_sml(text:str, sml_blocks:list[str])->str:
    for i, block in enumerate(sml_blocks):
        text = text.replace(chr(sml_escape_tag + i), block)
    return text

def sml_token(tag:str, value:str|None=None, close:bool=False)->str:
    if close:
        return f"[/{tag}]"
    if value is not None:
        return f"[{tag}:{value}]"
    return f"[{tag}]"

def normalize_text(text:str, lang:str, lang_iso1:str, tts_engine:str)->str:

    def _replace(match:re.Match)->str:
        token = match.group(1)
        for k, expansion in mapping.items():
            if token.lower() == k.lower():
                # "etc." closing a sentence: the period also ends the sentence, keep it
                if k.endswith('.') and re.match(r'\s*(?:$|[\p{Lu}«"“\[])', match.string[match.end():]):
                    return expansion + '.'
                return expansion
        return token  # fallback

    def _reduce_hard(match:re.Match)->str:
        # keep the most expressive mark of a run: "?." must stay a question
        run = match.group(0)
        for mark in ('?', '!', '…'):
            if mark in run:
                return mark + ' '
        return match.group(2) + ' '
            
    # Remove emojis
    emoji_pattern = re.compile(f"[{''.join(emojis_list)}]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    if lang in abbreviations_mapping:
        mapping = abbreviations_mapping[lang]
        # Sort keys by descending length so longer ones match first
        keys = sorted(mapping.keys(), key=len, reverse=True)
        # Build a regex that only matches whole “words” (tokens) exactly
        pattern = re.compile(
            r'(?<!\w)(' + '|'.join(re.escape(k) for k in keys) + r')(?!\w)',
            flags=re.IGNORECASE
        )
        text = pattern.sub(_replace, text)
    text = re.sub(r'\bO\.\s?K\.', 'Okay', text)
    # This regex matches sequences like a., c.i.a., f.d.a., m.c., etc…
    pattern = re.compile(r'\b(?:[a-zA-Z]\.){1,}[a-zA-Z]?\b\.?')
    # uppercase acronyms
    text = re.sub(r'\b(?:[a-zA-Z]\.){1,}[a-zA-Z]?\b\.?', lambda m: m.group().replace('.', '').upper(), text)
    # romanize foreign words
    if language_mapping[lang]['script'] == 'latin':
        text = foreign2latin(text, lang)
    # Replace multiple newlines ("\n\n", "\r\r", "\n\r", etc.) with a [pause] 1.4sec
    pattern = r'(?:\r\n|\r|\n){2,}'
    text = re.sub(pattern, f" {sml_token('pause')} ", text)
    # Replace single newlines ("\n" or "\r") with spaces
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    # the '...' key of punctuation_switch cannot match inside a character class
    text = re.sub(r'\.{3,}', '…', text)
    # Replace punctuations causing hallucinations
    pattern = f"[{''.join(map(re.escape, punctuation_switch.keys()))}]"
    text = re.sub(pattern, lambda match: punctuation_switch.get(match.group(), match.group()), text)
    # a soft punctuation right after a sentence end is an artefact of the switches above
    # ("(…celle-ci.) Cela" -> "celle-ci., Cela"), it would add a pause without meaning
    text = re.sub(r'([.!?…])(?:\s*[,;:])+', r'\1', text)
    # and a soft one right before it: ", ." left by an incise closing the sentence
    text = re.sub(r'(?:[,;:]\s*)+([.!?…])', r'\1', text)
    # remove unwanted chars
    chars_remove_table = str.maketrans({ch: ' ' for ch in chars_remove})
    text = text.translate(chars_remove_table)
    # quotes are typography: a quoted word must not get pauses around it
    # ("le mot «minotaure»" read "le mot, minotaure,")
    text = re.sub(r'"', ' ', text)
    # Replace multiple and spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Replace ok by 'Owkey'
    text = re.sub(r'\bok\b', 'Okay', text, flags=re.IGNORECASE)
    # Escape special characters in the punctuation list for regex
    pattern = '|'.join(map(re.escape, punctuation_split_hard_set))
    # Reduce multiple consecutive punctuations hard
    text = re.sub(rf'(\s*({pattern})\s*)+', _reduce_hard, text).strip()
    # Escape special characters in the punctuation list for regex
    pattern = '|'.join(map(re.escape, punctuation_split_soft_set))
    # Reduce multiple consecutive punctuations soft
    text = re.sub(rf'(\s*({pattern})\s*)+', r'\2 ', text).strip()
    # ordinals written inline ("1er", "2e", "21st") before digits and letters get split
    text = speak_inline_ordinals(text, lang)
    # Pattern 1: Add a space between UTF-8 characters and numbers
    text = re.sub(r'(?<=[\p{L}])(?=\d)|(?<=\d)(?=[\p{L}])', ' ', text)
    # Replace special chars with words
    specialchars = specialchars_mapping.get(lang, specialchars_mapping.get(default_language_code, specialchars_mapping['eng']))
    specialchars_table = {ord(char): f" {word} " for char, word in specialchars.items()}
    text = text.translate(specialchars_table)
    text = ' '.join(text.split())
    return text

def block_hash(block: dict) -> str:
    return hashlib.sha1(
        '|'.join((
            (block.get('text') or '').strip(),
            # the voice is identified by its file name: the session works on a copy of the
            # voice, a different folder for the same voice must not reconvert the chapter
            os.path.basename(block.get('voice') or ''),
            block.get('tts_engine') or TTS_ENGINES['XTTS'],
            block.get('fine_tuned') or 'internal',
            json.dumps(block.get('sentences') or [], ensure_ascii=False),
        )).encode('utf-8')
    ).hexdigest()

def text_hash(text:str)->str:
    # text-only hash, whitespace-insensitive: used to align blocks between two parses of the source file.
    # block_hash() cannot be used here since a fresh parse has empty sentences and default voice/engine.
    return hashlib.sha1(' '.join((text or '').split()).encode('utf-8')).hexdigest()

def purge_block_audio(session:dict, block_id:str)->None:
    try:
        ch_file = os.path.join(session['chapters_dir'], f'{block_id}.{default_audio_proc_format}')
        if os.path.exists(ch_file):
            os.unlink(ch_file)
        block_dir = os.path.join(session['sentences_dir'], block_id)
        if os.path.isdir(block_dir):
            shutil.rmtree(block_dir)
    except Exception as e:
        error = f'purge_block_audio() error: {e}'
        print(error)

def align_blocks(old_blocks:list, new_blocks:list, old_current:list, old_saved:list, ratio_min:float=0.6)->dict:
    # pure function: no session, no disk, no globals. unit testable with plain lists.
    # aligns a new parse of the source file against the previous parse so unchanged blocks
    # inherit their old id (and therefore keep their audio, which is keyed by id).
    # returns dict(merged_current, merged_saved, removed_ids, equal_ids, kept, changed, added).
    current_by_id = {b['id']: b for b in old_current if b.get('id')}
    saved_by_id = {b['id']: b for b in old_saved if b.get('id')}
    old_hashes = [text_hash(b.get('text', '')) for b in old_blocks]
    new_hashes = [text_hash(b.get('text', '')) for b in new_blocks]
    matcher = difflib.SequenceMatcher(None, old_hashes, new_hashes, autojunk=False)
    merged_current = []
    merged_saved = []
    matched_ids = set()
    equal_ids = set()
    n_kept = n_changed = n_added = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(j2 - j1):
                old_b = old_blocks[i1 + k]
                new_b = new_blocks[j1 + k]
                old_id = old_b['id']
                new_b['id'] = old_id
                matched_ids.add(old_id)
                equal_ids.add(old_id)
                cur_b = current_by_id.get(old_id)
                merged_current.append(copy.deepcopy(cur_b) if cur_b else copy.deepcopy(new_b))
                sav_b = saved_by_id.get(old_id)
                if sav_b:
                    merged_saved.append(copy.deepcopy(sav_b))
                n_kept += 1
        elif tag in ('replace', 'insert'):
            pairs = min(i2 - i1, j2 - j1) if tag == 'replace' else 0
            for k in range(j2 - j1):
                new_b = new_blocks[j1 + k]
                if k < pairs:
                    old_b = old_blocks[i1 + k]
                    ratio = difflib.SequenceMatcher(None, old_b.get('text', ''), new_b.get('text', ''), autojunk=False).ratio()
                    if ratio >= ratio_min:
                        # same block, edited: inherit id and per-block settings, force reconversion
                        old_id = old_b['id']
                        new_b['id'] = old_id
                        matched_ids.add(old_id)
                        cur_b = current_by_id.get(old_id)
                        block = copy.deepcopy(cur_b) if cur_b else copy.deepcopy(new_b)
                        block['id'] = old_id
                        block['text'] = new_b.get('text', '')
                        block['sentences'] = []
                        merged_current.append(block)
                        sav_b = saved_by_id.get(old_id)
                        if sav_b:
                            merged_saved.append(copy.deepcopy(sav_b))
                        n_changed += 1
                        continue
                merged_current.append(copy.deepcopy(new_b))
                n_added += 1
    removed_ids = [b['id'] for b in old_blocks if b.get('id') and b['id'] not in matched_ids]
    return dict(
        merged_current=merged_current,
        merged_saved=merged_saved,
        removed_ids=removed_ids,
        equal_ids=equal_ids,
        kept=n_kept,
        changed=n_changed,
        added=n_added,
    )

def remap_resume(old_current:list, merged_current:list, equal_ids:set, block_resume:int, sentence_resume:int)->tuple:
    # pure function: block_resume/sentence_resume are positional, so they must follow the block
    # they pointed at into the new order. if that block was edited or deleted, sentence_resume resets.
    if not old_current or not merged_current:
        return 0, 0
    new_index = {b['id']: i for i, b in enumerate(merged_current)}
    pos = min(block_resume, len(old_current) - 1)
    while pos >= 0:
        old_id = old_current[pos].get('id')
        if old_id in new_index:
            if pos == block_resume and old_id in equal_ids:
                return new_index[old_id], sentence_resume
            return new_index[old_id], 0
        pos -= 1
    return 0, 0

def realign_blocks(session_id:str, blocks_orig_old:dict)->bool:
    # thin i/o wrapper around align_blocks(): reads the session, purges the audio of deleted
    # blocks, remaps the resume pointer and persists. all decision logic lives in align_blocks().
    try:
        session = context.get_session(session_id)
        if not (session and session.get('id', False)):
            return False
        blocks_orig = session['blocks_orig']
        old_blocks = blocks_orig_old.get('blocks', [])
        new_blocks = blocks_orig.get('blocks', [])
        blocks_current = session['blocks_current']
        old_current = blocks_current.get('blocks', []) if blocks_current else []
        if not old_blocks or not new_blocks or not old_current:
            return False
        blocks_saved = session['blocks_saved']
        old_saved = blocks_saved.get('blocks', []) if blocks_saved else []
        result = align_blocks(old_blocks, new_blocks, old_current, old_saved)
        for removed_id in result['removed_ids']:
            purge_block_audio(session, removed_id)
        new_block_resume, new_sentence_resume = remap_resume(
            old_current, result['merged_current'], result['equal_ids'],
            blocks_current.get('block_resume', 0) or 0,
            blocks_current.get('sentence_resume', 0) or 0,
        )
        blocks_current['blocks'] = result['merged_current']
        blocks_current['block_resume'] = new_block_resume
        blocks_current['sentence_resume'] = new_sentence_resume
        session['blocks_current'] = blocks_current
        save_db_blocks(session_id)
        if blocks_saved:
            blocks_saved['blocks'] = result['merged_saved']
            session['blocks_saved'] = blocks_saved
            save_json_blocks(session_id, 'blocks_saved')
        session['blocks_orig'] = blocks_orig
        msg = (f"Source file changed: {result['kept']} unchanged, {result['changed']} modified, "
               f"{result['added']} added, {len(result['removed_ids'])} removed. "
               f'Only modified and added blocks will be reconverted.')
        show_alert(session_id, {'type': 'info', 'msg': msg})
        return True
    except Exception as e:
        exception_alert(session_id, f'realign_blocks() error: {e}')
        return False

def convert_chapters2audio(session_id:str)->bool:

    def _reset_chapter_file(block_id:str)->None:
        ch_file = os.path.join(session['chapters_dir'], f'{block_id}.{default_audio_proc_format}')
        if os.path.exists(ch_file):
            os.unlink(ch_file)
        block_dir = os.path.join(session['sentences_dir'], block_id)
        if os.path.isdir(block_dir):
            shutil.rmtree(block_dir)

    def _check_block_sentences(block_id:str, sentences:list)->set:
        block_dir = os.path.join(session['sentences_dir'], block_id)
        missing = set()
        for j in valid_idx:
            is_sml = bool(SML_TAG_PATTERN.fullmatch(sentences[j]))
            if (not is_sml) or (j == last_idx):
                sentence_file = os.path.join(block_dir, f'{j}.{default_audio_proc_format}')
                if not os.path.exists(sentence_file):
                    missing.add(j)
        return missing

    def _count_sentences(sentences:list)->int:
        return sum(1 for s in sentences if any(c.isalnum() for c in s.strip()))

    def _asr_chunks(block_id:str, sentences:list)->list[dict]:
        block_dir = os.path.join(session['sentences_dir'], block_id)
        return [
            {'j': j, 'wav': os.path.join(block_dir, f'{j}.{default_audio_proc_format}'), 'text': ' '.join(SML_TAG_PATTERN.sub(' ', s).split())}
            for j, s in enumerate(sentences) if any(c.isalnum() for c in s)
        ]

    def _valid_take(path:str)->bool:
        try:
            import soundfile as sf
            return sf.info(path).frames > 0
        except Exception:
            return False

    def _variant(wav:str, tag:str)->str:
        # "<j>.flac" -> "<j>.<tag>.flac": takes live next to the chunk, ignored by the concat
        return wav[:-len(default_audio_proc_format) - 1] + f'.{tag}.{default_audio_proc_format}'

    def _cadence_retakes(ch_num:int, j:int, kind:str, wav:str, sentence:str, block_voice:str|None)->None:
        # titles and chapter ends must close on a falling tune: when the take does not fall,
        # up to cadence.MAX_TAKES more takes are drawn and the most falling one is kept.
        # Every take stays on disk (".cadK") until the part correction, where Whisper may
        # need them (a title must also be read right)
        c = cadence.measure(wav)
        entry = {'wav': wav, 'ch_num': ch_num, 'j': j, 'kind': kind, 'before': c}
        # an unmeasurable cadence (one or two words) is left alone: new takes of a single
        # word are where XTTS hallucinates the most ("Chaussures." -> "C'est un homme")
        if c is None or cadence.falls(c):
            entry.update(after=c, take=0, takes=0)
            asr_jsonl_append(cadence_log, entry)
            return
        shutil.copyfile(wav, _variant(wav, 'cad0'))
        best = (cadence.rank(c), 0, c)
        drawn = 0
        for k in range(1, cadence.MAX_TAKES + 1):
            take = _variant(wav, f'cad{k}')
            if not _valid_take(take):
                tts_manager.engine.params['seed_offset'] = 104729 * k
                ok, err = tts_manager.convert_sentence2audio(take, sentence, block_voice=block_voice)
                tts_manager.engine.params['seed_offset'] = 0
                if not ok:
                    continue
            drawn = k
            ck = cadence.measure(take)
            if cadence.rank(ck) < best[0]:
                best = (cadence.rank(ck), k, ck)
            if cadence.falls(ck):
                break
        if best[1]:
            shutil.copyfile(_variant(wav, f'cad{best[1]}'), wav)
        entry.update(after=best[2], take=best[1], takes=drawn)
        asr_jsonl_append(cadence_log, entry)
        print(f"[cadence] ch {ch_num} {kind} troncon {j} : ne descendait pas {c} -> "
              f"{'essai ' + str(best[1]) if best[1] else 'original garde'} sur {drawn}, "
              f"{'DESCEND' if cadence.falls(best[2]) else 'ne descend toujours pas'} {best[2]}", flush=True)

    def _gen_takes(items:list[dict], ks:range, k_part:int, label:str)->list[dict]:
        # repair takes with new seeds; takes already on disk (interrupted run) are reused
        tts = None
        takes = []
        for i, b in enumerate(items, 1):
            _, block = by_id[b['block_id']]
            sentence = block['sentences'][b['j']].strip()
            block_voice = block.get('voice') or session.get('voice')
            for k in ks:
                take = _variant(b['wav'], f'take{k}')
                if not _valid_take(take):
                    if tts is None:
                        tts = TTSManager(session)
                    tts.engine.params['seed_offset'] = 7919 * k
                    ok, err = tts.convert_sentence2audio(take, sentence, block_voice=block_voice)
                    if not ok:
                        continue
                takes.append({'wav': take, 'text': b['text'], 'orig': b['wav'], 'stage': label})
            print(f'[verif] partie {k_part} {label} {i}/{len(items)} : {len(ks)} essais prets', flush=True)
        if tts is not None:
            tts.engine.params['seed_offset'] = 0
            unload_tts_manager(tts)
        return takes

    def _repair(bad:list[dict], k_part:int, again:set, control:str|None=None)->tuple[int, set]:
        """Repairs the confirmed bad chunks of one part. Returns (repaired count, touched chapters)."""
        attempted = {r.get('wav') for r in asr_verifier.repairs()}
        if again and control:
            # a control added after the part was corrected: new seeds, away from the repair
            # (1..3) and second chance (4..11) ones
            bad = [b for b in bad if (b['ch_num'], b['j']) in again]
            ks = range(40, 40 + AsrVerifier.REPAIR_TAKES + 1)
            print(f'[verif] controle {control} : {len(bad)} troncon(s) a reprendre, {len(ks)} essais chacun', flush=True)
        elif again:
            bad = [b for b in bad if (b['ch_num'], b['j']) in again]
            ks = range(AsrVerifier.REPAIR_TAKES + 1, AsrVerifier.REPAIR_TAKES + 1 + again_takes)
            print(f'[verif] seconde chance manuelle : {len(bad)} troncon(s), {again_takes} nouveaux essais chacun', flush=True)
        else:
            # a chunk already through a repair keeps its best take, the deterministic seeds
            # would only draw the same takes again
            bad = [b for b in bad if b['wav'] not in attempted]
            ks = range(1, AsrVerifier.REPAIR_TAKES + 1)
        if not bad:
            return 0, set()
        kinds = {}
        for b in bad:
            _, block = by_id[b['block_id']]
            kinds[b['wav']] = cadence.cadence_targets(block['sentences']).get(b['j'])
        takes = _gen_takes(bad, ks, k_part, 'reparation')
        # the current take is a candidate too: cutting its babbling tail may be the repair
        for b in bad:
            orig_copy = _variant(b['wav'], 'take0')
            shutil.copyfile(b['wav'], orig_copy)
            takes.append({'wav': orig_copy, 'text': b['text'], 'orig': b['wav'], 'stage': 'original'})
        # titles / chapter ends: the cadence takes drawn during the synthesis are candidates too
        for b in bad:
            if kinds[b['wav']]:
                for k in range(0, cadence.MAX_TAKES + 1):
                    cad = _variant(b['wav'], f'cad{k}')
                    if _valid_take(cad):
                        takes.append({'wav': cad, 'text': b['text'], 'orig': b['wav'], 'stage': 'cadence'})
        asr_verifier.score_takes(takes)
        second = set()
        if not again:
            # automatic second chance: still far from the text after the first takes
            def _best_cer(b):
                return min((t['cer'] for t in takes if t['orig'] == b['wav'] and 'cer' in t), default=1.0)
            late = [b for b in bad if _best_cer(b) > AsrVerifier.SECOND_CHANCE_CER]
            if late:
                print(f'[verif] partie {k_part} : seconde chance automatique pour {len(late)} troncon(s), '
                      f'{AsrVerifier.SECOND_CHANCE_TAKES} essais chacun', flush=True)
                ks2 = range(AsrVerifier.REPAIR_TAKES + 1, AsrVerifier.REPAIR_TAKES + 1 + AsrVerifier.SECOND_CHANCE_TAKES)
                takes2 = _gen_takes(late, ks2, k_part, 'seconde chance')
                asr_verifier.score_takes(takes2)
                takes += takes2
                second = {b['wav'] for b in late}
        repaired = 0
        touched = set()
        for b in bad:
            cands = [t for t in takes if t['orig'] == b['wav'] and 'cer' in t]
            kind = kinds[b['wav']]
            if kind:
                for t in cands:
                    t['cadence'] = cadence.measure(t['wav'])
                best = cadence.choose_take(cands, max(asr_confirm_cer, cadence.ACCEPT_CER))
            else:
                best = min(cands, key=lambda t: t['cer']) if cands else None
            fixed = best is not None and best['cer'] < b['cer']
            hyps2 = [t['asr'] for t in cands if t['stage'] == 'seconde chance']
            probable_fp = b['wav'] in second and same_transcription(hyps2) and not (fixed and best['cer'] <= asr_confirm_cer)
            if fixed:
                if os.path.abspath(best['wav']) != os.path.abspath(b['wav']):
                    shutil.copyfile(best['wav'], b['wav'])
                repaired += 1
                touched.add(b['ch_num'])
                asr_verifier.record_result(b['wav'], best['cer'], best['asr'])
            if again or b['wav'] in second:
                print(f"[verif] seconde chance ch {b['ch_num']} troncon {b['j']} : CER {b['cer']:.2f} -> "
                      f"{best['cer'] if best else float('nan'):.2f} " + ('REPARE' if fixed else 'inchange')
                      + (' (probable faux positif de Whisper)' if probable_fp else '') + f" | {best['asr'] if best else ''}", flush=True)
            asr_verifier.record_repair({
                'wav': b['wav'], 'ch_num': b['ch_num'], 'j': b['j'], 'text': b['text'], 'before_cer': b['cer'], 'before_asr': b['asr'],
                'after_cer': best['cer'] if best else None, 'after_asr': best['asr'] if best else None, 'fixed': fixed,
                'second_chance': b['wav'] in second or bool(again), 'probable_false_positive': probable_fp,
                'kind': kind, 'cadence': best.get('cadence') if best else None, 'part': k_part
            })
            for t in cands:
                if t['stage'] != 'cadence' and os.path.exists(t['wav']):
                    os.remove(t['wav'])
        return repaired, touched

    def _cadence_whisper_pass(part:list[int], k_part:int)->None:
        # titles / chapter ends retaken for their cadence during the synthesis: every take is
        # transcribed and the kept one is the most falling of those Whisper accepts
        groups = []
        for ch in part:
            block = kept[ch - 1]
            for j, kind in cadence.cadence_targets(block['sentences']).items():
                wav = os.path.join(session['sentences_dir'], block['id'], f'{j}.{default_audio_proc_format}')
                cads = [_variant(wav, f'cad{k}') for k in range(0, cadence.MAX_TAKES + 1) if _valid_take(_variant(wav, f'cad{k}'))]
                if cads:
                    text = ' '.join(SML_TAG_PATTERN.sub(' ', block['sentences'][j]).split())
                    groups.append((ch, j, kind, wav, [{'wav': c, 'text': text, 'orig': wav} for c in cads]))
        if not groups:
            return
        print(f'[cadence] partie {k_part} : Whisper sur les prises de {len(groups)} titre(s)/fin(s) reprises pour l\'intonation', flush=True)
        asr_verifier.score_takes([t for g in groups for t in g[4]])
        for ch, j, kind, wav, cands in groups:
            for t in cands:
                t['cadence'] = cadence.measure(t['wav'])
            best = cadence.choose_take(cands, max(asr_confirm_cer, cadence.ACCEPT_CER))
            if best is None:
                continue
            shutil.copyfile(best['wav'], wav)
            asr_verifier.record_result(wav, best['cer'], best['asr'])
            print(f"[cadence] ch {ch} {kind} troncon {j} : prise {os.path.basename(best['wav'])} CER {best['cer']:.3f} "
                  f"{'DESCEND' if cadence.falls(best['cadence']) else 'ne descend pas'}", flush=True)

    def _parts_state_save()->None:
        with open(parts_state_path, 'w', encoding='utf-8') as f:
            json.dump(parts_state, f, ensure_ascii=False, indent=1)

    def _delivery_name()->str:
        title = os.environ.get('E2A_DELIVERY_NAME') or (session.get('metadata') or {}).get('title') or Path(session.get('final_name') or 'audiobook').stem
        title = unicodedata.normalize('NFKD', title)
        title = ''.join(c for c in title if not unicodedata.combining(c))
        return ''.join('_' if c in '<>:"/\\|?*' else c for c in title).strip(' .')

    def _deliver(path:str)->str|None:
        # copy of a finished file to the delivery folder (E2A_DELIVERY_DIR), if any
        dest_dir = os.environ.get('E2A_DELIVERY_DIR')
        if not (dest_dir and path and os.path.exists(path)):
            return None
        try:
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(path))
            shutil.copyfile(path, dest)
            print(f'[livraison] {dest}', flush=True)
            return dest
        except OSError as e:
            print(f'[livraison] copie impossible de {path} : {e}')
            return None

    def _correct_part(k_part:int, again:set|None=None, control:str|None=None)->bool:
        """Correction of one part of the book, as soon as its last chapter is synthesised:
        Whisper check of its chunks, repairs (+ automatic second chance), chapters rebuilt
        with their silences, progressive export, one chaptered m4b of the part delivered."""
        nonlocal tts_manager
        part = parts[k_part - 1]
        again = again or set()
        t0 = time.time()
        print(f"\n[verif] ===== CORRECTION DE LA PARTIE {k_part}/{len(parts)} : chapitres {part[0]} a {part[-1]} =====", flush=True)
        show_alert(session_id, {'type': 'info', 'msg': f'Part {k_part}/{len(parts)}: Whisper check and repairs of chapters {part[0]} to {part[-1]}…'})
        if asr_verifier:
            asr_verifier.stop_watcher()
        unload_tts_manager(tts_manager)
        tts_manager = None
        repaired, touched, n_bad = 0, set(), 0
        if asr_verifier and not again:
            _cadence_whisper_pass(part, k_part)
        if asr_verifier:
            part_ids = {kept[ch - 1]['id'] for ch in part}
            bad = asr_verifier.verify_on_gpu(block_ids=part_ids)
            for b in bad:
                b['ch_num'] = by_id[b['block_id']][0]
            n_bad = len(bad)
            print(f'[verif] partie {k_part} : {n_bad} troncon(s) confirme(s) au-dessus de {asr_confirm_cer * 100:.0f} % de CER', flush=True)
            repaired, touched = _repair(bad, k_part, again, control)
        # final cadence of the titles / chapter ends of the part, for the report
        cad_final = []
        for ch in part:
            block = kept[ch - 1]
            for j, kind in cadence.cadence_targets(block['sentences']).items():
                wav = os.path.join(session['sentences_dir'], block['id'], f'{j}.{default_audio_proc_format}')
                c = cadence.measure(wav)
                cad_final.append({'ch_num': ch, 'j': j, 'kind': kind, 'falls': cadence.falls(c), 'cadence': c,
                                  'text': ' '.join(SML_TAG_PATTERN.sub(' ', block['sentences'][j]).split())})
                for k in range(0, cadence.MAX_TAKES + 1):
                    cad = _variant(wav, f'cad{k}')
                    if os.path.exists(cad):
                        os.remove(cad)
        # every chapter of the part is rebuilt (repairs, chapter silences) and exported
        for n, ch in enumerate(part, 1):
            if again and ch not in touched:
                continue
            block = kept[ch - 1]
            ch_file = os.path.join(session['chapters_dir'], f"{block['id']}.{default_audio_proc_format}")
            if not combine_audio_sentences(session_id, ch_file, block['id'], len(block['sentences'])):
                return False
            if exporter:
                exporter.add_chapter(ch, ch_file, chapter_title_of(block['sentences'], ch))
            print(f'[verif] partie {k_part} reconstruction {n}/{len(part)}', flush=True)
        m4b = None
        if exporter:
            name = f'{_delivery_name()} - partie {k_part} sur {len(parts)} (chapitres {part[0]} a {part[-1]}).m4b'
            title = f"{exporter.book_title} — partie {k_part} sur {len(parts)} (chapitres {part[0]} à {part[-1]})"
            m4b = exporter.export_range(part[0], part[-1], os.path.join(session['audiobooks_dir'], f'{exporter.stem} (parties)', name), title)
        delivered = _deliver(m4b)
        not_falling = [c for c in cad_final if not c['falls']]
        if again:
            done = parts_state['done'].setdefault(str(k_part), {'chapters': [part[0], part[-1]], 'bad': 0, 'repaired': 0})
            if control:
                done.setdefault('controls', ['cer', 'cadence'])
                if control not in done['controls']:
                    done['controls'].append(control)
                done[f'{control}_repaired'] = done.get(f'{control}_repaired', 0) + repaired
            else:
                done['again_repaired'] = done.get('again_repaired', 0) + repaired
            done.update(m4b=delivered or m4b, cadence_targets=len(cad_final),
                        cadence_not_falling=[[c['ch_num'], c['j'], c['kind'], c['text'][:80]] for c in not_falling])
            _parts_state_save()
            _deliver(_write_verification_report())
            if control:
                print(f'[verif] === CONTROLE {control.upper()} PARTIE {k_part}/{len(parts)} : {repaired}/{len(again)} troncon(s) repare(s) -> {delivered or m4b} ===', flush=True)
                if asr_verifier and k_part < len(parts):
                    asr_verifier.start_watcher()
            else:
                print(f'[verif] seconde chance partie {k_part} : {repaired} troncon(s) repare(s)', flush=True)
            return True
        parts_state['done'][str(k_part)] = {
            'chapters': [part[0], part[-1]], 'bad': n_bad, 'repaired': repaired, 'm4b': delivered or m4b, 'controls': list(asr_controls),
            'cadence_targets': len(cad_final), 'cadence_not_falling': [[c['ch_num'], c['j'], c['kind'], c['text'][:80]] for c in not_falling],
            'minutes': round((time.time() - t0) / 60, 1), 'finished': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        _parts_state_save()
        _deliver(_write_verification_report())
        print(f"[verif] === PARTIE {k_part}/{len(parts)} TERMINEE : chapitres {part[0]} a {part[-1]}, {repaired}/{n_bad} troncon(s) repare(s), "
              f"{len(cad_final) - len(not_falling)}/{len(cad_final)} titres/fins descendants, {(time.time() - t0) / 60:.1f} min -> {delivered or m4b} ===", flush=True)
        if asr_verifier and k_part < len(parts) and not again:
            asr_verifier.start_watcher()
        return True

    def _control_done_part(k_part:int)->None:
        done = parts_state['done'][str(k_part)]
        missing = [c for c in asr_controls if c not in done.get('controls', ['cer', 'cadence'])]
        if not (asr_verifier and 'point' in missing):
            return
        part = parts[k_part - 1]
        targets = asr_verifier.spoken_punct_chunks({kept[ch - 1]['id'] for ch in part})
        print(f'\n[verif] ===== CONTROLE POINT PARTIE {k_part}/{len(parts)} : chapitres {part[0]} a {part[-1]}, '
              f'{len(targets)} troncon(s) ou un "point" est entendu =====', flush=True)
        if targets:
            _correct_part(k_part, targets, control='point')
        else:
            done.setdefault('controls', ['cer', 'cadence']).append('point')
            _parts_state_save()

    def _after_chapter(ch_num:int)->None:
        k_part = part_of_last.get(ch_num)
        if not k_part:
            return
        if str(k_part) in parts_state['done']:
            try:
                _control_done_part(k_part)
            except Exception as e:
                DependencyError(e)
                print(f'[verif] controle de la partie {k_part} interrompu par une erreur : {e}')
            return
        # the correction improves the book, it must never cost the hours of synthesis
        try:
            if not _correct_part(k_part):
                print(f'[verif] partie {k_part} : un chapitre repare n\'a pas pu etre recombine, chapitres originaux gardes')
        except Exception as e:
            DependencyError(e)
            print(f'[verif] correction de la partie {k_part} interrompue par une erreur : {e}')

    def _write_verification_report()->str|None:
        if not asr_verifier:
            return None
        try:
            stem = Path(session.get('final_name') or 'audiobook').stem
            path = os.path.join(session['audiobooks_dir'], f'{stem} - verification Whisper.txt')
            # one entry per chunk: first verdict before any repair, last attempt after
            merged = {}
            for r in asr_verifier.repairs():
                key = r.get('wav') or (r['ch_num'], r['j'])
                if key in merged:
                    first = merged[key]
                    r = dict(r, before_asr=first['before_asr'], before_cer=first['before_cer'], fixed=first['fixed'] or r['fixed'],
                             after_asr=r['after_asr'] if r['fixed'] or not first['fixed'] else first['after_asr'],
                             after_cer=r['after_cer'] if r['fixed'] or not first['fixed'] else first['after_cer'])
                merged[key] = r
            entries = sorted(merged.values(), key=lambda r: (r['ch_num'], r['j']))
            fps = [r for r in entries if r.get('probable_false_positive')]
            remaining = [r for r in entries if (r.get('after_cer') if r.get('fixed') else r['before_cer']) > asr_confirm_cer and not r.get('probable_false_positive')]
            done = parts_state['done']
            lines = [f"Verification Whisper : {len(entries)} troncon(s) faux, {sum(1 for r in entries if r['fixed'])} repare(s), "
                     f"{len(fps)} probable(s) faux positif(s) de Whisper, {len(remaining)} defaut(s) restant(s) a ecouter",
                     f"Parties corrigees : {', '.join(sorted(done, key=int)) or 'aucune'} sur {len(parts)}", '']
            for k in sorted(done, key=int):
                d = done[k]
                lines.append(f"Partie {k} (chapitres {d['chapters'][0]} a {d['chapters'][1]}) : {d['repaired']}/{d['bad']} repare(s), "
                             f"titres/fins ne descendant pas : {len(d['cadence_not_falling'])}/{d['cadence_targets']}")
                for ch, j, kind, text in d['cadence_not_falling']:
                    lines.append(f'    intonation : chapitre {ch}, {kind} (troncon {j}) : {text}')
            lines.append('')
            if remaining:
                lines.append('A ECOUTER (defauts restants) :')
                for r in remaining:
                    lines.append(f"  chapitre {r['ch_num']}, troncon {r['j']} : {r['text'][:90]}")
                lines.append('')
            for r in entries:
                status = 'REPARE' if r['fixed'] else 'NON REPARE'
                if r.get('probable_false_positive'):
                    status += ' - probable faux positif (Whisper entend toujours la meme chose)'
                elif r.get('second_chance'):
                    status += ' (seconde chance)'
                lines.append(f"Chapitre {r['ch_num']}, troncon {r['j']} : {status}")
                lines.append(f"  texte  : {r['text']}")
                lines.append(f"  avant  : {r['before_asr']}  (CER {r['before_cer']})")
                if r['after_asr'] is not None:
                    lines.append(f"  apres  : {r['after_asr']}  (CER {r['after_cer']})")
                lines.append('')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f'Verification report: {path}')
            return path
        except Exception as e:
            print(f'verification report error: {e}')
            return None

    session = context.get_session(session_id)
    if not (session and session.get('id', False)):
        return False
    tts_manager = None
    asr_verifier = None
    conversion = False
    try:
        if session['cancellation_requested']:
            return False
        print(f'*********** Session: {session_id} **************\n{session_info}')
        tts_manager = TTSManager(session)
        blocks_current = session['blocks_current']
        blocks = blocks_current['blocks']
        block_resume = blocks_current['block_resume']
        sentence_resume = blocks_current['sentence_resume']
        blocks_saved = session['blocks_saved']
        prev_blocks_list = blocks_saved.get('blocks', [])
        prev_blocks = {b['id']: b for b in prev_blocks_list} if isinstance(prev_blocks_list, list) else prev_blocks_list
        xtts_languages = default_engine_settings[TTS_ENGINES['XTTS']].get('languages', {})
        _lang = session['language']
        if session.get('translate_enabled') and session.get('translate'):
            _lang = session['translate']
        if _lang != 'eng' and _lang in xtts_languages:
            is_voice_changed = False
            voice_cache = {}
            for block in blocks:
                old_voice = block.get('voice')
                if old_voice in voice_cache:
                    new_voice = voice_cache[old_voice]
                else:
                    if old_voice is None:
                        new_voice = None
                    else:
                        new_voice, error = tts_manager.set_voice(old_voice)
                        if new_voice is None and error is not None:
                            show_alert(session_id, {'type': 'warning', 'msg': error})
                            return False
                    voice_cache[old_voice] = new_voice
                if new_voice != old_voice:
                    is_voice_changed = True
                    block['voice'] = new_voice
            if is_voice_changed:
                blocks_current['blocks'] = blocks
                session['blocks_current'] = blocks_current
                save_db_blocks(session_id)
        total_chapters = sum(1 for b in blocks if b['keep'] and b['text'].strip())
        if total_chapters == 0:
            show_alert(session_id, {'type': 'warning', 'msg': 'No chapters found!'})
            return False
        total_sentences = sum(_count_sentences(b['sentences']) for b in blocks if b['keep'] and b['text'].strip())
        if total_sentences == 0:
            show_alert(session_id, {'type': 'warning', 'msg': 'No sentences found!'})
            return False
        if not session['ebook']:
            return False
        ebook_name = Path(session['ebook']).name
        chapters_dir = session['chapters_dir']
        sentences_dir = session['sentences_dir']
        global_sent = 0
        ch_num = 0
        last_save_time = time.monotonic()
        baseline_initialized = False
        # every finished chapter is exported right away to "<book> (en cours)"
        exporter = None
        if session.get('progressive_export', True) and session.get('audiobooks_dir'):
            try:
                exporter = ProgressiveExporter(session)
            except Exception as e:
                print(f'Progressive export disabled: {e}')
        # Whisper verification of the chunks (CPU watcher during the run, GPU repair at the end)
        if session.get('asr_check', True) and session['tts_engine'] in (TTS_ENGINES['XTTS'], TTS_ENGINES['F5TTS']):
            try:
                asr_verifier = AsrVerifier(session, models_dir)
                if asr_verifier.enabled:
                    asr_verifier.start_watcher()
                else:
                    print('Whisper models not found in models/, verification disabled')
                    asr_verifier = None
            except Exception as e:
                print(f'Whisper verification disabled: {e}')
                asr_verifier = None
        # the book is finished part by part (E2A_REPAIR_PARTS, 3 by default, whole chapters of
        # about the same number of chunks): as soon as the last chapter of a part is
        # synthesised, the part is checked by Whisper, repaired, rebuilt and delivered as its
        # own m4b, then the synthesis goes on. Finished parts are recorded so a resumed run
        # never does them again.
        kept = [b for b in blocks if b['keep'] and b['text'].strip()]
        by_id = {b['id']: (n, b) for n, b in enumerate(kept, start=1)}
        parts = split_in_parts([_count_sentences(b['sentences']) for b in kept], int(os.environ.get('E2A_REPAIR_PARTS', '3')))
        part_of_last = {p[-1]: k for k, p in enumerate(parts, start=1)}
        asr_confirm_cer = ASR_CONFIRM_CER
        # controls of a corrected part; a part delivered before one of them existed is checked
        # for it when the conversion resumes ('point': final full stop read aloud by XTTS)
        asr_controls = ['cer', 'cadence', 'point']
        cadence_on = os.environ.get('E2A_CADENCE', '1') != '0' and session['tts_engine'] in (TTS_ENGINES['XTTS'], TTS_ENGINES['F5TTS'])
        cadence_log = os.path.join(session['process_dir'], 'cadence.jsonl')
        # E2A_REPAIR_AGAIN=<file of "ch_num j" lines>: manual second chance for chosen chunks,
        # with new seeds, once every part is done
        again_file = os.environ.get('E2A_REPAIR_AGAIN')
        again_takes = int(os.environ.get('E2A_REPAIR_AGAIN_TAKES', '8'))
        again = set()
        if again_file and os.path.exists(again_file):
            for line in open(again_file, encoding='utf-8-sig'):
                if line.split() and not line.startswith('#'):
                    again.add(tuple(int(x) for x in line.split()[:2]))
        parts_state_path = os.path.join(session['process_dir'], 'asr_parts_done.json')
        parts_state = {'parts': parts, 'done': {}}
        try:
            with open(parts_state_path, encoding='utf-8-sig') as f:
                saved_state = json.load(f)
            if saved_state.get('parts') == parts:
                parts_state['done'] = saved_state.get('done', {})
            else:
                print('[verif] decoupage en parties modifie : les parties deja faites seront refaites')
        except (OSError, ValueError):
            pass
        print(f"[verif] {len(parts)} partie(s) : " + ', '.join(f'{k} = chapitres {p[0]} a {p[-1]}' + (' (deja faite)' if str(k) in parts_state['done'] else '') for k, p in enumerate(parts, 1)), flush=True)
        msg = (f'---------<br/>'
               f"{session['filename_noext']}<br/>"
               f"A total of {total_chapters} {'block' if total_chapters <= 1 else 'blocks'} "
               f"and {total_sentences} {'sentence' if total_sentences <= 1 else 'sentences'}."
               f'<br/>---------')
        show_alert(session_id, {'type': 'info', 'msg': msg})
        with tqdm(total=total_sentences, desc='0.00%', bar_format='{desc}: {n_fmt}/{total_fmt} ', unit='step', initial=0) as t:
            for x, block in enumerate(blocks):
                if not (block['keep'] and block['text'].strip()):
                    continue
                if session['cancellation_requested']:
                    return False
                ch_num += 1
                block_id = block['id']
                sentences = block['sentences']
                block_len = len(sentences)
                valid_idx = {i for i,s in enumerate(sentences) if any(c.isalnum() for c in s.strip())}
                last_idx = block_len - 1
                sent_start = global_sent
                current_hash = block_hash(block)
                block_ref = prev_blocks.get(block_id)
                hash_ref = block_hash(block_ref) if block_ref else None
                block_changed = block_ref is not None and hash_ref != current_hash
                missing_sentences = set()
                start_sentence = 0
                chapter_audio_file = os.path.join(chapters_dir, f'{block_id}.{default_audio_proc_format}')
                block_dir = os.path.join(sentences_dir, block_id)
                if x < block_resume and not block_changed:
                    if not os.path.exists(chapter_audio_file):
                        show_alert(session_id, {'type': 'warning', 'msg': f'Block {x} chapter audio missing, reconverting entire block…'})
                        _reset_chapter_file(block_id)
                    else:
                        missing_sentences = _check_block_sentences(block_id, sentences)
                        if not missing_sentences:
                            print(f'Chapter {ch_num} (block {x}) — has all sentences')
                            if exporter and not exporter.has(ch_num):
                                exporter.add_chapter(ch_num, chapter_audio_file, chapter_title_of(sentences, ch_num))
                            if asr_verifier:
                                asr_verifier.enqueue_chapter(block_id, ch_num, _asr_chunks(block_id, sentences))
                            cnt = len(valid_idx)
                            global_sent += cnt
                            t.update(cnt)
                            _after_chapter(ch_num)
                            continue
                        show_alert(session_id, {'type': 'warning', 'msg': f'Block {x} has {len(missing_sentences)} missing audio files, reconverting…'})
                        _reset_chapter_file(block_id)
                elif block_changed and x <= block_resume:
                    show_alert(session_id, {'type': 'info', 'msg': f'Chapter {ch_num} (block {x}) — changed, reconverting'})
                    _reset_chapter_file(block_id)
                elif x == block_resume and not block_changed:
                    if sentence_resume == 0 and os.path.isdir(block_dir):
                        shutil.rmtree(block_dir)
                    start_sentence = sentence_resume
                show_alert(session_id, {'type': 'info', 'msg': f'Chapter {ch_num} (block {x}) containing {block_len} sentences…'})
                os.makedirs(block_dir, exist_ok=True)
                blocks_current['block_resume'] = x
                blocks_current['sentence_resume'] = start_sentence
                session['blocks_current'] = blocks_current
                save_db_stamp(session_id)
                converted = False
                block_voice = block.get('voice') or session.get('voice')
                cad_targets = cadence.cadence_targets(sentences) if cadence_on else {}
                for j in range(block_len):
                    if session['cancellation_requested']:
                        msg = 'Conversion Cancelled'
                        return False
                    sentence = sentences[j].strip()
                    if j in valid_idx:
                        if j >= start_sentence or j in missing_sentences:
                            if j == start_sentence and start_sentence > 0:
                                show_alert(session_id, {'type': 'info', 'msg': f'*** Resuming from sentence {global_sent} ***'})
                            sentence_file = os.path.join(block_dir, f'{j}.{default_audio_proc_format}')
                            if tts_manager is None:
                                # unloaded by the correction of the previous part
                                tts_manager = TTSManager(session)
                            run, error = tts_manager.convert_sentence2audio(sentence_file, sentence, block_voice=block_voice)
                            if not run:
                                show_alert(session_id, {'type': 'warning', 'msg': error})
                                return False
                            if j in cad_targets:
                                _cadence_retakes(ch_num, j, cad_targets[j], sentence_file, sentence, block_voice)
                            converted = True
                            blocks_current['sentence_resume'] = j
                            now = time.monotonic()
                            if not baseline_initialized:
                                session['blocks_current'] = blocks_current
                                session['blocks_saved'] = copy.deepcopy(blocks_current)
                                save_json_blocks(session_id, 'blocks_saved')
                                baseline_initialized = True
                            elif now - last_save_time >= 5:
                                session['blocks_current'] = blocks_current
                                save_db_stamp(session_id)
                                last_save_time = now
                        global_sent += 1
                        total_progress = (t.n + 1) / total_sentences
                        if session['is_gui_process']:
                            progress_bar(progress=total_progress, desc=f'{ebook_name} - {sentence}')
                        t.set_description(f'{total_progress * 100:.2f}%')
                        print(f' : {sentence}')
                        t.update(1)
                sent_end = global_sent - 1
                show_alert(session_id, {'type': 'info', 'msg': f'End of Chapter {ch_num} (block {x})'})
                if converted or block_changed or missing_sentences:
                    show_alert(session_id, {'type': 'info', 'msg': f'Combining chapter {ch_num} (block {x}) to audio, sentence {sent_start} to {sent_end}'})
                    session['blocks_current'] = blocks_current
                    save_db_stamp(session_id)
                    last_save_time = time.monotonic()
                    if not combine_audio_sentences(session_id, chapter_audio_file, block_id, block_len):
                        show_alert(session_id, {'type': 'warning', 'msg': 'combine_audio_sentences() failed!'})
                        return False
                    if exporter:
                        exporter.add_chapter(ch_num, chapter_audio_file, chapter_title_of(sentences, ch_num))
                if asr_verifier:
                    asr_verifier.enqueue_chapter(block_id, ch_num, _asr_chunks(block_id, sentences))
                # the chapter is complete: a resume starts after it (resuming on its last
                # sentence would synthesise it again over a repaired or cadence-chosen take)
                blocks_current['block_resume'] = x + 1
                blocks_current['sentence_resume'] = 0
                session['blocks_current'] = blocks_current
                save_db_stamp(session_id)
                _after_chapter(ch_num)
            if asr_verifier and again:
                # manual second chance on chosen chunks, part by part
                for k_part in range(1, len(parts) + 1):
                    try:
                        _correct_part(k_part, again)
                    except Exception as e:
                        DependencyError(e)
                        print(f'[verif] seconde chance de la partie {k_part} interrompue : {e}')
            _deliver(_write_verification_report())
            #blocks_current['block_resume'] = 0
            #blocks_current['sentence_resume'] = 0
            session['blocks_current'] = blocks_current
            save_db_stamp(session_id)
            session['blocks_saved'] = copy.deepcopy(blocks_current)
            save_json_blocks(session_id, 'blocks_saved')
            conversion = True
            return True
    except Exception as e:
        DependencyError(e)
        exception_alert(session_id, f'convert_chapters2audio() error: {e}')
        return False
    finally:
        # every exit that is not a completed conversion drops the engine and evicts
        # its models: there are ten early `return False` paths in here, so a finally
        # is the only way to cover them all.
        if asr_verifier is not None and asr_verifier.proc is not None:
            asr_verifier.stop_watcher()
        if not conversion:
            unload_tts_manager(tts_manager)

def deliver_final_files(session:Any, exported_files:list[str])->None:
    # E2A_DELIVERY_DIR: the finished audiobook and its Whisper report are copied there
    dest_dir = os.environ.get('E2A_DELIVERY_DIR')
    if not dest_dir:
        return
    try:
        os.makedirs(dest_dir, exist_ok=True)
        stem = Path(session.get('final_name') or 'audiobook').stem
        report = os.path.join(session['audiobooks_dir'], f'{stem} - verification Whisper.txt')
        for path in list(exported_files) + [report]:
            if path and os.path.exists(path):
                shutil.copyfile(path, os.path.join(dest_dir, os.path.basename(path)))
                print(f'[livraison] {os.path.join(dest_dir, os.path.basename(path))}', flush=True)
    except OSError as e:
        print(f'[livraison] copie impossible : {e}')

def split_in_parts(sizes:list[int], n_parts:int)->list[list[int]]:
    """Chapter numbers (1-based) grouped in n_parts runs of whole chapters of about the
    same total size (number of chunks): a chapter goes to the part where its middle falls."""
    n_parts = max(1, n_parts)
    total = max(1, sum(sizes))
    parts = [[] for _ in range(n_parts)]
    acc = 0
    for n, size in enumerate(sizes, start=1):
        parts[min(n_parts - 1, int((acc + size / 2) * n_parts / total))].append(n)
        acc += size
    return [p for p in parts if p]

CHAPTER_LEAD_IN_S = 1.0
CHAPTER_TAIL_S = 1.5

def chapter_silence_files(work_dir:str, model_file:str|Path)->tuple[str|None, str|None]:
    # silences in the exact format of the sentence files, so the concat stays a stream copy
    try:
        import soundfile as sf
        import numpy as np
        info = sf.info(str(model_file))
        files = []
        for name, seconds in (('lead_in', CHAPTER_LEAD_IN_S), ('tail', CHAPTER_TAIL_S)):
            path = os.path.join(work_dir, f'_chapter_{name}_{info.samplerate}_{info.channels}_{info.subtype}.{default_audio_proc_format}')
            if not os.path.exists(path):
                sf.write(path, np.zeros((int(seconds * info.samplerate), info.channels), dtype='float32'), info.samplerate, subtype=info.subtype)
            files.append(path)
        return files[0], files[1]
    except Exception as e:
        print(f'chapter_silence_files() error: {e}')
        return None, None

def combine_audio_sentences(session_id:str, file:str, block_id:str, sentence_count:int)->bool:
    try:
        session = context.get_session(session_id)
        if not session or not session.get('id', False):
            error = 'Session expired!'
            print(error)
            return False
        if sentence_count == 0:
            error = f'No sentences to combine for block {block_id}.'
            print(error)
            return False
        block_dir = Path(session['sentences_dir']) / block_id
        ext = default_audio_proc_format
        selected_files = []
        missing = []
        for i in range(sentence_count):
            path = block_dir / f'{i}.{ext}'
            if path.is_file():
                selected_files.append(path)
            else:
                missing.append(i)
        if missing:
            error = f'Missing sentence files in block {block_id}: {missing}'
            print(error)
            return False
        concat_dir = session['process_dir']
        concat_list = os.path.join(concat_dir, 'concat_list_sentences.txt')
        # silence around every chapter: players (VLC…) need a moment to open the audio
        # output and eat the first syllable of a chapter starting at 0 s, and the listener
        # needs a breath between two chapters
        lead_in, tail = chapter_silence_files(concat_dir, selected_files[0])
        with open(concat_list, 'w') as f:
            if lead_in:
                f.write(f"file '{Path(lead_in).as_posix()}'\n")
            for path in selected_files:
                if session['cancellation_requested']:
                    return False
                f.write(f"file '{path.as_posix()}'\n")
            if tail:
                f.write(f"file '{Path(tail).as_posix()}'\n")
        result = assemble_audio_chunks(concat_list, file, session['is_gui_process'])
        if not result:
            error = 'combine_audio_sentences() FFmpeg concat failed.'
            print(error)
            return False
        msg = f'********* Combined block audio file saved in {file}'
        print(msg)
        return True
    except Exception as e:
        DependencyError(e)
        return False

CHAPTER_EXCERPT_CHARS = 38

def chapter_title_of(sentences:list[str], ch_num:int|None=None)->str:
    # the chapter title is its heading (the first spoken line, closed by a [pause]); a short
    # heading followed by a second one (part title merged by split_spine(), series name then
    # subtitle: "Doux chagrins" / "Jadis, il y a fort longtemps…") is completed with it.
    # A chapter without heading is named "Chapitre N — first words…": its first sentence
    # would make an unreadable chapter name. The name is only metadata, nothing is spoken.
    def _clean(s:str)->str:
        s = ' '.join(SML_TAG_PATTERN.sub(' ', s).split()).lstrip(' .;:…').rstrip(' .;:')
        return s[:1].upper() + s[1:]

    spoken = [s for s in sentences if any(c.isalnum() for c in SML_TAG_PATTERN.sub(' ', s))]

    def _is_heading(k:int)->bool:
        # the last chunk ends with the [pause] closing the chapter, it is not a heading
        return k < len(spoken) - 1 and bool(re.search(r'\[pause\]', spoken[k]))

    if not spoken:
        return f'Chapitre {ch_num}' if ch_num else ''
    if _is_heading(0) or ch_num is None:
        title = _clean(spoken[0])
        if len(title) < 40 and _is_heading(0) and _is_heading(1):
            title = f'{title} — {_clean(spoken[1])}'
        return title
    return f'Chapitre {ch_num} — {chapter_excerpt(_clean(spoken[0]))}'

def chapter_excerpt(text:str, max_chars:int=CHAPTER_EXCERPT_CHARS)->str:
    # first words of a text, cut on a word boundary: "Zachary Ezra Rawlins est assis par…"
    words = text.split()
    out = []
    for w in words:
        if out and len(' '.join(out + [w])) > max_chars:
            break
        out.append(w)
    return ' '.join(out).rstrip(' ,;:.!?…—-') + '…'

def combine_audio_chapters(session_id:str)->list[str]|None:
    
    def _on_progress(p:float, desc:str)->None:
        if is_gui_process:
            progress_bar(p / 100.0, desc=desc)

    def _generate_ffmpeg_metadata(part_chapters:list[tuple[str,str]], output_metadata_path:str, default_audio_proc_format:str, part_num:int=None)->str|bool:
        try:
            out_fmt = session['output_format']
            is_mp4_like = out_fmt in ['mp4', 'm4a', 'm4b', 'mov']
            is_vorbis = out_fmt in ['ogg', 'webm']
            is_mp3 = out_fmt == 'mp3'
            def tag(key):
                return key.upper() if is_vorbis else key
            ffmpeg_metadata = ';FFMETADATA1\n'
            if session['metadata'].get('title'):
                ffmpeg_metadata += f"{tag('title')}={session['metadata']['title']}\n"
            if session['metadata'].get('creator'):
                ffmpeg_metadata += f"{tag('artist')}={session['metadata']['creator']}\n"
            if session['metadata'].get('language'):
                ffmpeg_metadata += f"{tag('language')}={session['metadata']['language']}\n"
            if session['metadata'].get('description'):
                ffmpeg_metadata += f"{tag('description')}={session['metadata']['description']}\n"
            if session['metadata'].get('publisher') and (is_mp4_like or is_mp3):
                ffmpeg_metadata += f"{tag('publisher')}={session['metadata']['publisher']}\n"
            if session['metadata'].get('published'):
                try:
                    if '.' in session['metadata']['published']:
                        year = datetime.strptime(session['metadata']['published'], '%Y-%m-%dT%H:%M:%S.%f%z').year
                    else:
                        year = datetime.strptime(session['metadata']['published'], '%Y-%m-%dT%H:%M:%S%z').year
                except Exception:
                    year = datetime.now().year
            else:
                year = datetime.now().year
            if is_vorbis:
                ffmpeg_metadata += f"{tag('date')}={year}\n"
            else:
                ffmpeg_metadata += f"{tag('year')}={year}\n"
            if session['metadata'].get('identifiers') and isinstance(session['metadata']['identifiers'], dict):
                if is_mp3 or is_mp4_like:
                    isbn = session['metadata']['identifiers'].get('isbn')
                    if isbn:
                        ffmpeg_metadata += f"{tag('isbn')}={isbn}\n"
                    asin = session['metadata']['identifiers'].get('mobi-asin')
                    if asin:
                        ffmpeg_metadata += f"{tag('asin')}={asin}\n"
            start_time = 0
            total = len(part_chapters)
            progress_desc = f'Metadata Part {part_num}' if part_num is not None else 'Metadata'
            bar = None if is_gui_process else tqdm(total=total, desc=progress_desc, unit='ch', file=sys.stdout, dynamic_ncols=True, leave=True)
            for i, (filename, chapter_title) in enumerate(part_chapters):
                if session['cancellation_requested']:
                    if bar:
                        bar.close()
                    return False
                filepath = os.path.join(session['chapters_dir'], filename)
                duration_ms = len(AudioSegment.from_file(filepath, format=default_audio_proc_format))
                clean_title = re.sub(r'(^#)|[=\\]|(-$)', lambda m: '\\' + (m.group(1) or m.group(0)), sanitize_meta_chapter_title(chapter_title))
                ffmpeg_metadata += '[CHAPTER]\nTIMEBASE=1/1000\n'
                ffmpeg_metadata += f'START={start_time}\nEND={start_time + duration_ms}\n'
                ffmpeg_metadata += f"{tag('title')}={clean_title}\n"
                start_time += duration_ms
                if is_gui_process:
                    _on_progress((((i + 1) / total) * 100.0), progress_desc)
                else:
                    bar.update(1)
            if bar:
                bar.close()
            with open(output_metadata_path, 'w', encoding='utf-8') as f:
                f.write(ffmpeg_metadata)
            return output_metadata_path
        except Exception as e:
            error = f'_generate_ffmpeg_metadata() Error: {e}'
            print(error)
            return False

    def _export_audio(combined_audio:str, metadata_file:str, final_file:str, block_indices:set=None, part_num:int=None)->bool:
        try:
            if session['cancellation_requested']:
                return False
            ffprobe_cmd = [
                shutil.which('ffprobe'), '-v', 'error', '-threads', '0', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,sample_fmt',
                '-of', 'default=nokey=1:noprint_wrappers=1', combined_audio
            ]
            probe = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            if probe.returncode != 0:
                error = f'ffprobe failed for {combined_audio}: {probe.stderr.strip()}'
                print(error)
                return False
            codec_info = probe.stdout.strip().splitlines()
            input_codec = codec_info[0] if len(codec_info) > 0 else None
            input_rate = codec_info[1] if len(codec_info) > 1 else None
            cmd = [shutil.which('ffmpeg'), '-hide_banner', '-nostats', '-hwaccel', 'auto', '-thread_queue_size', '1024', '-i', combined_audio]
            target_codec, target_rate = None, None
            if session['output_format'] == 'wav':
                target_codec = 'pcm_s16le'
                target_rate = '44100'
                cmd += ['-map', '0:a', '-ar', target_rate, '-sample_fmt', 's16']
            elif session['output_format'] == 'aac':
                target_codec = 'aac'
                target_rate = '44100'
                cmd += ['-c:a', 'aac', '-b:a', '192k', '-ar', target_rate, '-movflags', '+faststart']
            elif session['output_format'] == 'flac':
                target_codec = 'flac'
                target_rate = '44100'
                cmd += ['-c:a', 'flac', '-compression_level', '5', '-ar', target_rate]
            else:
                cmd += ['-f', 'ffmetadata', '-i', metadata_file, '-map', '0:a']
                if session['output_format'] in ['m4a', 'm4b', 'mp4', 'mov']:
                    target_codec = 'aac'
                    target_rate = '44100'
                    cmd += ['-c:a', 'aac', '-b:a', '192k', '-ar', target_rate, '-movflags', '+faststart+use_metadata_tags']
                elif session['output_format'] == 'mp3':
                    target_codec = 'mp3'
                    target_rate = '44100'
                    cmd += ['-c:a', 'libmp3lame', '-b:a', '192k', '-ar', target_rate]
                elif session['output_format'] == 'webm':
                    target_codec = 'opus'
                    target_rate = '48000'
                    cmd += ['-c:a', 'libopus', '-b:a', '192k', '-ar', target_rate]
                elif session['output_format'] == 'ogg':
                    target_codec = 'opus'
                    target_rate = '48000'
                    cmd += ['-c:a', 'libopus', '-compression_level', '0', '-b:a', '192k', '-ar', target_rate]
                cmd += ['-map_metadata', '1']
            if session['output_channel'] == 'stereo':
                cmd += ['-ac', '2']
            else:
                cmd += ['-ac', '1']
            if input_codec == target_codec and input_rate == target_rate:
                cmd = [
                    shutil.which('ffmpeg'), '-hide_banner', '-nostats', '-hwaccel', 'auto', '-thread_queue_size', '1024', '-i', combined_audio,
                    '-threads', '0', '-f', 'ffmetadata', '-i', metadata_file,
                    '-map', '0:a', '-map_metadata', '1', '-c', 'copy',
                    '-progress', 'pipe:2',
                    '-y', final_file
                ]
            else:
                cmd += [
                    '-filter_threads', '0',
                    '-filter_complex_threads', '0',
                    '-af', 'dynaudnorm=f=150:g=15,afftdn=nf=-70',
                    '-threads', '0',
                    '-progress', 'pipe:2',
                    '-y', final_file
                ]
            progress_desc = f'Export Part {part_num}' if part_num is not None else 'Export'
            proc_pipe = SubprocessPipe(cmd, is_gui_process=is_gui_process, total_duration=get_audio_duration(combined_audio), msg='Export', on_progress=lambda p: _on_progress(p, progress_desc))
            if not proc_pipe.result:
                error = f'ffmpeg export failed for {final_file}'
                print(error)
                return False
            if not (os.path.exists(final_file) and os.path.getsize(final_file) > 0):
                error = f'{Path(final_file).name} is corrupted or does not exist'
                print(error)
                return False
            if session['cover'] is not None:
                cover_path = session['cover']
                msg = f'Adding cover {cover_path} into the final audiobook file…'
                print(msg)
                if session['output_format'] == 'webm':
                    msg = 'Cover embedding skipped: mutagen has no Matroska/WebM writer'
                    print(msg)
                else:
                    with open(cover_path, 'rb') as f:
                        cover_data = f.read()
                    mime = 'image/png' if Path(cover_path).suffix.lower() == '.png' else 'image/jpeg'
                    width, height, depth = 0, 0, 24
                    try:
                        from PIL import Image
                        with Image.open(cover_path) as img:
                            width, height = img.size
                            depth = len(img.getbands()) * 8
                    except Exception:
                        pass
                    audio = None
                    if session['output_format'] == 'mp3':
                        from mutagen.mp3 import MP3
                        from mutagen.id3 import ID3, APIC, error as id3_error
                        audio = MP3(final_file, ID3=ID3)
                        try:
                            audio.add_tags()
                        except id3_error:
                            pass
                        audio.tags.delall('APIC')
                        audio.tags.add(APIC(encoding=3, mime=mime, type=3, desc='Cover', data=cover_data))
                    elif session['output_format'] in ['mp4', 'm4a', 'm4b', 'mov']:
                        from mutagen.mp4 import MP4, MP4Cover
                        img_fmt = MP4Cover.FORMAT_PNG if mime == 'image/png' else MP4Cover.FORMAT_JPEG
                        audio = MP4(final_file)
                        audio['covr'] = [MP4Cover(cover_data, imageformat=img_fmt)]
                    elif session['output_format'] in ['flac', 'ogg']:
                        from mutagen.flac import Picture
                        pic = Picture()
                        pic.type = 3
                        pic.mime = mime
                        pic.desc = 'Cover'
                        pic.width, pic.height, pic.depth = width, height, depth
                        pic.data = cover_data
                        if session['output_format'] == 'flac':
                            from mutagen.flac import FLAC
                            audio = FLAC(final_file)
                            audio.clear_pictures()
                            audio.add_picture(pic)
                        else:
                            from mutagen.oggopus import OggOpus
                            audio = OggOpus(final_file)
                            audio['metadata_block_picture'] = [base64.b64encode(pic.write()).decode('ascii')]
                    elif session['output_format'] == 'wav':
                        from mutagen.wave import WAVE
                        from mutagen.id3 import APIC
                        audio = WAVE(final_file)
                        if audio.tags is None:
                            audio.add_tags()
                        audio.tags.delall('APIC')
                        audio.tags.add(APIC(encoding=3, mime=mime, type=3, desc='Cover', data=cover_data))
                    elif session['output_format'] == 'aac':
                        from mutagen.id3 import ID3, APIC, ID3v1SaveOptions, ID3NoHeaderError
                        try:
                            tags = ID3(final_file)
                        except ID3NoHeaderError:
                            tags = ID3()
                        tags.delall('APIC')
                        tags.add(APIC(encoding=3, mime=mime, type=3, desc='Cover', data=cover_data))
                        tags.save(final_file, v1=ID3v1SaveOptions.REMOVE, v2_version=3)
                    if audio is not None:
                        audio.save()
            final_vtt = os.path.join(session['audiobooks_dir'], f'{Path(final_file).stem}.vtt')
            vtt_built, error = build_vtt_file(session, vtt_path=final_vtt, block_indices=block_indices)
            if not vtt_built:
                error = f'build_vtt_file() error: {error}'
                print(error)
                return False
            return True
        except Exception as e:
            error = f'Export failed: {e}'
            print(error)
            return False

    try:
        session = context.get_session(session_id)
        if not (session and session.get('id', False)):
            return None
        is_gui_process = session['is_gui_process']
        chapter_files = []
        chapter_titles = []
        chapter_positions = []
        for x, block in enumerate(session['blocks_current']['blocks']):
            if not (block['keep'] and block['text'].strip()):
                continue
            if not block.get('sentences'):
                error = f"Block {x} (id {block['id']}) has no sentences but is marked keep"
                print(error)
                return None
            block_id = block['id']
            fname = f'{block_id}.{default_audio_proc_format}'
            fpath = os.path.join(session['chapters_dir'], fname)
            if not os.path.exists(fpath):
                error = f'Missing chapter audio for block {x} (id {block_id}): {fpath}'
                print(error)
                return None
            chapter_files.append(fname)
            chapter_titles.append(chapter_title_of(block['sentences'], len(chapter_files)))
            chapter_positions.append(x)
        if len(chapter_files) == 0:
            print('No block files exist!')
            return None
        chunks_size = 892
        total_duration = 0.0
        durations = []
        for i in range(0, len(chapter_files), chunks_size):
            filepaths = [
                os.path.join(session['chapters_dir'], f)
                for f in chapter_files[i:i + chunks_size]
            ]
            durations_dict = get_audiolist_duration(filepaths)
            for path in filepaths:
                dur = durations_dict.get(path, 0.0)
                durations.append(dur)
                total_duration += dur
        if len(durations) != len(chapter_files):
            error = f'Duration count mismatch: {len(durations)} durations vs {len(chapter_files)} chapter files'
            print(error)
            return None
        exported_files = []
        concat_dir = session['process_dir']
        if session.get('output_split'):
            part_files = []
            part_chapter_indices = []
            cur_part = []
            cur_indices = []
            cur_duration = 0
            max_part_duration = int(session['output_split_hours']) * 3600
            for idx, (file, dur) in enumerate(zip(chapter_files, durations)):
                if session['cancellation_requested']:
                    return None
                if cur_part and (cur_duration + dur > max_part_duration):
                    part_files.append(cur_part)
                    part_chapter_indices.append(cur_indices)
                    cur_part = []
                    cur_indices = []
                    cur_duration = 0
                cur_part.append(file)
                cur_indices.append(idx)
                cur_duration += dur
            if cur_part:
                part_files.append(cur_part)
                part_chapter_indices.append(cur_indices)
            pad_width = len(str(len(part_files)))
            is_multi_part = len(part_files) > 1
            for part_idx, (part_file_list, indices) in enumerate(zip(part_files, part_chapter_indices)):
                concat_list = os.path.join(concat_dir, f'concat_list_chapters_{part_idx+1:0{pad_width}d}.txt')
                with open(concat_list, 'w') as f:
                    for file in part_file_list:
                        if session['cancellation_requested']:
                            return None
                        path = Path(session['chapters_dir']) / file
                        f.write(f"file '{path.as_posix()}'\n")
                merged_audio = Path(session['process_dir']) / f"{get_sanitized(session['metadata']['title'])}_part{part_idx+1:0{pad_width}d}.{default_audio_proc_format}"
                result = assemble_audio_chunks(concat_list, merged_audio, is_gui_process)
                if not result:
                    error = f'assemble_audio_chunks() Final merge failed for part {part_idx+1}.'
                    print(error)
                    return None
                metadata_file = Path(session['process_dir']) / f'metadata_part{part_idx+1:0{pad_width}d}.txt'
                part_chapters = [(chapter_files[i], chapter_titles[i]) for i in indices]
                _generate_ffmpeg_metadata(part_chapters, str(metadata_file), default_audio_proc_format)
                final_file = os.path.join(
                    session['audiobooks_dir'],
                    f"{Path(session['final_name']).stem}_part{part_idx+1:0{pad_width}d}.{session['output_format']}"
                    if is_multi_part else session['final_name']
                )
                block_indices = {chapter_positions[i] for i in indices} if is_multi_part else None
                if _export_audio(merged_audio, metadata_file, final_file, block_indices=block_indices, part_num=part_idx+1):
                    exported_files.append(final_file)
        else:
            concat_list = os.path.join(concat_dir, 'concat_list_chapters_1.txt')
            merged_audio = Path(session['process_dir']) / f"{get_sanitized(session['metadata']['title'])}.{default_audio_proc_format}"
            with open(concat_list, 'w') as f:
                for file in chapter_files:
                    if session['cancellation_requested']:
                        return None
                    path = Path(session['chapters_dir']) / file
                    f.write(f"file '{path.as_posix()}'\n")
            result = assemble_audio_chunks(concat_list, merged_audio, is_gui_process)
            if not result:
                print(f'assemble_audio_chunks() Final merge failed for {merged_audio}.')
                return None
            metadata_file = os.path.join(session['process_dir'], 'metadata.txt')
            chapters_zip = list(zip(chapter_files, chapter_titles))
            _generate_ffmpeg_metadata(chapters_zip, metadata_file, default_audio_proc_format)
            final_file = os.path.join(session['audiobooks_dir'], session['final_name'])
            if _export_audio(merged_audio, metadata_file, final_file):
                exported_files.append(final_file)
        return exported_files if exported_files else None
    except Exception as e:
        DependencyError(e)
        return None

def assemble_audio_chunks(txt_file:str, out_file:str, is_gui_process:bool)->bool:

    def _on_progress(p:float)->None:
        if is_gui_process:
            progress_bar(p / 100.0, desc='Assemble')

    try:
        total_duration = 0.0
        filepaths = []
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip().startswith('file'):
                        file_path = (
                            line.strip()
                            .split('file ')[1]
                            .strip()
                            .strip("'")
                            .strip('"')
                        )
                        if os.path.exists(file_path):
                            filepaths.append(file_path)
            durations = get_audiolist_duration(filepaths)
            total_duration = sum(durations.values())
        except Exception as e:
            error = f'assemble_audio_chunks() open file {txt_file} Error: {e}'
            print(error)
            return False
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            error = 'ffmpeg not found'
            print(error)
            return False
        cmd = [
            ffmpeg,
            '-hide_banner',
            '-nostats',
            '-safe', '0',
            '-f', 'concat',
            '-i', txt_file,
            '-c:a', default_audio_proc_format,
            '-map_metadata', '-1',
            '-threads', '0',
            '-progress', 'pipe:2',
            '-y', out_file
        ]
        proc_pipe = SubprocessPipe(
            cmd=cmd,
            is_gui_process=is_gui_process,
            total_duration=total_duration,
            msg='Assemble',
            on_progress=_on_progress
        )
        if proc_pipe.result and os.path.exists(out_file):
            msg = f'Completed → {out_file}'
            print(msg)
            return True
        else:
            error = f'Failed (proc_pipe) → {out_file}'
            print(error)
            return False
    except subprocess.CalledProcessError as e:
        DependencyError(e)
        return False
    except Exception as e:
        error = f'assemble_audio_chunks() Error: Failed to process {txt_file} → {out_file}: {e}'
        print(error)
        return False

def ellipsize_utf8_bytes(s:str, max_bytes:int, ellipsis:str='…')->str:
    s = '' if s is None else str(s)
    if max_bytes <= 0:
        return ''
    raw = s.encode('utf-8')
    e = ellipsis.encode('utf-8')
    if len(raw) <= max_bytes:
        return s
    if len(e) >= max_bytes:
        # return as many bytes of the ellipsis as fit
        return e[:max_bytes].decode('utf-8', errors='ignore')
    budget = max_bytes - len(e)
    out = bytearray()
    for ch in s:
        b = ch.encode('utf-8')
        if len(out) + len(b) > budget:
            break
        out.extend(b)
    return out.decode('utf-8') + ellipsis

def sanitize_meta_chapter_title(title:str, max_bytes:int=140)->str:
    # avoid None and embedded NULs which some muxers accidentally keep
    title = (title or '').replace('\x00', '')
    title = title.replace(sml_token('pause'), '')
    return ellipsize_utf8_bytes(title, max_bytes=max_bytes, ellipsis='…')

def strip_invalid_filename_characters(filename:str, max_bytes:int=200)->str:
    # Replicates gradio upload filename sanitization for headless mode.
    try:
        from gradio_client.utils import strip_invalid_filename_characters as gr_strip
        return gr_strip(filename, max_bytes)
    except ImportError:
        pass
    use_blocklist = True
    try:
        import gradio_client
        parts = gradio_client.__version__.split(".")
        use_blocklist = (int(parts[0]), int(parts[1])) > (1, 13)
    except Exception:
        pass
    name, ext = os.path.splitext(filename)
    if use_blocklist:
        # gradio_client > 1.13: remove only dangerous chars
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f`$!{}]', "", name)
    else:
        # gradio_client 1.13.x: keep only alnum + "._-, "
        name = "".join(char for char in name if char.isalnum() or char in "._-, ")
    filename = name + ext
    while len(filename.encode()) > max_bytes and name:
        name = name[:-1]
        filename = name + ext
    return filename

def delete_proc_audio_files(dir:str, files:list)->None:
    base = Path(dir)
    for file in base.rglob(f'[0-9]*.{default_audio_proc_format}'):
        if file.stem.isdigit() and file in files:
            file.unlink()

def delete_folder(folder_path:str)->None:
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)

def delete_unused_tmp_dirs(session_id:str, output_dir:str, days:int)->None:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        dir_array = [
            tmp_dir,
            output_dir,
            os.path.join(models_dir, '__sessions'),
            os.path.join(voices_dir, '__sessions')
        ]
        current_user_dirs = {
            f'proc-{session_id}',
            f'web-{session_id}',
            f'cli-{session_id}',
            f'voice-{session_id}',
            f'model-{session_id}'
        }
        current_time = time.time()
        threshold_time = current_time - (days * 24 * 60 * 60)  # Convert days to seconds
        for dir_path in dir_array:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                for dir in os.listdir(dir_path):
                    if dir in current_user_dirs:        
                        full_dir_path = os.path.join(dir_path, dir)
                        if os.path.isdir(full_dir_path):
                            try:
                                dir_mtime = os.path.getmtime(full_dir_path)
                                dir_ctime = os.path.getctime(full_dir_path)
                                if dir_mtime < threshold_time and dir_ctime < threshold_time:
                                    shutil.rmtree(full_dir_path, ignore_errors=True)
                                    msg = f'Deleted expired session: {full_dir_path}'
                                    print(msg)
                            except Exception as e:
                                error = f'Error deleting {full_dir_path}: {e}'
                                print(error)

def get_compatible_tts_engines(language:str)->list[str]:
    return [
        engine
        for engine, cfg in default_engine_settings.items()
        if language in cfg.get('languages', {})
    ]

def translate_blocks(session_id:str, raw_blocks:list)->tuple:
    try:
        session = context.get_session(session_id)
        if not session or not session.get('id', False):
            return raw_blocks, 'Session expired'
        if not session.get('translate_enabled') or not session.get('translate'):
            return raw_blocks, None
        source_iso1 = session.get('language_iso1')
        target_iso1 = session.get('translate_iso1')
        if not source_iso1 or not target_iso1:
            msg = f'Translation iso1 codes missing: {source_iso1} -> {target_iso1}'
            return raw_blocks, msg
        if source_iso1 == target_iso1:
            return raw_blocks, None
        total = len(raw_blocks)
        msg = f'Translating {total} block(s) {source_iso1} -> {target_iso1}'
        show_alert(session_id, {"type": "warning", "msg": msg})
        translator = ArgosTranslator(neural_machine='argostranslate')
        error, ok = translator.start(source_iso1, target_iso1)
        if not ok:
            return raw_blocks, error
        if progress_bar is not None:
            progress_bar(0.0, desc=msg)
        out = []
        tag_keys = '|'.join(map(re.escape, TTS_SML.keys()))
        sml_patterns = re.compile(
            rf'''
            \[
                \s*
                (?P<close>/)?
                \s*
                (?P<tag>{tag_keys})
                (?:\s*:\s*(?P<value>[^\]]*))?
                \s*
            \]
            ''',
            re.VERBOSE
        )
        with tqdm(total=total, desc='translate', unit='block') as t:
            for idx, text in enumerate(raw_blocks):
                if session['cancellation_requested']:
                    return raw_blocks, 'Conversion cancelled'
                if not text or not text.strip():
                    out.append(text)
                else:
                    translated, ok = translator.translate(text, sml_patterns)
                    if not ok:
                        error = f'Translation failed at block {idx}: {translated}'
                        return raw_blocks, error
                    out.append(translated)
                t.update(1)
                if progress_bar is not None:
                    progress_bar((t.n) / total, desc=f'Translating block {t.n}/{total} {source_iso1} -> {target_iso1}')
        msg = 'Translation done.'
        print(msg)
        if progress_bar is not None:
            progress_bar(1.0, desc=msg)
        return out, None
    except Exception as e:
        error = f'translate_blocks() error: {e}'
        return raw_blocks, error

def convert_ebook(args:dict)->tuple:
    try:
        global context
        error = None
        session_id = None
        info_session = None
        if not args.get('id'):
            error = 'Session ID is missing!'
            return error, False
        session_id = str(args['id'])
        session = context.get_session(session_id)
        if not session or (session and not session.get('id', False)):
            error = 'Session expired or does not exist!'
            return error, False
        if args['language'] is not None:
            try:
                if len(args['language']) in (2, 3):
                    lang_dict = Lang(args['language'])
                    if lang_dict:
                        args['language'] = lang_dict.pt3
                        args['language_iso1'] = lang_dict.pt1
                else:
                    args['language_iso1'] = None
            except Exception as e:
                pass
            if args['language'] not in language_mapping.keys():
                error = 'The language you provided is not (yet) supported'
                return error, False
            translate_to = args.get('translate') if args.get('translate') != args.get('language') else False
            translate_enabled = bool(args.get('translate_enabled')) and bool(translate_to)
            if translate_enabled:
                try:
                    if len(translate_to) in (2, 3):
                        ld_t = Lang(translate_to)
                        if ld_t:
                            translate_to = ld_t.pt3
                except Exception:
                    pass
                if translate_to not in language_mapping.keys():
                    error = f'--translate target language {translate_to} is not (yet) supported'
                    return error, False
                try:
                    target_iso1 = Lang(translate_to).pt1
                except Exception:
                    target_iso1 = None
                if not target_iso1:
                    error = f'--translate target {translate_to} has no iso639-1 mapping'
                    return error, False
                args['translate_enabled'] = True
                args['translate'] = translate_to
                args['translate_iso1'] = target_iso1
                final_language = str(args['translate'])
            else:
                args['translate_enabled'] = False
                args['translate'] = None
                args['translate_iso1'] = None
                final_language = str(args['language'])
            session['ebook_mode'] = args['ebook_mode']
            if session['ebook_mode'] == ebook_modes['DIRECTORY'] and isinstance(session.get('ebook_list'), list):
                session['ebook_list'] = sorted(session['ebook_list'], key=natural_sort_key)
            if session['ebook_mode'] == ebook_modes['TEXT']:
                if not args['ebook_textarea']:
                    error = 'Ebook textarea is empty.'
                    return error, False
                text = args['ebook_textarea']
                text_name = get_sanitized(text[:64])
                text_name_hash = hashlib.md5(f'{text_name}_{session_id}'.encode()).hexdigest()
                text_filename = SML_TAG_PATTERN.sub('', text_name)
                text_filename = get_sanitized(text_filename)
                text_filename = f'{text_filename[:48]}_{session_id}.txt'
                text_filepath = os.path.join(tempfile.gettempdir(), text_filename)
                with open(text_filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                session['ebook_textarea'] = args['ebook_textarea']
                session['ebook_textarea_src'] = text_filepath
                ebook_file = strip_invalid_filename_characters(text_filename)
                ebook_name = Path(text_filename).stem
            else:
                if not args.get('ebook_src'):
                    error = 'File source is empty.'
                    return error, False
                elif not os.path.splitext(args['ebook_src'])[1]:
                    error = f"{args['ebook_src']} needs a format extension."
                    return error, False
                elif not os.path.exists(args['ebook_src']):
                    error = 'File does not exist or Directory empty.'
                    return error, False
                session['ebook_src'] = str(args['ebook_src'])
                ebook_file = strip_invalid_filename_characters(Path(session['ebook_src']).name)
                ebook_name = get_sanitized(Path(session['ebook_src']).stem)
            ebook_name = strip_invalid_filename_characters(ebook_name)
            if session['ebook_mode'] != ebook_modes['TEXT']:
                if session.get('ebook_loaded') != session['ebook_src']:
                    session['blocks_orig'] = {}
                    session['blocks_current'] = {}
                    session['blocks_saved'] = {}
                    session['ebook_loaded'] = session['ebook_src']
            print(f"Processing eBook file: {ebook_file}")
            session['custom_model_dir'] = os.path.join(models_dir, '__sessions',f"model-{session_id}")
            session['script_mode'] = str(args['script_mode']) if args.get('script_mode') is not None else NATIVE
            session['is_gui_process'] = bool(args['is_gui_process'])
            session['blocks_preview'] = bool(args['blocks_preview']) if args.get('blocks_preview') else False
            session['device'] = str(args['device'])
            session['language'] = str(args['language'])
            session['language_iso1'] = str(args['language_iso1'])
            session['translate_enabled'] = bool(args.get('translate_enabled', False))
            session['translate'] = args.get('translate')
            session['translate_iso1'] = args.get('translate_iso1')
            session['tts_engine'] = str(args['tts_engine'])
            session['custom_model'] =  args['custom_model']
            session['fine_tuned'] = str(args['fine_tuned'])
            session['voice'] = args.get('voice', None)
            session['xtts_temperature'] =  float(args['xtts_temperature'])
            session['xtts_length_penalty'] = float(args['xtts_length_penalty'])
            session['xtts_num_beams'] = int(args['xtts_num_beams'])
            session['xtts_repetition_penalty'] = float(args['xtts_repetition_penalty'])
            session['xtts_top_k'] =  int(args['xtts_top_k'])
            session['xtts_top_p'] = float(args['xtts_top_p'])
            session['xtts_speed'] = float(args['xtts_speed'])
            session['xtts_enable_text_splitting'] = bool(args['xtts_enable_text_splitting'])
            session['bark_text_temp'] =  float(args['bark_text_temp'])
            session['bark_waveform_temp'] =  float(args['bark_waveform_temp'])
            session['output_format'] = str(args['output_format'])
            session['output_channel'] = str(args['output_channel'])
            session['output_split'] = bool(args['output_split'])
            session['output_split_hours'] = args['output_split_hours']if args['output_split_hours'] is not None else default_output_split_hours
            session['model_cache'] = f"{session['tts_engine']}-{session['fine_tuned']}"
            session['session_dir'] = os.path.join(tmp_dir, f'proc-{session_id}')
            session['status'] = status_tags['EDIT'] if session['blocks_preview'] else status_tags['CONVERTING'] 
            lang_prfx = (f'_{final_language}' if session.get('translate_enabled') else '')
            session['process_dir'] = os.path.join(session['session_dir'], hashlib.md5((ebook_name + lang_prfx).encode()).hexdigest())
            session['chapters_dir'] = os.path.join(session['process_dir'], 'chapters')
            session['sentences_dir'] = os.path.join(session['chapters_dir'], 'sentences')
            cleanup_models_cache()
            if session['is_gui_process']:
                session['final_name'] = ebook_name + lang_prfx + '.' + session['output_format']
            else:
                session['system'] = DEVICE_SYSTEM
                session['audiobooks_dir'] = os.path.abspath(args['output_dir']) if args.get('output_dir') is not None else os.path.join(audiobooks_cli_dir, f'cli-{session_id}')
                session['final_name'] = os.path.join(session['audiobooks_dir'], ebook_name + lang_prfx + '.' + session['output_format'])
                session['voice_dir'] = os.path.join(voices_dir, '__sessions', f'voice-{session_id}', final_language)
                session['abs_url'] = str(args.get('abs_url', ''))
                session['abs_api_token'] = str(args.get('abs_api_token', ''))
                session['abs_library'] = str(args.get('abs_library', ''))
                os.makedirs(session['voice_dir'], exist_ok=True)
                audio_pre_final_exist = os.path.exists(os.path.join(session['process_dir'], ebook_name + '.' + default_audio_proc_format))
                audio_sentences_exist = any(Path(session['sentences_dir']).rglob(f'*.{default_audio_proc_format}'))
                if audio_pre_final_exist or audio_sentences_exist:
                    msg = f"Warning! audio sentences or final file {ebook_name} of this conversion already exists!"
                    # audio exists, so the previous global voice matters: warn before the prompt,
                    # since [r]esume with a different global voice reconverts the affected blocks.
                    voice_note = build_voice_change_note(session['process_dir'], session.get('voice'), html=False)
                    if voice_note:
                        msg += voice_note
                    print(msg)
                    # unattended runs (tools/run_resumable.py) answer through the environment
                    choice = os.environ.get('E2A_RESUME_CHOICE', '').strip().lower()
                    if choice in ('s', 'r', 'd'):
                        print(f'[{choice}] chosen by E2A_RESUME_CHOICE')
                    while choice not in ('s', 'r', 'd'):
                        choice = input("[s]kip / [r]esume / [d]elete and convert again: ").strip().lower()
                        if choice in ('s', 'r', 'd'):
                            break
                        print("Please enter 's', 'r' or 'd'.")
                    if choice == 'r':
                        if audio_pre_final_exist:
                            os.unlink(audio_pre_final_exist)
                        if os.path.exists(session['final_name']):
                            os.unlink(session['final_name'])
                    elif choice == 'd':
                        delete_folder(session['process_dir'])
                    elif choice == 's':
                        msg = 'Conversion skipped.'
                        return msg, True
                if error is None:
                    delete_unused_tmp_dirs(session_id, audiobooks_cli_dir, tmp_expire)
                    if session['custom_model'] is not None:
                        if not os.path.exists(session['custom_model_dir']):
                            os.makedirs(session['custom_model_dir'], exist_ok=True)
                        custom_src_path = Path(session['custom_model'])
                        custom_src_name = custom_src_path.stem
                        if not os.path.exists(os.path.join(session['custom_model_dir'], custom_src_name)):
                            try:
                                if analyze_uploaded_file(session['custom_model'], default_engine_settings[session['tts_engine']]['files']):
                                    model = extract_custom_model(session_id)
                                    if model is not None:
                                        session['custom_model'] = model
                                    else:
                                        error = f"{model} could not be extracted or mandatory files are missing"
                                else:
                                    error = f'{os.path.basename(f)} is not a valid model or some required files are missing'
                            except ModuleNotFoundError as e:
                                error = f"No presets module for TTS engine '{session['tts_engine']}': {e}"
                    if session.get('voice'):
                        voice_name = os.path.splitext(os.path.basename(session['voice']))[0].replace('&', 'And')
                        voice_name = get_sanitized(voice_name)
                        final_voice_file = os.path.join(session['voice_dir'], f'{voice_name}.wav')
                        if not os.path.exists(final_voice_file):
                            extractor = VoiceExtractor(session, session['voice'], voice_name)
                            voice_status, msg = extractor.extract_voice()
                            if voice_status:
                                session['voice'] = final_voice_file
                            else:
                                error = f'VoiceExtractor.extract_voice() failed! {msg}'
                        else:
                            # resumed session: keep using the voice prepared by the first run,
                            # the same audio (and path) as the chapters already converted
                            session['voice'] = final_voice_file
            if error is None:
                if prepare_dirs(session_id):
                    session['ebook'] = os.path.join(session['process_dir'], ebook_file)
                    shutil.copy((session['ebook_textarea_src'] if session['ebook_mode'] == ebook_modes['TEXT'] else session['ebook_src']), session['ebook'])
                    session['filename_noext'] = os.path.splitext(os.path.basename(session['ebook']))[0]
                    msg = ''
                    msg_extra = ''                      
                    if session['device'] == devices['CUDA']['proc']:
                        if not devices['CUDA']['found']:
                            session['device'] = devices['CPU']['proc']
                            msg += f'CUDA not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                    elif session['device'] == devices['JETSON']['proc'] or session['device'] == devices['JETSON']['proc']:
                        if not devices['JETSON']['found']:
                            session['device'] = devices['CPU']['proc']
                            msg += f'JETSON CUDA not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                    elif session['device'] == devices['MPS']['proc']:
                        if not devices['MPS']['found']:
                            session['device'] = devices['CPU']['proc']
                            msg += f'MPS not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                    elif session['device'] == devices['ROCM']['proc']:
                        if not devices['ROCM']['found']:
                            session['device'] = devices['CPU']['proc']
                            msg += f'ROCM not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                    elif session['device'] == devices['XPU']['proc']:
                        # devices['XPU']['found'] is a static capability flag: on an
                        # image built for xpu it stays True even when no Intel GPU is
                        # visible at runtime (Level Zero driver absent, /dev/dri not
                        # passed to the container). Unlike CUDA that never trips the
                        # branch below, so the failure surfaces as a crash inside
                        # TTSManager() instead of a switch to CPU. Ask torch directly.
                        xpu_error = None
                        if not devices['XPU']['found']:
                            xpu_error = 'XPU not supported by the Torch installed!'
                        else:
                            try:
                                import torch
                                if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
                                    xpu_error = 'XPU not available: no Intel GPU visible to Torch!'
                            except Exception as e:
                                xpu_error = f'XPU not available: runtime probe failed ({e!r})'
                        if xpu_error is not None:
                            session['device'] = devices['CPU']['proc']
                            msg += f'{xpu_error}<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                    if session['device'] == devices['CPU']['proc']:
                        os.environ['OMP_NUM_THREADS'] = '4'
                    vram_dict = VRAMDetector().detect_vram(session['device'], session['script_mode'])
                    print(f'vram_dict: {vram_dict}')
                    total_vram_gb = vram_dict.get('total_vram_gb', 0)
                    detected_free_vram_gb = vram_dict.get('free_vram_gb', 0)
                    session['free_vram_gb'] = detected_free_vram_gb
                    if session['free_vram_gb'] == 0:
                        msg_extra += f"<br/>Memory capacity not detected! restrict to {session['free_vram_gb']}GB max"
                    else:
                        msg_extra += f"<br/>Free Memory available: {session['free_vram_gb']}GB"
                        if session['free_vram_gb'] < default_engine_settings[session['tts_engine']]['rating']['VRAM']:
                            msg_extra += f"<br/>Free Memory {session['free_vram_gb']} is lower than VRAM/RAM {default_engine_settings[session['tts_engine']]['rating']['VRAM']}GB required!<br/>It will probably crash the conversion!"
                        if session['free_vram_gb'] > 4.0:
                            if session['tts_engine'] == TTS_ENGINES['BARK']:
                                os.environ['SUNO_USE_SMALL_MODELS'] = 'FALSE'  
                    if session['tts_engine'] == TTS_ENGINES['BARK']:
                        if session['free_vram_gb'] < 12.0:
                            os.environ['SUNO_OFFLOAD_CPU'] = 'TRUE'
                            os.environ['SUNO_USE_SMALL_MODELS'] = "TRUE"
                            msg_extra += f'<br/>Switching BARK to SMALL models'
                        else:
                            os.environ['SUNO_OFFLOAD_CPU'] = 'FALSE'
                            os.environ['SUNO_USE_SMALL_MODELS'] = 'FALSE'
                    if msg == '':
                        msg_extra = f"Using {session['device'].upper()}" + msg_extra
                    device_vram_required = default_engine_settings[session['tts_engine']]['rating']['RAM'] if session['device'] == devices['CPU']['proc'] else default_engine_settings[session['tts_engine']]['rating']['VRAM']
                    if float(total_vram_gb) >= float(device_vram_required):
                        if msg:
                            show_alert(session_id, {"type": "warning", "msg": msg + msg_extra})
                        else:
                            show_alert(session_id, {"type": "info", "msg": msg_extra})
                        session['epub_path'] = os.path.join(session['process_dir'], f"__{session['filename_noext']}.epub")
                        session['blocks_orig_json'] = os.path.join(session['process_dir'], f"{file_prefixes['clone']}{session['filename_noext']}.json")
                        session['blocks_saved_json']   = os.path.join(session['process_dir'], f"{file_prefixes['saved']}{session['filename_noext']}.json")
                        session['blocks_current_db']   = os.path.join(session['process_dir'], f"{file_prefixes['current']}{session['filename_noext']}.db")
                        ok_checksum, error = compare_checksums(session_id)
                        blocks_orig_old = {}
                        if not ok_checksum or not os.path.exists(session['epub_path']):
                            result_epub = convert2epub(session_id)
                            if result_epub:
                                if os.path.exists(session['epub_path']):
                                    if os.path.exists(session['blocks_orig_json']):
                                        # keep the previous parse as baseline to realign block ids after the new parse
                                        blocks_orig_old = load_json_blocks(session['blocks_orig_json'])
                                        os.unlink(session['blocks_orig_json'])
                                    if not blocks_orig_old.get('blocks'):
                                        # no baseline to realign against: hard reset as before
                                        blocks_orig_old = {}
                                        if os.path.exists(session['blocks_saved_json']):
                                            os.unlink(session['blocks_saved_json'])
                                        db = session['blocks_current_db']
                                        for f in (db, db + '-wal', db + '-shm'):
                                            if os.path.exists(f):
                                                os.unlink(f)
                                    msg = f"NOTE: process folder {session['process_dir']} is strictly used for internal tasks and has nothing to do with the final conversion."
                                    print(msg)
                                else:
                                    error = f"convert2epub() {session['epub_path']} does not exists! check write permissions."
                            else:
                                error = 'convert2epub() error: could not convert to epub file!'
                        if error is None:
                            missing_orig_json = True
                            is_changed = False
                            blocks_orig = {}
                            if os.path.exists(session['blocks_orig_json']):
                                missing_orig_json = False
                                blocks_orig = load_json_blocks(session['blocks_orig_json'])
                                if blocks_orig:
                                    blocks = blocks_orig.get('blocks', [])
                                    new_blocks = []
                                    for block in blocks:
                                        if any(c.isalnum() for c in block.get('text','')):
                                            if not block.get('id'):
                                                block['id'] = str(uuid.uuid4())
                                                is_changed = True
                                            new_blocks.append(block)
                                    blocks_orig['blocks'] = new_blocks
                                    session['blocks_orig'] = blocks_orig
                                if is_changed:
                                    save_json_blocks(session_id, 'blocks_orig')
                            # load previous work unconditionally: it must survive a source file change
                            # so realign_blocks() has something to migrate. the positional id backfill
                            # is only valid within one parse, never across a re-parse (realign's job).
                            if os.path.exists(session['blocks_saved_json']):
                                blocks_saved = load_json_blocks(session['blocks_saved_json'])
                                if blocks_saved:
                                    session['blocks_saved'] = blocks_saved
                                    if is_changed and not missing_orig_json:
                                        blocks = blocks_saved.get('blocks', [])
                                        for i, block in enumerate(blocks):
                                            if i < len(blocks_orig['blocks']):
                                                block['id'] = blocks_orig['blocks'][i]['id']
                                        blocks_saved['blocks'] = blocks
                                        session['blocks_saved'] = blocks_saved
                                        save_json_blocks(session_id, 'blocks_saved')
                            if os.path.exists(session['blocks_current_db']):
                                blocks_current = load_db_blocks(session['blocks_current_db'])
                                if blocks_current:
                                    session['blocks_current'] = blocks_current
                                    if is_changed and not missing_orig_json:
                                        blocks = blocks_current.get('blocks', [])
                                        for i, block in enumerate(blocks):
                                            if i < len(blocks_orig['blocks']):
                                                block['id'] = blocks_orig['blocks'][i]['id']
                                        blocks_current['blocks'] = blocks
                                        session['blocks_current'] = blocks_current
                                        save_db_blocks(session_id)
                            epubBook = epub.read_epub(session['epub_path'], {'ignore_ncx': True})
                            if epubBook:
                                metadata = dict(session['metadata'])
                                for key, value in metadata.items():
                                    data = epubBook.get_metadata('DC', key)
                                    if data:
                                        for value, attributes in data:
                                            metadata[key] = value
                                metadata['language'] = final_language
                                metadata['title'] = metadata['title'] or Path(session['ebook']).stem.replace('_', ' ')
                                metadata['creator'] = False if not metadata['creator'] or metadata['creator'] == 'Unknown' else metadata['creator']
                                session['metadata'] = metadata
                                try:
                                    if len(session['metadata']['language']) == 2:
                                        lang_dict = Lang(final_language)
                                        if lang_dict:
                                            session['metadata']['language'] = lang_dict.pt3
                                except Exception as e:
                                    pass
                                if not session.get('translate_enabled'):
                                    if session['metadata']['language'] != session['language']:
                                        error = f"WARNING!!! language selected {session['language']} differs from the EPUB file language {session['metadata']['language']}"
                                        show_alert(session_id, {'type': 'warning', 'msg': error})
                                is_lang_in_tts_engine = (
                                    session.get('tts_engine') in default_engine_settings and
                                    final_language in default_engine_settings[session['tts_engine']].get('languages', {})
                                )
                                if is_lang_in_tts_engine:
                                    session['cover'] = get_cover(epubBook, session_id)
                                    if session.get('cover', False):
                                        if missing_orig_json:
                                            raw_blocks = get_blocks(session_id, epubBook)
                                            if raw_blocks and session.get('translate_enabled'):
                                                raw_blocks, error = translate_blocks(session_id, list(raw_blocks))
                                                if error is not None:
                                                    return error, False
                                            if raw_blocks:
                                                session['blocks_orig'] = {
                                                    "page": 0,
                                                    "block_resume": 0,
                                                    "sentence_resume": 0,
                                                    "voice": session['voice'],
                                                    "tts_engine": session['tts_engine'],
                                                    "fine_tuned": session['fine_tuned'],
                                                    "blocks": [
                                                        {
                                                            "id": str(uuid.uuid4()),
                                                            "expand": False,
                                                            "keep": True,
                                                            "text": t,
                                                            "voice": session['voice'],
                                                            "tts_engine": session['tts_engine'],
                                                            "fine_tuned": session['fine_tuned'],
                                                            "sentences": [],
                                                        }
                                                        for t in raw_blocks if t
                                                    ],
                                                }
                                            if session.get('blocks_orig', {}):
                                                if blocks_orig_old:
                                                    if not realign_blocks(session_id, blocks_orig_old):
                                                        # realign failed or nothing to migrate: restart from scratch.
                                                        # loud on purpose: a silent fallback here looks like a random
                                                        # full reconversion and is the whole bug class this guards.
                                                        msg = 'realign_blocks() could not migrate the previous work: restarting the conversion from scratch.'
                                                        print(msg)
                                                        show_alert(session_id, {'type': 'warning', 'msg': msg})
                                                        session['blocks_saved'] = {}
                                                        session['blocks_current'] = {}
                                                        if os.path.exists(session['blocks_saved_json']):
                                                            os.unlink(session['blocks_saved_json'])
                                                save_json_blocks(session_id, 'blocks_orig')
                                        if not session.get('blocks_current', {}):
                                            session['blocks_current'] = copy.deepcopy(session['blocks_orig'])
                                            save_db_blocks(session_id)
                                        if session.get('blocks_orig', {}) and session.get('blocks_current', {}):
                                            sync_globals_to_blocks(session_id)
                                            if session['blocks_preview']:
                                                msg = f'Chapters preview requested. Select which block to convert:'
                                                print(msg)
                                                progress_status = os.path.basename(session['ebook'])
                                                return progress_status, True
                                            else:
                                                progress_status, passed = finalize_audiobook(session_id)
                                                return progress_status, passed
                                        else:
                                            error = f"get_blocks() or save_json_blocks() failed! {session['blocks_orig']}"
                                    else:
                                        error = 'get_cover() failed!'
                                else:
                                    error = f"language {final_language} not supported by {session['tts_engine']}!"
                            else:
                                error = 'epubBook.read_epub failed!'
                    else:
                        error = f"Your device has not enough memory ({total_vram_gb}GB) to run {session['tts_engine']} engine ({device_vram_required}GB)"
                else:
                    error = f"Temporary directory {session['process_dir']} not removed due to failure."
        if session['cancellation_requested']:
            error = 'Conversion Cancelled'
        return error, False
    except Exception as e:
        error = f'convert_ebook() Exception: {e}\n{traceback.format_exc()}'
        return error, False

def finalize_audiobook(session_id:str)->tuple:
    try:
        session = context.get_session(session_id)
        is_preview = session.get('blocks_preview', False) if session else False
        result = lambda msg, ok: (gr.update(value=msg), gr.update(value=ok)) if is_preview else (msg, ok)

        def _fail(error):
            session['status'] = status_tags['END']
            return result(error, False)

        if not session or not session.get('id', False):
            msg = 'session expired!'
            return result(msg, False)
        if session['status'] not in [status_tags['EDIT'], status_tags['CONVERTING']]:
            msg = 'No blocks have been selected for the conversion!'
            return result(msg, False)
        if not session.get('blocks_current', {}):
            error = 'finalize_audiobook() failed! blocks_current empty!'
            return _fail(error)
        session['status'] = status_tags['CONVERTING']
        print('Get sentences…')
        blocks_current = session['blocks_current']
        blocks = blocks_current['blocks']
        for idx, block in enumerate(blocks):
            if session['cancellation_requested']:
                if session['status'] == status_tags['DISCONNECTED']:
                    context_tracker.end_session(session_id, session['socket_hash'])
                    msg = 'Frontend disconnected!'
                    return result(msg, False)
                msg = 'Conversion cancelled'
                return result(msg, False)
            if not block['keep'] or not block['text'].strip():
                block['sentences'] = []
                continue
            if block.get('sentences', []):
                print(f'Block {idx} — sentences already split, skipping')
                continue
            sentences_list = get_sentences(session_id, block['text'])
            if sentences_list is None:
                error = 'No sentences found!'
                return result(error, False)
            block['sentences'] = sentences_list
        blocks_current['blocks'] = blocks
        session['blocks_current'] = blocks_current
        conversion = convert_chapters2audio(session_id)
        if not conversion:
            error = 'convert_chapters2audio() failed!'
            session = context.get_session(session_id)
            if session and session.get('id', False):
                if session['cancellation_requested']:
                    error = 'Conversion cancelled'
            return _fail(error)
        show_alert(session_id, {'type': 'info', 'msg': 'Combining sentences and chapters…'})
        exported_files = combine_audio_chapters(session_id)
        if exported_files is None:
            return _fail('combine_audio_chapters() error: exported_files not created!')
        session['audiobook'] = exported_files[-1]
        deliver_final_files(session, exported_files)
        if not session['is_gui_process'] and session['abs_url'] and session['abs_api_token'] and session['abs_library']:
            from lib.classes.audiobookshelf import fetch_libraries, upload_to_abs
            try:
                abs_libs = fetch_libraries(session['abs_url'], session['abs_api_token'])
                if abs_libs:
                    abs_library_id = next((v for name, v in abs_libs if name == session['abs_library']), '')
                    if abs_library_id:
                        a_title = os.path.basename(session['audiobook'])
                        a_author = str(session.get('metadata', {}).get('creator') or '')
                        ok, msg = upload_to_abs([session['audiobook']], a_title, a_author, session['abs_url'],  session['abs_api_token'], abs_library_id)
                        if ok:
                            msg = f'ABS upload: {msg}'
                            print(msg)
                        else:
                            error = f'ABS upload failed: {msg}'
                            print(error)
                    else:
                        error = 'ABS upload failed: library not found.'
                        print(error)
                else:
                    error = 'ABS upload failed: Could not search libraries.'
                    print(error)
            except Exception as e:
                error = f'ABS upload error: {e}'
                print(error)
        filename = os.path.basename(session['ebook'])
        count_ebook = 0
        if session['ebook_mode'] == ebook_modes['DIRECTORY']:
            if isinstance(session['ebook_list'], list):
                if session['ebook_src'] in session['ebook_list']:
                    ebook_list = session['ebook_list']
                    ebook_list.remove(session['ebook_src'])
                    session['ebook_list'] = ebook_list
                count_ebook = len(session['ebook_list'])
        if count_ebook > 0:
            reset_ebook_session(session_id, force=True, filter_keys=False)
            show_alert(session_id, {'type': 'success', 'msg': f'{filename} / converted. {count_ebook} ebook(s) conversion remaining…'})
        else:
            if session['ebook_mode'] == ebook_modes['DIRECTORY']:
                session['ebook_list'] = None
                session['voice_map'] = {}
                session['ebook_selected'] = None
            elif session['ebook_mode'] == ebook_modes['SINGLE']:
                session['ebook_src'] = None
            elif session['ebook_mode'] == ebook_modes['TEXT']:
                try:
                    os.remove(session['ebook_textarea_src'])
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            session['status'] = status_tags['END']
            reset_ebook_session(session_id, force=True, filter_keys=False)
            show_alert(session_id, {'type': 'success', 'msg': f'{filename} / converted.'})
            print(f'*********** Session: {session_id} **************\n{session_info}')
        return result(filename, True)
    except Exception as e:
        session['status'] = status_tags['END']
        reset_ebook_session(session_id, force=True, filter_keys=False)
        DependencyError(e)
        error = f'finalize_audiobook(): {e}'
        exception_alert(session_id, error)
        return result(error, False)

def on_unload(req:gr.Request)->None:
    socket_hash = req.session_hash
    if any(socket_hash in session for session in context.sessions.values()):
        session_id = context.find_id_by_hash(socket_hash)
        if session_id:
            session = context.get_session(session_id)
            if session['status'] == status_tags['CONVERTING']:
                session['cancellation_requested'] = True
                session['status'] = status_tags['DISCONNECTED']
                session['socket_hash'] = socket_hash
            else:
                context_tracker.end_session(session_id, socket_hash)

def restore_session_from_data(data:dict, session:DictProxy, force:bool, filter_keys:bool)->None:
    try:
        for key, value in data.items():
            if key in session:
                if filter_keys and key in save_session_keys_except:
                    continue
                if isinstance(value, dict) and value and isinstance(session[key], dict):
                    nested = session[key]
                    restore_session_from_data(value, nested, force, filter_keys)
                    session[key] = nested
                else:
                    if not force:
                        if value is None and session[key] is not None:
                            continue
                    session[key] = value
    except Exception as e:
        DependencyError(e)
        error = f'restore_session_from_data(): {e}'
        exception_alert(session['id'], error)

def reset_ebook_session(session_id:str, force:bool, filter_keys:bool)->None:
    session = context.get_session(session_id)
    data = {
        "ebook": None,
        "ebook_textarea_src": None,
        "process_dir": None,
        "chapters_dir": None,
        "sentences_dir": None,
        "epub_path": None,
        "final_name": None,
        "filename_noext": None,
        "cover": None,
        "blocks_orig": {},
        "blocks_saved": {},
        "blocks_current": {},
        "blocks_orig_json": None,
        "blocks_saved_json": None,
        "blocks_current_db": None,
        "audiobook_overridden": None,
        "metadata": {
            "title": None, 
            "creator": None,
            "contributor": None,
            "language": None,
            "identifier": None,
            "publisher": None,
            "date": None,
            "description": None,
            "subject": None,
            "rights": None,
            "format": None,
            "type": None,
            "coverage": None,
            "relation": None,
            "Source": None,
            "Modified": None,
        }
    }
    restore_session_from_data(data, session, force, filter_keys=filter_keys)

def unload_tts_manager(tts_manager:Any)->None:
    # called when convert_chapters2audio() gives up. The engine's own
    # cleanup_memory() only flushes the caching allocator; the weights stay alive
    # because loaded_tts holds them under the engine's key, and
    # cleanup_models_cache() deliberately protects that key while the session is
    # still active. Order matters: popping loaded_tts frees nothing while
    # tts_manager.engine still references the model, so the engine goes first.
    # tts_manager is None when TTSManager() itself failed (engine load OOM):
    # there is no engine to evict, but the half-loaded model's garbage is still
    # there, so the flush below must run regardless.
    try:
        if tts_manager is not None:
            engine = getattr(tts_manager, 'engine', None)
            keys = [getattr(engine, attr, None) for attr in ('tts_key', 'tts_zs_key')]
            # engines driving a worker process (F5-TTS) must stop it to free the GPU
            if engine is not None and hasattr(engine, 'close'):
                try:
                    engine.close()
                except Exception:
                    pass
            tts_manager.engine = None
            engine = None
            for key in keys:
                if key:
                    loaded_tts.pop(key, None)
        gc.collect()
        if sys.platform == 'linux':
            # gc frees the python objects but glibc keeps the pages in its arena:
            # on jetson unified memory those unreturned pages starve CUDA itself
            # (NvMap ENOMEM), so hand them back to the OS.
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    except Exception as e:
        error = f'unload_tts_manager() error: {e}'
        print(error)

def cleanup_models_cache()->None:
    try:
        active_models = {
            cache
            for session in context.sessions.values()
            for cache in (session.get('model_cache'), session.get('model_zs_cache'), session.get('stanza_cache'))
            if cache is not None
        }
        for key in list(loaded_tts.keys()):
            if key not in active_models:
                del loaded_tts[key]
        gc.collect()
        if sys.platform == 'linux':
            # return freed pages to the OS: on jetson unified memory glibc's
            # retained arena counts against the same pool CUDA allocates from.
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass
        # gc drops the python refs, but the caching allocator keeps the freed blocks
        # reserved: without this the next conversion sees allocated ~14MB against
        # reserved ~4GB, and vram_dict reports driver-free so it looks fully used.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    except Exception as e:
        error = f"cleanup_models_cache() error: {e}"
        print(error)

def show_alert(session_id:str|None, state:dict|None)->None:
    if state is not None:
        if state.get('msg'):
            print(state['msg'].replace('<br/>', '\n'))
            if session_id is not None:
                session = context.get_session(session_id)
                if session.get('is_gui_process'):
                    if isinstance(state, dict):
                        if state['type'] is not None:
                            if state['type'] == 'error':
                                raise gr.Error(state['msg'])
                            elif state['type'] == 'warning':
                                gr.Warning(state['msg'])
                            elif state['type'] == 'info':
                                gr.Info(state['msg'])
                            elif state['type'] == 'success':
                                gr.Success(state['msg'])

def exception_alert(session_id:str|None, error:str|None)->None:
    if error is not None:
        print(error.replace('<br/>', '\n'))
        if session_id is not None:
            session = context.get_session(session_id)
            if session and session.get('id', False):
                session['status'] = status_tags['READY']
                if session['is_gui_process']:
                    raise gr.Error(error)

def get_all_ip_addresses()->list:
    ip_addresses = []
    for interface, addresses in psutil.net_if_addrs().items():
        for address in addresses:
            if address.family in [socket.AF_INET, socket.AF_INET6]:
                ip_addresses.append(address.address)
    return ip_addresses
