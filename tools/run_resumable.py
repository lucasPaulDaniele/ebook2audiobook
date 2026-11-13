"""
Runs a headless ebook2audiobook conversion and resumes it automatically when it stops
abnormally (crash, GPU driver reset…), on the same session so nothing already synthesised
is lost.

  python tools/run_resumable.py [--session <id>] [--retries 10] -- <app.py headless args>

Without --session the first run creates one and its id is read from the output
("proc-<uuid>"); every relaunch then adds "--session <id>" and answers the resume prompt
automatically (E2A_RESUME_CHOICE=r). The id is also written to <output_dir>/SESSION.txt
together with a REPRENDRE.cmd to resume by hand after a reboot or a closed window.
"""

import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_RE = re.compile(r'proc-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')


def log(msg:str)->None:
    print(f"[reprise {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_resume_files(app_args:list[str], session:str, extra_env:dict|None=None)->None:
    if '--output_dir' not in app_args:
        return
    out = app_args[app_args.index('--output_dir') + 1]
    try:
        with open(os.path.join(out, 'SESSION.txt'), 'w', encoding='utf-8') as f:
            f.write(session + '\n')
        args = ' '.join(f'"{a}"' if (' ' in a or not a) else a for a in app_args)
        with open(os.path.join(out, 'REPRENDRE.cmd'), 'w', encoding='utf-8') as f:
            f.write('@echo off\r\n')
            f.write('rem Reprend la conversion de ce livre la ou elle s\'est arretee (meme session).\r\n')
            f.write('title E2A - Reprise de la conversion\r\n')
            f.write(f'cd /d "{ROOT}"\r\n')
            sets = ''.join(f' --set "{k}={v}"' for k, v in (extra_env or {}).items())
            f.write(f'"{sys.executable}" -u "{os.path.abspath(__file__)}" --session {session}{sets} -- {args}\r\n')
            f.write('timeout /t 10\r\n')
    except OSError as e:
        log(f'fichiers de reprise non ecrits : {e}')


def copy_output(proc:subprocess.Popen, state:dict, app_args:list[str], extra_env:dict)->None:
    buf = b''
    while True:
        byte = proc.stdout.read(1)
        if not byte:
            break
        sys.stdout.buffer.write(byte)
        if byte in (b'\n', b'\r'):
            sys.stdout.flush()
            if state['session'] is None:
                m = SESSION_RE.search(buf.decode('utf-8', errors='ignore'))
                if m:
                    state['session'] = m.group(1)
                    write_resume_files(app_args, state['session'], extra_env)
                    log(f"session {state['session']} enregistree pour la reprise")
            buf = b''
        else:
            buf += byte
    sys.stdout.flush()


def kill_orphans(dead_pid:int)->int:
    """Kills what is left of a finished app.py (children whose parent is gone): they would
    keep the GPU, the output pipe or the session files busy for the next run."""
    try:
        import psutil
    except ImportError:
        return 0
    victims = []
    parents = {dead_pid}
    found = True
    while found:
        found = False
        for p in psutil.process_iter(['pid', 'ppid']):
            if p.info['ppid'] in parents and p.info['pid'] not in parents:
                parents.add(p.info['pid'])
                victims.append(p)
                found = True
    for p in victims:
        try:
            p.kill()
        except psutil.Error:
            pass
    return len(victims)


def main()->None:
    argv = sys.argv[1:]
    if '--' not in argv:
        print(__doc__)
        sys.exit(1)
    own, app_args = argv[:argv.index('--')], argv[argv.index('--') + 1:]
    session = own[own.index('--session') + 1] if '--session' in own else None
    retries = int(own[own.index('--retries') + 1]) if '--retries' in own else 10
    # --set KEY=VALUE: environment of every run (E2A_REPAIR_PARTS=5, E2A_DELIVERY_DIR=...),
    # also written to REPRENDRE.cmd
    extra_env = dict(own[i + 1].split('=', 1) for i, a in enumerate(own) if a == '--set')
    env = dict(os.environ, E2A_RESUME_CHOICE='r', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    env.update(extra_env)
    if extra_env:
        log('variables : ' + ', '.join(f'{k}={v}' for k, v in extra_env.items()))
    attempt = 0
    while True:
        cmd = [sys.executable, '-u', os.path.join(ROOT, 'app.py')] + app_args
        if session:
            cmd += ['--session', session]
            write_resume_files(app_args, session, extra_env)
            log(f'session {session} (tentative {attempt + 1})')
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        state = {'session': session}
        # the output is copied by a thread: when app.py dies abruptly, its children
        # (multiprocessing manager, Whisper watcher) keep the pipe open and an EOF would
        # never come, the end of app.py itself is what counts
        reader = threading.Thread(target=copy_output, args=(proc, state, app_args, extra_env), daemon=True)
        reader.start()
        code = proc.wait()
        orphans = kill_orphans(proc.pid)
        if orphans:
            log(f'{orphans} processus orphelin(s) de la conversion arrete(s)')
        reader.join(timeout=15)
        session = state['session']
        if code == 0:
            log('conversion terminee')
            sys.exit(0)
        attempt += 1
        if session is None or attempt > retries:
            log(f'arret definitif (code {code}, {attempt} tentative(s))')
            sys.exit(code)
        log(f'arret anormal (code {code}), reprise dans 30 s…')
        time.sleep(30)


if __name__ == '__main__':
    main()
