#!/usr/bin/env python3
"""Fuzzy Grand Prix Race Control: a public leaderboard for JSON fuzzy drivers.

Run from the repository root::

    python examples/race_control/server.py

Then open http://127.0.0.1:8080.  Anyone with the URL can enter a name and a driver
JSON (see ``example_driver.json``); each entry is queued, raced once on every
circuit by :mod:`grand_prix`, and ranked by total lap time.

Submissions are data, never code: a spec can only choose membership functions,
rules and operators from a fixed vocabulary, with size limits, so it is safe to
expose the page publicly (for example with ``ngrok http 8080``).  The only
privileged action, deleting an entry, needs the instructor key printed at start-up.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import queue
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import grand_prix as gp
from fuzzy_torch.config import ConfigError
from fuzzy_torch.envs import make_track

HERE = Path(__file__).resolve().parent
WEB = HERE / 'web'
DEFAULT_STORE = HERE / 'leaderboard.json'
MAX_BODY = 256 * 1024
MAX_NAME = 24
MAX_QUEUE = 40
# Tiles show mini-maps for this many leaders; the rest of the table ships without paths.
TILE_COUNT = 12


def _name_key(name: str) -> str:
    """Names are matched case- and spacing-insensitively, so 'Ana ' is 'ana'."""
    return ' '.join(name.split()).casefold()


def _sort_key(entry: dict[str, Any]):
    total = entry.get('total')
    return (total is None, total if total is not None else math.inf,
            -entry.get('finished', 0), -entry.get('completion', 0.0), entry['submitted_at'])


class RaceControl:
    def __init__(self, store: Path, workers: int = 1) -> None:
        self.store = store
        self.workers = workers
        # Laps run in separate processes: the simulation loop is Python-bound, so
        # threads would serialise on the GIL.
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.entries: dict[str, dict[str, Any]] = {}
        self.version = 0
        self.control_key = ''
        self.tracks = self._tracks()
        self._load()

    # ------------------------------------------------------------- storage

    def _load(self) -> None:
        if not self.store.exists():
            return
        try:
            items = json.loads(self.store.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"could not read {self.store}: {exc}")
        for entry in items:
            # A restart loses the queue; anything unfinished is raced again.
            if entry['status'] in ('queued', 'racing'):
                entry.update(status='queued', track=None)
                self.queue.put(entry['id'])
            self.entries[entry['id']] = entry

    def _save(self) -> None:
        tmp = self.store.with_suffix('.tmp')
        tmp.write_text(json.dumps(list(self.entries.values())))
        os.replace(tmp, self.store)

    def _changed(self) -> None:
        """Call with the lock held after any change spectators should see."""
        self.version += 1
        self._save()

    @staticmethod
    def _tracks() -> dict[str, Any]:
        out = {}
        for name in gp.TRACKS:
            track = make_track(name)
            out[name] = {'points': [[round(float(x), 3), round(float(y), 3)] for x, y in track.points],
                         'width': track.width, 'length': round(track.length, 2)}
        return out

    # ---------------------------------------------------------- submission

    def submit(self, name: Any, spec: Any) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ConfigError('name: please enter your name')
        name = ' '.join(name.split())
        if len(name) > MAX_NAME:
            raise ConfigError(f'name: at most {MAX_NAME} characters')
        # Full validation up front, so mistakes come back immediately, not after the queue.
        gp.build_driver(spec)
        with self.lock:
            pending = [e for e in self.entries.values() if e['status'] in ('queued', 'racing')]
            if len(pending) >= MAX_QUEUE:
                raise OverflowError('The queue is full; try again in a minute.')
            if any(_name_key(e['name']) == _name_key(name) for e in pending):
                raise OverflowError(f"{name} already has a run waiting; wait for it to finish.")
            entry = {
                'id': uuid.uuid4().hex[:12], 'name': name,
                'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'status': 'queued', 'track': None, 'error': None,
                'description': str(spec.get('description', ''))[:200],
                'summary': gp.describe(spec), 'spec': spec,
                'laps': {}, 'total': None, 'finished': 0, 'completion': 0.0,
            }
            self.entries[entry['id']] = entry
            self._changed()
        self.queue.put(entry['id'])
        return {'id': entry['id'], 'position': len(pending) + 1}

    def delete(self, entry_id: str) -> bool:
        with self.lock:
            if self.entries.pop(entry_id, None) is None:
                return False
            self._changed()
            return True

    # -------------------------------------------------------------- worker

    def worker(self) -> None:
        """Race queued entries one at a time; ``workers`` of these run side by side."""
        while True:
            entry_id = self.queue.get()
            with self.lock:
                entry = self.entries.get(entry_id)
                if entry is None or entry['status'] != 'queued':
                    continue
                entry['status'] = 'racing'
                spec = entry['spec']
                self._changed()

            def on_track(track: str) -> None:
                with self.lock:
                    entry['track'] = track
                    self.version += 1

            try:
                laps = {}
                for track in gp.TRACKS:
                    on_track(track)
                    laps[track] = self.pool.submit(gp.race_track, spec, track).result()
                update = {'status': 'done', 'track': None, 'laps': laps, **gp.score(laps)}
            except Exception as exc:  # a bad spec must not take the worker down
                traceback.print_exc()
                update = {'status': 'error', 'track': None, 'error': str(exc)}
            with self.lock:
                if entry_id in self.entries:
                    entry.update(update)
                    entry['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                    self._changed()

    # ------------------------------------------------------------ payloads

    def _queue_position(self, entry: dict[str, Any]) -> int | None:
        if entry['status'] not in ('queued', 'racing'):
            return None
        pending = sorted((e for e in self.entries.values() if e['status'] in ('queued', 'racing')),
                         key=lambda e: (e['status'] != 'racing', e['submitted_at']))
        return next(i for i, e in enumerate(pending) if e['id'] == entry['id'])

    @staticmethod
    def _public(entry: dict[str, Any], with_paths: bool) -> dict[str, Any]:
        laps = {t: {k: v for k, v in lap.items() if with_paths or k != 'path'}
                for t, lap in entry['laps'].items()}
        keep = ('id', 'name', 'submitted_at', 'status', 'track', 'error', 'description',
                'summary', 'total', 'finished', 'completion')
        return {**{k: entry.get(k) for k in keep}, 'laps': laps}

    def leaderboard(self) -> dict[str, Any]:
        with self.lock:
            done = [e for e in self.entries.values() if e['status'] == 'done']
            attempts: dict[str, int] = {}
            best: dict[str, dict[str, Any]] = {}
            for entry in sorted(done, key=_sort_key):
                key = _name_key(entry['name'])
                attempts[key] = attempts.get(key, 0) + 1
                best.setdefault(key, entry)
            standings = []
            for i, entry in enumerate(sorted(best.values(), key=_sort_key)):
                row = self._public(entry, with_paths=False)
                row['attempts'] = attempts[_name_key(entry['name'])]
                if i < TILE_COUNT:
                    # Every other sample is plenty for a 200-pixel mini-map.
                    row['paths'] = {t: [p[:2] for p in lap['path'][::2]]
                                    for t, lap in entry['laps'].items()}
                standings.append(row)

            records = {}
            for track in gp.TRACKS:
                laps = [(e['laps'][track]['time'], e['submitted_at'], e['name'], e['id']) for e in done
                        if track in e['laps'] and e['laps'][track]['time'] is not None]
                if laps:
                    # Ties go to whoever set the time first.
                    t, _, who, entry_id = min(laps)
                    records[track] = {'time': t, 'name': who, 'id': entry_id}

            pending = sorted((e for e in self.entries.values() if e['status'] in ('queued', 'racing')),
                             key=lambda e: (e['status'] != 'racing', e['submitted_at']))
            recent = sorted((e for e in self.entries.values() if e['status'] in ('done', 'error')),
                            key=lambda e: e.get('finished_at', ''), reverse=True)[:8]
            return {
                'version': self.version,
                'standings': standings,
                'records': records,
                'queue': [{'name': e['name'], 'status': e['status'], 'track': e['track']}
                          for e in pending],
                'recent': [{'id': e['id'], 'name': e['name'], 'status': e['status'],
                            'total': e.get('total'), 'finished': e.get('finished', 0),
                            'error': e.get('error'), 'at': e.get('finished_at')} for e in recent],
                'runs': len(done),
            }

    def entry(self, entry_id: str) -> dict[str, Any] | None:
        with self.lock:
            entry = self.entries.get(entry_id)
            if entry is None:
                return None
            return {**self._public(entry, with_paths=True), 'spec': entry['spec'],
                    'position': self._queue_position(entry), 'version': self.version}


CONTROL: RaceControl


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def _json(self, status: int, payload: Any, etag: str | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        if etag:
            self.send_header('ETag', etag)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {'error': message})

    def _authorised(self, query: dict[str, list[str]]) -> bool:
        key = query.get('key', [''])[0]
        return bool(CONTROL.control_key) and secrets.compare_digest(key, CONTROL.control_key)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/leaderboard':
            etag = f'"v{CONTROL.version}"'
            if self.headers.get('If-None-Match') == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header('ETag', etag)
                self.end_headers()
                return
            return self._json(HTTPStatus.OK, CONTROL.leaderboard(), etag)
        if url.path == '/api/tracks':
            return self._json(HTTPStatus.OK, {'sample_dt': gp.SAMPLE_DT, 'tracks': CONTROL.tracks})
        if url.path == '/api/example':
            # Sent verbatim: the file's hand-made layout is easier to edit than json.dumps.
            body = (HERE / 'example_driver.json').read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path.startswith('/api/entries/'):
            entry = CONTROL.entry(url.path.rsplit('/', 1)[-1])
            if entry is None:
                return self._error(HTTPStatus.NOT_FOUND, 'No such entry')
            return self._json(HTTPStatus.OK, entry)
        if url.path == '/api/admin':
            return self._json(HTTPStatus.OK, {'admin': self._authorised(parse_qs(url.query))})
        if url.path.startswith('/api/'):
            return self._error(HTTPStatus.NOT_FOUND, 'Unknown endpoint')
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != '/api/submit':
            return self._error(HTTPStatus.NOT_FOUND, 'Unknown endpoint')
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               f'Submissions are limited to {MAX_BODY // 1024} KB')
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ConfigError('expected {"name": ..., "spec": {...}}')
            result = CONTROL.submit(body.get('name'), body.get('spec'))
        except json.JSONDecodeError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, f'Not valid JSON: {exc}')
        except ConfigError as exc:
            return self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
        except OverflowError as exc:
            return self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
        return self._json(HTTPStatus.ACCEPTED, result)

    def do_DELETE(self):
        url = urlparse(self.path)
        if not url.path.startswith('/api/entries/'):
            return self._error(HTTPStatus.NOT_FOUND, 'Unknown endpoint')
        if not self._authorised(parse_qs(url.query)):
            return self._error(HTTPStatus.FORBIDDEN, 'The instructor key is required')
        if not CONTROL.delete(url.path.rsplit('/', 1)[-1]):
            return self._error(HTTPStatus.NOT_FOUND, 'No such entry')
        return self._json(HTTPStatus.OK, {'deleted': True})

    def log_message(self, format, *args):
        # Leaderboard polling would drown everything else out.
        if '/api/leaderboard' not in self.requestline:
            print('[race-control] ' + format % args)


def main() -> None:
    global CONTROL
    parser = argparse.ArgumentParser(description='Run the Fuzzy Grand Prix leaderboard')
    parser.add_argument('--host', default='127.0.0.1',
                        help='use 0.0.0.0 to share on the classroom LAN')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE,
                        help='results file; use a fresh one per lecture')
    parser.add_argument('--control-key', help='stable instructor key (default: random)')
    parser.add_argument('--workers', type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                        help='entries raced in parallel (default: half the cores, at most 4)')
    args = parser.parse_args()
    CONTROL = RaceControl(args.store, max(1, args.workers))
    CONTROL.control_key = args.control_key or secrets.token_urlsafe(12)
    for _ in range(CONTROL.workers):
        threading.Thread(target=CONTROL.worker, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Fuzzy Grand Prix ready at http://{args.host}:{args.port}')
    print(f'Results file: {args.store} · {CONTROL.workers} evaluation worker(s)')
    print(f'Instructor URL (can delete entries): '
          f'http://127.0.0.1:{args.port}/?key={CONTROL.control_key}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nRace control stopped.')


if __name__ == '__main__':
    main()
