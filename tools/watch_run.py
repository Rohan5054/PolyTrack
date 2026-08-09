#!/usr/bin/env python3
"""Drive a track with a trained agent, then watch the run back in the game.

    python tools/watch_run.py runs/<name>/agent.pt --track "Winter 5" \
        --game path/to/app.asar-contents

Two halves. The first drives the track greedily through the verified backend and
encodes the inputs into PolyTrack's own recording format -- the same format the
game stores its ghosts in, so what comes out is a real replay and not a
visualisation of one. It is checked before anything else happens: the recording
is replayed back through the acceptance gate and has to land on the same finish
frame, because a ghost that does not reproduce is not the run.

The second serves your local copy of the game over 127.0.0.1 with the record
already installed in local storage, so the track shows the run as its best time
and you can watch the replay in the actual game with the actual car.

**This never touches your real save.** The served copy is ``http://127.0.0.1``,
a different origin from the Electron app's ``file://``, so it gets its own empty
local storage. Your own personal bests are in the app and are not read, written
or reachable from here.

**And it never uploads.** The injected hook blocks any request to Kodub's
servers outright. Bot and TAS runs stay off the human leaderboard -- Linesight
states the rule for Trackmania and the PolyTrack community draws the same line.
The block is belt and braces on top of the fact that a fresh origin has no
session to upload with.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/javascript", ".js")

#: Installs the record and severs the upload path. ``__RECORDS__`` becomes a
#: list of ``{trackId, frames, recording}``.
#:
#: The record cannot simply be written: ``loadRecord`` in main.bundle.js reads
#: ``tokenHash`` out of the stored JSON and returns null unless it equals the
#: *current profile's* hash, which is ``sha256(profile.token)`` over a token the
#: game generates randomly on first boot. So the hook has to wait for the
#: profile to exist, read its token, and derive the hash -- js-sha256 emits
#: lowercase hex of the UTF-8 bytes, which is exactly what crypto.subtle gives.
#: 127.0.0.1 counts as a secure context, so subtle is available.
HOOK = """
<script>
(function () {
  var records = __RECORDS__;
  var PREFIX = 'polytrack_v5_prod_';

  function hex(buffer) {
    return Array.from(new Uint8Array(buffer))
      .map(function (b) { return b.toString(16).padStart(2, '0'); }).join('');
  }

  function install() {
    var slotRaw, slot, profileRaw, token;
    try {
      slotRaw = localStorage.getItem(PREFIX + 'user_slot');
      slot = slotRaw == null ? 0 : JSON.parse(slotRaw);
      profileRaw = localStorage.getItem(PREFIX + 'user_' + slot);
      if (!profileRaw) return Promise.resolve(false);
      token = JSON.parse(profileRaw).token;
      if (typeof token !== 'string' || !token.length) return Promise.resolve(false);
    } catch (e) { return Promise.resolve(false); }

    return crypto.subtle
      .digest('SHA-256', new TextEncoder().encode(token))
      .then(function (digest) {
        var tokenHash = hex(digest);
        records.forEach(function (r) {
          var value = JSON.stringify({
            uploadId: null, tokenHash: tokenHash,
            frames: r.frames, recording: r.recording
          });
          // Both determinism buckets: which one the game reads depends on its
          // own self-test, and writing the other costs nothing.
          ['default', 'undeterministic'].forEach(function (kind) {
            try {
              localStorage.setItem(
                PREFIX + 'record_' + slot + '_' + kind + '_' + r.trackId, value
              );
            } catch (e) {}
          });
        });
        return true;
      })
      .catch(function () { return false; });
  }

  function attempt(tries) {
    return install().then(function (ok) {
      if (ok) return true;
      if (tries <= 0) return false;
      return new Promise(function (resolve) {
        setTimeout(function () { resolve(attempt(tries - 1)); }, 200);
      });
    });
  }

  attempt(150).then(function (ok) {
    if (!ok) {
      banner('Could not install the run: the game never created a profile.',
             '#b91c1c');
      return;
    }
    // The profile did not exist on a first visit, so the game booted without
    // the record. One reload and it starts with it already in place.
    if (!sessionStorage.getItem('watch_run_reloaded')) {
      sessionStorage.setItem('watch_run_reloaded', '1');
      location.reload();
      return;
    }
    banner('Agent run installed \\u2014 open the track and it is the record. '
           + 'Uploads are blocked; this is not your real save.', '#1d4ed8');
  });

  // Bot runs do not go on the human leaderboard. A fresh origin has no session
  // to upload with, but do not rely on that alone.
  var blocked = /vps\\.kodub\\.com|leaderboard|\\/api\\//i;
  var fetch0 = window.fetch;
  window.fetch = function (input) {
    var url = (typeof input === 'string') ? input : (input && input.url) || '';
    if (blocked.test(url)) {
      console.warn('[watch_run] blocked upload to', url);
      return Promise.reject(new Error('uploads disabled'));
    }
    return fetch0.apply(this, arguments);
  };
  var open0 = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url) {
    if (blocked.test(String(url))) {
      console.warn('[watch_run] blocked upload to', url);
      throw new Error('uploads disabled');
    }
    return open0.apply(this, arguments);
  };

  function banner(text, colour) {
    function show() {
      var d = document.createElement('div');
      d.textContent = text;
      d.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:2147483647;'
        + 'padding:10px;background:' + colour + ';color:#fff;'
        + 'font:600 14px system-ui,sans-serif;text-align:center';
      (document.body || document.documentElement).appendChild(d);
      setTimeout(function () { d.remove(); }, 15000);
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', show);
    } else { show(); }
  }
})();
</script>
"""


def drive(checkpoint: Path, library_path: Path, track_name: str) -> dict:
    """Drive one track greedily and return a verified recording of it."""
    from polytrack.env import PolyTrackEnv
    from polytrack.learning.iqn import IQNAgent
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.tas import format_time
    from polytrack.verify import verify

    import torch

    torch.set_num_threads(2)

    entry = TrackLibrary.load(library_path).by_name(track_name)
    assets = GeometryAssets.load()
    agent = IQNAgent.load(checkpoint)

    env = PolyTrackEnv(
        track=entry.line,
        backend=WasmPhysics(entry.track_code(), assets=assets),
        time_limit_ms=entry.par_frames * 3,
        par_frames=entry.par_frames,
    )
    observation, _ = env.reset()
    info: dict = {}
    while True:
        observation, _, terminated, truncated, info = env.step(
            agent.act(observation, epsilon=0.0)
        )
        if terminated or truncated:
            break

    finished = bool(info.get("finished"))
    frames = env.elapsed_frames
    recording = Recording.from_masks(env.input_masks)

    print(f"  {entry.name}: {format_time(frames)} "
          f"({'finished' if finished else info.get('failure', 'did not finish')}), "
          f"par {format_time(entry.par_frames)}")

    # A ghost that does not reproduce is not the run. Same check as the gate.
    if finished:
        fresh = WasmPhysics(entry.track_code(), assets=assets)
        result = verify(fresh, recording, frames)
        print(f"  replay check: {result.describe()}")
        if not result.passed:
            raise SystemExit(
                "the encoded recording does not reproduce the run -- refusing "
                "to install a ghost that is not what the agent drove"
            )

    return {
        "track_id": entry.track_code().track_id,
        "name": entry.name,
        "frames": frames,
        "finished": finished,
        "recording": recording.serialize(),
        "tas": env.export_tas().to_text(),
    }


def build_handler(root: Path, hook: str):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *_):  # quiet
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (root / "index.html").read_text(encoding="utf-8")
                # The bundles are deferred, so a classic script in <head> runs
                # first and the record is in place before the game reads it.
                body = html.replace("<head>", "<head>" + hook, 1).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--track", required=True)
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument(
        "--game", type=Path, default=None,
        help="directory with index.html from your app.asar. Without it the run "
             "is exported to files and nothing is served",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/exported"))
    parser.add_argument("--port", type=int, default=8732)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    run = drive(args.checkpoint, args.library, args.track)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = run["name"].replace(" ", "_").lower()
    (args.out / f"{stem}.tas").write_text(run["tas"], encoding="utf-8")
    (args.out / f"{stem}.json").write_text(json.dumps({
        k: v for k, v in run.items() if k != "tas"
    }, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out / (stem + '.tas')}")
    print(f"  wrote {args.out / (stem + '.json')}")

    if not run["finished"]:
        print("\n  The run did not finish, so there is no time to install. The")
        print("  .tas still holds exactly what it drove.")
        return 1
    if args.game is None:
        print("\n  Pass --game path/to/app.asar-contents to watch it in the game.")
        return 0
    if not (args.game / "index.html").exists():
        print(f"\n  no index.html in {args.game}")
        return 2

    # The key needs the profile slot and the tokenHash, neither of which exists
    # until the game boots, so the hook assembles it in the browser.
    hook = HOOK.replace("__RECORDS__", json.dumps([{
        "trackId": run["track_id"],
        "frames": run["frames"],
        "recording": run["recording"],
    }]))

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), build_handler(args.game, hook)
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  serving your copy of the game at {url}")
    print(f"  open '{run['name']}' and watch the replay. Ctrl+C here when done.")
    print("  this is a separate origin: your real saves are untouched, and")
    print("  uploads to Kodub's servers are blocked.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("  stopped")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
