#!/usr/bin/env python3
"""Capture the game's real collision geometry by letting the game compute it.

``tools/extract_geometry.mjs`` reimplements the client's model pipeline and gets
185 of 186 parts bit-identical. The remaining one, 159 ``StraightTilted``, is the
only part in the whole table that uses ``scale`` or ``quaternion`` -- no other
part exercises those two code paths -- and reading the source has not explained
the difference. Reimplementation was always the fragile approach; this replaces
it with observation.

The client hands its simulation worker every part's ``physicsShapeVertices`` in
the ``Init`` message, so that message *is* the ground truth. This serves your own
local copy of the game over HTTP with a small hook injected ahead of its scripts,
the hook posts the ``Init`` payload back, and the result is written in exactly the
shape ``tools/build_geometry.py`` already consumes::

    python tools/capture_geometry.py path/to/app.asar-contents -o geometry.json
    python tools/build_geometry.py geometry.json geometry.npz

Everything stays on 127.0.0.1 and nothing is uploaded. The game files are
Kodub's; keep them local and do not redistribute them.

Why a local server rather than the console: both ``SimulationManager`` instances
are constructed during app boot, so ``Init`` has already been sent by the time a
menu is on screen and a pasted snippet is far too late. And the desktop build
sets ``devTools: false``, so it has no console at all. Serving a copy with the
hook already in place is the only version of this that reliably wins the race.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import struct
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The game fetches its wasm with instantiateStreaming, which rejects anything
# not served as application/wasm.
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/javascript", ".js")

#: Injected ahead of the deferred bundles. Wraps Worker.postMessage, copies the
#: Init payload, and posts it back. It never modifies what the game sends.
HOOK = """
<script>
(function () {
  var sent = false;
  var original = Worker.prototype.postMessage;
  function toArray(v) {
    if (v == null) return null;
    return Array.from(v instanceof Float32Array ? v : new Float32Array(v));
  }
  function banner(text, colour) {
    var d = document.getElementById('capture-banner') || document.createElement('div');
    d.id = 'capture-banner';
    d.textContent = text;
    d.style.cssText = 'position:fixed;left:0;right:0;top:0;z-index:2147483647;'
      + 'padding:14px;background:' + colour + ';color:#fff;'
      + 'font:600 18px system-ui,sans-serif;text-align:center';
    if (!d.parentNode) (document.body || document.documentElement).appendChild(d);
  }
  Worker.prototype.postMessage = function (message) {
    try {
      if (!sent && message && message.messageType === 0 && message.trackParts) {
        sent = true;
        var payload = {
          version: message.version,
          massOffset: message.carMassOffset,
          carCollisionShapeVertices: toArray(message.carCollisionShapeVertices),
          parts: message.trackParts.map(function (p) {
            return {
              id: p.id,
              vertices: toArray(p.vertices),
              detector: p.detector == null ? null : {
                type: p.detector.type,
                center: Array.from(p.detector.center),
                size: Array.from(p.detector.size)
              },
              startOffset: p.startOffset == null ? null : Array.from(p.startOffset)
            };
          })
        };
        banner('Capturing ' + payload.parts.length + ' parts...', '#1d4ed8');
        fetch('/capture', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }).then(function (r) { return r.text(); })
          .then(function (t) {
            document.title = 'CAPTURED';
            banner(t + '  -  done, you can close this tab.', '#15803d');
          })
          .catch(function (e) {
            banner('Capture failed: ' + e, '#b91c1c');
          });
      }
    } catch (e) {
      banner('Capture hook error: ' + e, '#b91c1c');
    }
    return original.apply(this, arguments);
  };
})();
</script>
"""

#: The detector enum, which arrives numerically here but is written out by name
#: so the file matches what extract_geometry.mjs produces.
DETECTOR_NAMES = {0: "Checkpoint", 1: "Finish"}


def build_handler(root: Path, out: Path, done: threading.Event, names: dict[int, str]):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt, *args):  # quieter than the default
            if "/capture" in (self.path or ""):
                sys.stderr.write("  " + fmt % args + "\n")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (root / "index.html").read_text(encoding="utf-8")
                # The bundles are deferred, so a classic script placed before
                # them in <head> is guaranteed to run first.
                patched = html.replace("<head>", "<head>" + HOOK, 1).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(patched)))
                self.end_headers()
                self.wfile.write(patched)
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/capture":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            message = write_capture(payload, out, names)
            body = message.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

    return Handler


def write_capture(payload: dict, out: Path, names: dict[int, str]) -> str:
    """Convert the Init payload into extract_geometry.mjs's output format."""
    parts = []
    for entry in payload["parts"]:
        vertices = entry["vertices"]
        raw = struct.pack(f"<{len(vertices)}f", *vertices)
        detector = entry.get("detector")
        parts.append(
            {
                "id": entry["id"],
                "name": names.get(entry["id"], f"Part{entry['id']}"),
                "vertices": vertices,
                "detector": None if detector is None else {
                    "type": DETECTOR_NAMES.get(detector["type"], str(detector["type"])),
                    "center": detector["center"],
                    "size": detector["size"],
                },
                "startOffset": None if entry.get("startOffset") is None else {
                    "x": entry["startOffset"][0],
                    "y": entry["startOffset"][1],
                    "z": entry["startOffset"][2],
                },
                "checksum": hashlib.sha256(raw).hexdigest(),
            }
        )

    document = {
        "version": payload.get("version", "0.6.2"),
        "massOffset": payload["massOffset"],
        "carCollisionShapeVertices": payload["carCollisionShapeVertices"],
        "parts": parts,
    }
    out.write_text(json.dumps(document))

    total = sum(len(p["vertices"]) for p in parts)
    return f"captured {len(parts)} parts, {total:,} floats"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "game", type=Path,
        help="directory holding index.html and main.bundle.js (app.asar contents)",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("geometry.json"))
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not (args.game / "index.html").exists():
        print(f"no index.html in {args.game}")
        return 1

    # Reuse the names from a previous extraction if one is around; they are
    # cosmetic, but they make the output readable and diffable.
    names: dict[int, str] = {}
    for candidate in (Path("geometry.npz"), args.out.with_suffix(".npz")):
        if candidate.exists():
            try:
                import numpy as np
                with np.load(candidate) as data:
                    manifest = json.loads(bytes(data["manifest"]).decode("utf-8"))
                names = {p["id"]: p["name"] for p in manifest["parts"]}
                break
            except Exception:
                pass

    done = threading.Event()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), build_handler(args.game, args.out, done, names)
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {args.game} at {url}")
    print("open it, wait for the main menu, and the capture happens by itself.")
    print("nothing leaves this machine.\n")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not done.wait(timeout=300):
            print("\ntimed out after 5 minutes with no capture.")
            print("did the game reach its main menu? check the browser console.")
            return 1
    except KeyboardInterrupt:
        return 1
    finally:
        server.shutdown()

    print(f"\nwrote {args.out}")
    print(f"next:  python tools/build_geometry.py {args.out} geometry.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
