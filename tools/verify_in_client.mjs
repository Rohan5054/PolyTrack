#!/usr/bin/env node
// Run the game's own simulation worker over our recordings.
//
//   node tools/verify_in_client.mjs --game ../game-extracted \
//        --payload runs/client_check.json
//
// `polytrack/verify.py` mirrors the client's Verify loop and reproduces eight
// real recordings frame-exactly, which validates the simulator. For a lap this
// project *generated* the target frame is our own search's claim, so our PASS
// means "these inputs reproduce that frame here" and the step to "the game
// agrees" is an inference. This measures it instead: the client's worker
// decodes the track code and deserialises the recording with its own code, so a
// PASS exercises our decoder and our recording codec too.
//
// The client's files are read from wherever they were extracted and nothing
// from them is copied into this repo.

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { pathToFileURL } from "node:url";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const gameDir = path.resolve(arg("game", "../game-extracted"));
const payloadPath = path.resolve(arg("payload", "runs/client_check.json"));
const workerPath = path.join(gameDir, "simulation_worker.bundle.js");
const wasmPath = path.join(gameDir, "polytrack_physics.wasm");

for (const p of [workerPath, wasmPath, payloadPath]) {
  if (!fs.existsSync(p)) {
    console.error(`  missing: ${p}`);
    process.exit(2);
  }
}

// Base64 -> Float32Array, copied rather than viewed: a Buffer's byteOffset is
// not guaranteed to be four-aligned and a misaligned view throws.
function f32(b64) {
  const bytes = Buffer.from(b64, "base64");
  const out = new Float32Array(bytes.byteLength / 4);
  Buffer.from(out.buffer).set(bytes);
  return out;
}

const payload = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
const wasmBinary = new Uint8Array(fs.readFileSync(wasmPath));

// --- the worker's environment ---------------------------------------------
const results = [];
globalThis.self = globalThis;

globalThis.location = { href: pathToFileURL(gameDir + path.sep).href };
globalThis.postMessage = (msg) => results.push(msg);
// A worker's global scope predefines these, so the bundle's top-level
// `onmessage = ...` is an assignment to an existing property. Node's global has
// no such property, and the bundle runs under "use strict", where assigning to
// an undeclared identifier is a ReferenceError rather than an implicit global.
globalThis.onmessage = null;
globalThis.onerror = null;
globalThis.onmessageerror = null;
// A DedicatedWorkerGlobalScope is an EventTarget. Node's global is not, and the
// bundle registers listeners on itself at load time. Only "message" matters
// here, and it is delivered through the onmessage property below.
const listeners = new Map();
globalThis.addEventListener = (type, fn) => {
  if (!listeners.has(type)) listeners.set(type, []);
  listeners.get(type).push(fn);
};
globalThis.removeEventListener = (type, fn) => {
  const fns = listeners.get(type);
  if (fns) fns.splice(fns.indexOf(fn) >>> 0, 1);
};
globalThis.dispatchEvent = (event) => {
  for (const fn of listeners.get(event?.type) ?? []) fn(event);
  return true;
};
globalThis.importScripts = (rel) => {
  const file = path.join(gameDir, rel);
  vm.runInThisContext(fs.readFileSync(file, "utf8"), { filename: file });
  // The glue fetches its .wasm over HTTP unless handed the bytes. It is called
  // with no arguments the statement after this returns, so wrapping has to
  // happen here rather than after the bundle has been evaluated.
  if (typeof globalThis.PolyTrackPhysics === "function") {
    const inner = globalThis.PolyTrackPhysics;
    globalThis.PolyTrackPhysics = (opts = {}) => inner({ ...opts, wasmBinary });
  }
};

// The bundled js-sha256 sniffs for Node and switches to require("crypto") and
// require("buffer").Buffer -- both stubbed out as empty modules when webpack
// built this for the browser, so that path throws on the very first hash. Its
// JS_SHA256_NO_NODE_JS opt-out cannot be reached from here: the flag is read
// off `s ? window : {}`, which is a fresh empty object when there is no window,
// and that read happens before the variable is repointed at the global. Hiding
// `process` for the duration of the evaluation takes the browser branch
// instead, and is narrower than faking `window`, which would also change which
// paths three.js takes.
const realProcess = globalThis.process;
globalThis.process = undefined;
try {
  vm.runInThisContext(fs.readFileSync(workerPath, "utf8"), { filename: workerPath });
} catch (err) {
  globalThis.process = realProcess;
  // The bundle is one 300 KB line, so letting this propagate prints the whole
  // file as the error's source context.
  console.error(`  the worker failed to load: ${err.name}: ${err.message}`);
  process.exit(2);
} finally {
  globalThis.process = realProcess;
}

if (typeof globalThis.onmessage !== "function") {
  console.error("  the worker did not install an onmessage handler");
  process.exit(2);
}

// Messages sent before the module finishes booting are queued and drained in
// order, so everything can go in now.
const init = payload.init;
globalThis.onmessage({
  data: {
    messageType: 0, // Init
    version: init.version,
    isRealtime: false,
    carMassOffset: init.carMassOffset,
    carCollisionShapeVertices: f32(init.carCollisionShapeVertices),
    trackParts: init.trackParts.map((p) => ({
      id: p.id,
      vertices: f32(p.vertices),
      detector: p.detector,
      startOffset: p.startOffset,
    })),
  },
});

payload.laps.forEach((lap, index) => {
  globalThis.onmessage({
    data: {
      messageType: 1, // Verify
      carId: index + 1,
      trackData: lap.trackData,
      carRecording: lap.carRecording,
      mountainVertices: f32(lap.mountainVertices),
      mountainOffset: lap.mountainOffset,
      targetFrames: lap.targetFrames,
    },
  });
});

const deadline = Date.now() + 15 * 60 * 1000;
while (results.filter((m) => m.messageType === 8).length < payload.laps.length) {
  if (Date.now() > deadline) {
    console.error("  timed out waiting for VerifyResult");
    process.exit(2);
  }
  await new Promise((r) => setTimeout(r, 200));
}

console.log(`\n  the client's own simulation worker, version ${init.version}`);
console.log(`  ${path.relative(process.cwd(), workerPath)}\n`);
console.log(`  ${"artifact".padEnd(32)}${"frames".padStart(8)}   client verdict`);
let failures = 0;
for (const m of results.filter((m) => m.messageType === 8)) {
  const lap = payload.laps[m.carId - 1];
  const verdict = m.result ? "PASS" : "FAIL";
  if (!m.result) failures++;
  console.log(
    `  ${lap.label.padEnd(32)}${String(lap.targetFrames).padStart(8)}   ${verdict}`
  );
}
console.log(
  failures
    ? `\n  ${failures} lap(s) the game does not agree with.`
    : `\n  The game's own verifier agrees with every lap.`
);
process.exit(failures ? 1 : 0);
