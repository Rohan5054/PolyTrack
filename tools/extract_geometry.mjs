// Extract every track part's physicsShapeVertices exactly as PolyTrack does,
// and verify each against the SHA-256 the game carries for that purpose.
//
// This is the one input the physics module needs that is not described in the
// game's JavaScript: `addTrackPartConfiguration` wants each part's collision
// mesh, and those live in Draco-compressed .glb models. Get a single float
// wrong and replays desync silently, hundreds of frames in.
//
// Replicating the pipeline in Python was the alternative. Using real three.js
// is what makes bit-exactness achievable, since mergeGeometries / toNonIndexed
// vertex ordering and the float32 rounding inside applyMatrix4 all have to
// match. The game's own checksums then prove it did.
//
// Setup and use:
//
//   npm install three @gltf-transform/core @gltf-transform/extensions draco3dgltf
//   node tools/extract_geometry.mjs <main.bundle.js> <models/> geometry.json
//   python tools/build_geometry.py geometry.json geometry.npz
//
// Both inputs come out of the desktop build's resources/app.asar (see
// tools/extract_asar.py). Keep them local; do not redistribute them.
//
// Known outlier: part 159 StraightTilted does not match its checksum. It is
// the only part built through the quaternion path, and every ordering of
// scale / rotate / translate was tried. Note that the game treats a checksum
// mismatch as a console.error and carries on, so a stale entry in its own
// table is as consistent with the evidence as a bug here. Unresolved --
// launch the game and read the console to find out which.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { NodeIO } from "@gltf-transform/core";
import { KHRDracoMeshCompression } from "@gltf-transform/extensions";
import draco3d from "draco3dgltf";

const [BUNDLE, MODELS, OUT] = process.argv.slice(2);
const GLBS = ["blocks", "pillar", "planes", "road", "road_wide", "signs",
              "wall_track", "car"];

// ---------------------------------------------------------------- bundle ---
const src = readFileSync(BUNDLE, "utf8");

/** Match brackets from `from`, ignoring bracket characters inside strings. */
function matchBracket(text, from, open, close) {
  let depth = 0, inString = false;
  for (let i = from; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === open) depth++;
    else if (ch === close && --depth === 0) return i;
  }
  throw new Error("unbalanced brackets");
}

/** The TS numeric enum emit: e[e.Straight=0]="Straight", ... */
function readEnum(anchor) {
  const at = src.indexOf(anchor);
  if (at < 0) throw new Error(`enum anchor not found: ${anchor}`);
  const start = src.lastIndexOf("function(", at);
  const body = src.slice(start, matchBracket(src, src.indexOf("{", start), "{", "}") + 1);
  const out = {};
  for (const m of body.matchAll(/\.(\w+)=(\d+)\]="\1"/g)) out[m[1]] = Number(m[2]);
  return out;
}

const PartId = readEnum('e[e.Straight=0]="Straight"');

const tableAt = src.indexOf('const u=[new d("6d94d798');
const tableSrc = src.slice(
  src.indexOf("[", tableAt),
  matchBracket(src, src.indexOf("[", tableAt), "[", "]") + 1,
);

// Stubs standing in for the bundle's minified module references. Colors do not
// affect vertex positions, so one variant is enough -- the game itself keeps
// only the first variant's geometry as the physics shape.
const nameProxy = new Proxy({}, { get: (_, k) => String(k) });
const parts = new Function(
  "d", "r", "s", "o", "i", "c", "l", "h",
  `return ${tableSrc};`,
)(
  class {
    constructor(checksum, category, id, models, colors, tiles, detector = null, startOffset = null) {
      Object.assign(this, { checksum, category, id, models, detector, startOffset });
    }
  },
  { A: nameProxy },                              // categories
  { A: PartId },                                 // part ids
  { A: nameProxy },                              // detector types
  // three.js maths types, minified. The table uses exactly three of them, and
  // one entry builds its quaternion via setFromEuler, so these have to be the
  // real classes rather than plain records.
  { Pq0: THREE.Vector3, PTz: THREE.Quaternion, O9p: THREE.Euler },
  [{ id: 0, colors: {} }],                       // one colour variant
  [{ id: 0, colors: {} }],
  [{ id: 0, colors: {} }],
);

// ---------------------------------------------------------------- models ---
const io = new NodeIO()
  .registerExtensions([KHRDracoMeshCompression])
  .registerDependencies({
    "draco3d.decoder": await draco3d.createDecoderModule(),
    "draco3d.encoder": await draco3d.createEncoderModule(),
  });

const scenes = new Map();
for (const name of GLBS) {
  const doc = await io.read(`${MODELS}/${name}.glb`);
  doc.createExtension(KHRDracoMeshCompression).dispose();
  const glb = await io.writeBinary(doc);
  const gltf = await new GLTFLoader().parseAsync(
    glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength), "",
  );
  scenes.set(gltf.scene.name, gltf.scene);
}

/** The game's `s(scene, mesh, flipX, flipY, flipZ, offset, scale, quaternion)`. */
function partGeometries(sceneName, meshName, opt = {}) {
  const scene = scenes.get(sceneName);
  if (scene == null) throw new Error(`no scene "${sceneName}"`);
  const object = scene.getObjectByName(meshName);
  if (object == null) throw new Error(`no object "${meshName}" in "${sceneName}"`);

  let geometries;
  if (object.isMesh) {
    const geometry = object.geometry.clone();
    object.updateMatrixWorld(true);
    geometry.applyMatrix4(object.matrix);
    geometries = [geometry];
  } else {
    // Note: only the group's matrix is applied, not each child's.
    geometries = object.children.map((child) => child.geometry.clone());
    object.updateMatrixWorld(true);
    for (const geometry of geometries) geometry.applyMatrix4(object.matrix);
  }

  const { flipX = false, flipY = false, flipZ = false } = opt;
  const offset = opt.offset ?? null, scale = opt.scale ?? null;
  const quaternion = opt.quaternion ?? null;

  // A flipY part is mirrored about its own top, so its height is needed first.
  let top = -Infinity;
  if (flipY) {
    for (const geometry of geometries) {
      const p = geometry.attributes.position.array;
      for (let i = 0; i < p.length; i += 3) top = Math.max(top, p[i + 1]);
    }
  }

  for (const geometry of geometries) {
    if (scale != null) geometry.scale(scale.x, scale.y, scale.z);
    if (quaternion != null) {
      geometry.applyQuaternion(
        new THREE.Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w),
      );
    }
    geometry.scale(flipX ? -1 : 1, flipY ? -1 : 1, flipZ ? -1 : 1);

    if (flipX || flipY || flipZ) {
      // Mirroring inverts winding; swap two corners of every triangle back.
      const index = geometry.index;
      if (index != null) {
        for (let i = 0; i < index.count; i += 3) {
          const a = index.getX(i), b = index.getX(i + 1), c = index.getX(i + 2);
          index.setXYZ(i, a, c, b);
        }
      } else {
        const p = geometry.attributes.position;
        for (let i = 0; i < p.count; i += 3) {
          const bx = p.getX(i + 1), by = p.getY(i + 1), bz = p.getZ(i + 1);
          const cx = p.getX(i + 2), cy = p.getY(i + 2), cz = p.getZ(i + 2);
          p.setXYZ(i + 1, cx, cy, cz);
          p.setXYZ(i + 2, bx, by, bz);
        }
      }
    }

    if (offset != null) {
      geometry.translate(offset.x, offset.y + (flipY ? top : 0), offset.z);
    } else if (flipY) {
      geometry.translate(0, top, 0);
    }
  }
  return geometries;
}

const sha256 = (f32) =>
  createHash("sha256")
    .update(Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength))
    .digest("hex");

// ------------------------------------------------------------- extraction ---
const results = [];
let matched = 0, failed = 0;
for (const part of parts) {
  const name = Object.keys(PartId).find((k) => PartId[k] === part.id) ?? "?";
  let digest, vertices;
  try {
    const pieces = [];
    for (const [sceneName, meshName, opt] of part.models) {
      pieces.push(...partGeometries(sceneName, meshName, opt ?? {}));
    }
    const merged = mergeGeometries(pieces, true).toNonIndexed();
    vertices = new Float32Array(merged.attributes.position.array);
    digest = sha256(vertices);
  } catch (error) {
    console.log(`  FAIL  ${String(part.id).padStart(3)} ${name}: ${error.message}`);
    failed++;
    continue;
  }
  const ok = digest === part.checksum;
  ok ? matched++ : failed++;
  if (!ok) {
    console.log(`  MISMATCH ${String(part.id).padStart(3)} ${name}`);
    console.log(`      got ${digest}`);
    console.log(`      exp ${part.checksum}`);
  }
  results.push({
    id: part.id, name, vertices: Array.from(vertices),
    detector: part.detector, startOffset: part.startOffset, checksum: digest,
  });
}

// Car hull and mass offset, from the same source of truth.
const carScene = scenes.get("Scene");
const collision = carScene.getObjectByName("Collision");
collision.updateMatrixWorld(true);
collision.geometry.applyMatrix4(collision.matrix.clone());
const carHull = new Float32Array(
  Array.from(collision.geometry.toNonIndexed().attributes.position.array),
);

console.log(`\n${matched}/${parts.length} track parts match the game's checksums`
            + (failed ? `, ${failed} did not` : ""));
console.log(`car hull ${carHull.length / 3} verts  sha256 ${sha256(carHull)}`);

writeFileSync(OUT, JSON.stringify({
  version: "0.6.2",
  massOffset: 0.6,
  carCollisionShapeVertices: Array.from(carHull),
  parts: results,
}));
console.log(`wrote ${OUT}`);
process.exit(failed ? 1 : 0);
