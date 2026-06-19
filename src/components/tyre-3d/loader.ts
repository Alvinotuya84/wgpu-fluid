import { Asset } from 'expo-asset';

export interface MeshData {
  name:        string;
  positions:   Float32Array;
  normals:     Float32Array;
  uvs:         Float32Array;
  indices:     Uint16Array | Uint32Array;
  indexFormat: 'uint16' | 'uint32';
}

export interface RawTextures {
  tireBaseColor:          Uint8Array; // bufferView 4
  tireNormal:             Uint8Array; // bufferView 5
  wheelBaseColor:         Uint8Array; // bufferView 6
  wheelMetallicRoughness: Uint8Array; // bufferView 7
  wheelNormal:            Uint8Array; // bufferView 8
}

export interface LoadResult {
  meshes:   MeshData[];
  textures: RawTextures;
}

// ── Minimal GLTF types ────────────────────────────────────────────────────────

interface GLTFAccessor {
  bufferView:    number;
  byteOffset?:   number;
  componentType: number; // 5126=FLOAT 5123=UNSIGNED_SHORT 5125=UNSIGNED_INT
  type:          string; // 'SCALAR' 'VEC2' 'VEC3' 'VEC4'
  count:         number;
}

interface GLTFBufferView {
  buffer:      number;
  byteOffset?: number;
  byteLength:  number;
  byteStride?: number;
}

interface GLTFPrimitive {
  attributes: { POSITION: number; NORMAL?: number; TEXCOORD_0?: number };
  indices:    number;
  mode?:      number;
}

interface GLTFMesh {
  name:       string;
  primitives: GLTFPrimitive[];
}

interface GLTFJson {
  meshes:      GLTFMesh[];
  accessors:   GLTFAccessor[];
  bufferViews: GLTFBufferView[];
}

// ── Component-type helpers ────────────────────────────────────────────────────

function compCount(type: string): number {
  switch (type) {
    case 'SCALAR': return 1;
    case 'VEC2':   return 2;
    case 'VEC3':   return 3;
    case 'VEC4':   return 4;
    default:       return 1;
  }
}

function compBytes(componentType: number): number {
  switch (componentType) {
    case 5120: case 5121: return 1;
    case 5122: case 5123: return 2;
    case 5125: case 5126: return 4;
    default:               return 4;
  }
}

// ── GLB binary parser ─────────────────────────────────────────────────────────

function parseGLB(arrayBuffer: ArrayBuffer): LoadResult {
  const dv = new DataView(arrayBuffer);

  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546c67) throw new Error('Not a valid GLB file (wrong magic).');

  const jsonLen  = dv.getUint32(12, true);
  const jsonType = dv.getUint32(16, true);
  if (jsonType !== 0x4e4f534a) throw new Error('Expected JSON chunk first.');

  const jsonText = new TextDecoder().decode(new Uint8Array(arrayBuffer, 20, jsonLen));
  const json: GLTFJson = JSON.parse(jsonText);

  const binOffset = 20 + jsonLen;
  const binLen    = dv.getUint32(binOffset, true);
  const binType   = dv.getUint32(binOffset + 4, true);
  if (binType !== 0x004e4942) throw new Error('Expected BIN chunk after JSON.');

  const binStart  = binOffset + 8;
  const binBuffer = arrayBuffer.slice(binStart, binStart + binLen);

  // ── Accessor resolver ───────────────────────────────────────────────────────
  function getFloat32(accessorIdx: number): Float32Array {
    const acc  = json.accessors[accessorIdx];
    const bv   = json.bufferViews[acc.bufferView];
    const cc   = compCount(acc.type);
    const cb   = compBytes(acc.componentType);
    const stride    = bv.byteStride ?? (cc * cb);
    const bvOff     = bv.byteOffset ?? 0;
    const accOff    = acc.byteOffset ?? 0;
    const startByte = bvOff + accOff;

    if (stride === cc * cb) {
      const byteLen = acc.count * cc * cb;
      return new Float32Array(binBuffer.slice(startByte, startByte + byteLen));
    }

    const out   = new Float32Array(acc.count * cc);
    const srcDV = new DataView(binBuffer);
    for (let i = 0; i < acc.count; i++) {
      const base = startByte + i * stride;
      for (let c = 0; c < cc; c++) {
        out[i * cc + c] = srcDV.getFloat32(base + c * 4, true);
      }
    }
    return out;
  }

  function getIndexBuffer(accessorIdx: number): { data: Uint16Array | Uint32Array; format: 'uint16' | 'uint32' } {
    const acc    = json.accessors[accessorIdx];
    const bv     = json.bufferViews[acc.bufferView];
    const bvOff  = bv.byteOffset ?? 0;
    const accOff = acc.byteOffset ?? 0;
    const start  = bvOff + accOff;
    const cb     = compBytes(acc.componentType);
    const bytes  = acc.count * cb;
    const slice  = binBuffer.slice(start, start + bytes);
    if (acc.componentType === 5123) {
      return { data: new Uint16Array(slice), format: 'uint16' };
    }
    return { data: new Uint32Array(slice), format: 'uint32' };
  }

  // Extracts raw bytes from a bufferView (for embedded PNG textures)
  function extractBV(bvIdx: number): Uint8Array {
    const bv     = json.bufferViews[bvIdx];
    const offset = bv.byteOffset ?? 0;
    return new Uint8Array(binBuffer, offset, bv.byteLength).slice();
  }

  // ── Generate flat normals fallback ──────────────────────────────────────────
  function flatNormals(pos: Float32Array, idx: Uint16Array | Uint32Array): Float32Array {
    const out = new Float32Array(pos.length);
    for (let t = 0; t < idx.length / 3; t++) {
      const [i0, i1, i2] = [idx[t*3], idx[t*3+1], idx[t*3+2]];
      const ax = pos[i0*3], ay = pos[i0*3+1], az = pos[i0*3+2];
      const ex = pos[i1*3]-ax, ey = pos[i1*3+1]-ay, ez = pos[i1*3+2]-az;
      const fx = pos[i2*3]-ax, fy = pos[i2*3+1]-ay, fz = pos[i2*3+2]-az;
      const nx = ey*fz-ez*fy, ny = ez*fx-ex*fz, nz = ex*fy-ey*fx;
      const l  = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
      for (const i of [i0, i1, i2]) {
        out[i*3]   += nx/l;
        out[i*3+1] += ny/l;
        out[i*3+2] += nz/l;
      }
    }
    for (let i = 0; i < out.length / 3; i++) {
      const x = out[i*3], y = out[i*3+1], z = out[i*3+2];
      const l = Math.sqrt(x*x+y*y+z*z) || 1;
      out[i*3] /= l; out[i*3+1] /= l; out[i*3+2] /= l;
    }
    return out;
  }

  // ── Extract each mesh ───────────────────────────────────────────────────────
  const meshes: MeshData[] = [];

  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      const positions  = getFloat32(prim.attributes.POSITION);
      const { data: indices, format: indexFormat } = getIndexBuffer(prim.indices);
      const normals = prim.attributes.NORMAL !== undefined
        ? getFloat32(prim.attributes.NORMAL)
        : flatNormals(positions, indices);
      const uvs = prim.attributes.TEXCOORD_0 !== undefined
        ? getFloat32(prim.attributes.TEXCOORD_0)
        : new Float32Array(positions.length / 3 * 2); // zero UVs fallback

      meshes.push({ name: mesh.name ?? 'mesh', positions, normals, uvs, indices, indexFormat });
    }
  }

  // ── Extract embedded PNG textures from known bufferViews ───────────────────
  const textures: RawTextures = {
    tireBaseColor:          extractBV(4),
    tireNormal:             extractBV(5),
    wheelBaseColor:         extractBV(6),
    wheelMetallicRoughness: extractBV(7),
    wheelNormal:            extractBV(8),
  };

  return { meshes, textures };
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function loadWheelGLB(): Promise<LoadResult> {
  const asset = Asset.fromModule(
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require('../../../assets/models/car_wheel_with_brake_disc_low_poly.glb'),
  );
  await asset.downloadAsync();

  const response = await fetch(asset.localUri!);
  const buffer   = await response.arrayBuffer();

  return parseGLB(buffer);
}
