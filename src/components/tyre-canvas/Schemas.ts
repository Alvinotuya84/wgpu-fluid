import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { MAX_PARTICLES, TOTAL_TYRE_VERTS } from './params';

// ── Uniforms (CPU → GPU, every frame) ────────────────────────────────────────
export const TyreUniforms = d.struct({
  time:        d.f32,
  dt:          d.f32,
  touchActive: d.u32,
  aspectRatio: d.f32, // canvas_width / canvas_height
});

export type TyreUniforms = d.Infer<typeof TyreUniforms>;

// ── Physics state (GPU-owned, written by compute) ─────────────────────────────
export const PhysicsState = d.struct({
  angularVelocity: d.f32,
  angle:           d.f32,
  touchDuration:   d.f32,
  _pad:            d.f32,
});

export type PhysicsState = d.Infer<typeof PhysicsState>;

// ── Smoke particle ────────────────────────────────────────────────────────────
export const SmokeParticle = d.struct({
  position: d.vec2f, // in circle-space (NDC-Y units, Y-up, center=0)
  velocity: d.vec2f,
  life:     d.f32,   // 0 = dead, 1 = freshly spawned
  seed:     d.f32,   // per-particle fixed random seed [0, 1)
  _pad0:    d.f32,
  _pad1:    d.f32,
});

export const SmokeParticleBuffer = d.arrayOf(SmokeParticle, MAX_PARTICLES);

// ── Pre-baked tyre vertex (CPU builds once, GPU applies rotation each frame) ──
// All floats — avoids vec alignment surprises in the struct.
export const TyreVertex = d.struct({
  baseAngle: d.f32, // rest-pose angle (radians)
  radius:    d.f32, // distance from center in circle-space
  r:         d.f32,
  g:         d.f32,
  b:         d.f32,
  _pad:      d.f32,
});

export const TyreVertexBuffer = d.arrayOf(TyreVertex, TOTAL_TYRE_VERTS);

// ── Compute bind group: physics + particles are read-write ────────────────────
export const tyreComputeLayout = tgpu.bindGroupLayout({
  uniforms:  { uniform: TyreUniforms },
  physics:   { storage: PhysicsState,        access: 'mutable' },
  particles: { storage: SmokeParticleBuffer, access: 'mutable' },
});

// ── Tyre render bind group: all read-only + the prebuilt vertex buffer ────────
export const tyreRenderLayout = tgpu.bindGroupLayout({
  uniforms:     { uniform: TyreUniforms },
  physics:      { storage: PhysicsState },
  tyreVertices: { storage: TyreVertexBuffer },
});

// ── Particle render bind group: uniforms + read-only particles ────────────────
export const particleRenderLayout = tgpu.bindGroupLayout({
  uniforms:  { uniform: TyreUniforms },
  particles: { storage: SmokeParticleBuffer },
});
