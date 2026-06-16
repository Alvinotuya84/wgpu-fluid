import tgpu from "typegpu";
import * as d from "typegpu/data";
import { PARTICLE_COUNT } from "./params";

// A single particle: position/velocity in normalized [0,1]×[0,1] sim
// space, plus a mass that scales how strongly it pulls others.
export const Particle = d.struct({
  position: d.vec2f,
  velocity: d.vec2f,
  mass: d.f32,
});

export type Particle = d.Infer<typeof Particle>;

// The full particle array — flat list of PARTICLE_COUNT particles.
export const ParticleBuffer = d.arrayOf(Particle, PARTICLE_COUNT);

// Uniforms passed to the compute shader every frame.
export const NBodyUniforms = d.struct({
  dt: d.f32,
  touchX: d.f32, // normalized touch X [0..1]
  touchY: d.f32, // normalized touch Y [0..1]
  touchActive: d.u32, // bool — is user touching?
});

export type NBodyUniforms = d.Infer<typeof NBodyUniforms>;

// Bind group layout shared by the compute and render pipelines.
// ping-pong: read from `current`, write to `next`.
export const nbodyBindGroupLayout = tgpu.bindGroupLayout({
  current: { storage: ParticleBuffer },
  next: { storage: ParticleBuffer, access: "mutable" },
  uniforms: { uniform: NBodyUniforms },
});
