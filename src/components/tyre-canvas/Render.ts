import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import { PARTICLE_NDC_SIZE } from './params';
import {
  particleRenderLayout,
  tyreRenderLayout,
} from './Schemas';

// ── Shared quad corners (CCW, reused for smoke particle quads) ────────────────
const quadCorners = tgpu.const(d.arrayOf(d.vec2f), [
  d.vec2f(-1, -1),
  d.vec2f( 1, -1),
  d.vec2f(-1,  1),
  d.vec2f(-1,  1),
  d.vec2f( 1, -1),
  d.vec2f( 1,  1),
]);

// ── TYRE GEOMETRY PIPELINE ────────────────────────────────────────────────────

/**
 * Reads a pre-baked TyreVertex (baseAngle, radius, rgb) from the storage buffer,
 * applies the current physics rotation, and outputs clip-space position + color.
 *
 * Aspect-ratio correction keeps the circle looking circular on portrait screens:
 *   NDC_x = cos(θ) * r / aspectRatio
 *   NDC_y = sin(θ) * r
 */
export const tyreVertexShader = tgpu.vertexFn({
  in:  { vid: d.builtin.vertexIndex },
  out: { pos: d.builtin.position, color: d.vec3f },
})((input) => {
  'use gpu';

  const vert    = tyreRenderLayout.$.tyreVertices[input.vid];
  const physics = tyreRenderLayout.$.physics;
  const aspect  = tyreRenderLayout.$.uniforms.aspectRatio;

  const theta = vert.baseAngle + physics.angle;
  const r     = vert.radius;

  const x = std.cos(theta) * r / aspect;
  const y = std.sin(theta) * r;

  return {
    pos:   d.vec4f(x, y, 0, 1),
    color: d.vec3f(vert.r, vert.g, vert.b),
  };
});

export const tyreFragmentShader = tgpu.fragmentFn({
  in:  { color: d.vec3f },
  out: d.vec4f,
})((input) => {
  'use gpu';
  return d.vec4f(input.color, d.f32(1));
});

// ── SMOKE PARTICLE PIPELINE ───────────────────────────────────────────────────

/**
 * Instanced quads — one per particle.
 * Dead particles are pushed off-screen.
 * Smoke fades from near-white to transparent gray as life drains.
 *
 * Particle positions are in circle-space (NDC-Y units), so X is divided
 * by aspectRatio to undo the same correction applied to the tyre geometry.
 */
export const particleVertexShader = tgpu.vertexFn({
  in:  { vid: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
  out: { pos: d.builtin.position, life: d.f32 },
})((input) => {
  'use gpu';

  const particle = particleRenderLayout.$.particles[input.iid];
  const aspect   = particleRenderLayout.$.uniforms.aspectRatio;
  const corner   = quadCorners.$[input.vid];
  const life     = particle.life;

  if (life <= 0.0) {
    // Push dead particle off screen so the GPU still fills 6 vertices
    return { pos: d.vec4f(d.f32(10), d.f32(10), 0, 1), life: d.f32(0) };
  }

  // Particles grow slightly as they age
  const size = d.f32(PARTICLE_NDC_SIZE) * (d.f32(1) + (d.f32(1) - life) * d.f32(0.6));

  const px = particle.position.x / aspect + corner.x * size;
  const py = particle.position.y             + corner.y * size;

  return { pos: d.vec4f(px, py, 0, 1), life };
});

export const particleFragmentShader = tgpu.fragmentFn({
  in:  { life: d.f32 },
  out: d.vec4f,
})((input) => {
  'use gpu';

  const l = input.life;
  // Slight blue tint for tire-smoke look; intensity tracks life
  return d.vec4f(
    d.f32(0.78) * l,
    d.f32(0.80) * l,
    d.f32(0.88) * l,
    d.f32(1),
  );
});
