import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { PARTICLE_SIZE } from "./params";
import { nbodyBindGroupLayout as layout } from "./Schemas";

/**
 * RENDER PIPELINE
 *
 * Draws each particle as a small quad, colored by speed (slow = cool
 * blue, fast = warm yellow/white). Reuses the fluid sim's fullscreen-quad
 * trick — a `tgpu.const` table of corner offsets indexed by vertex index —
 * but instanced once per particle and scaled down to a dot size.
 */
const quadCorners = tgpu.const(d.arrayOf(d.vec2f), [
  d.vec2f(-1, -1),
  d.vec2f(1, -1),
  d.vec2f(-1, 1),
  d.vec2f(-1, 1),
  d.vec2f(1, -1),
  d.vec2f(1, 1),
]);

export const particleVertexShader = tgpu.vertexFn({
  in: { vid: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
  out: { pos: d.builtin.position, speed: d.f32 },
})((input) => {
  "use gpu";

  const particle = layout.$.current[input.iid];
  const corner = quadCorners.$[input.vid];

  // [0,1] sim space → clip space, offset by the quad corner scaled to a dot
  const cx = particle.position.x * 2 - 1 + corner.x * PARTICLE_SIZE;
  const cy = particle.position.y * 2 - 1 + corner.y * PARTICLE_SIZE;

  return {
    pos: d.vec4f(cx, cy, 0, 1),
    speed: std.length(particle.velocity),
  };
});

export const particleFragmentShader = tgpu.fragmentFn({
  in: { speed: d.f32 },
  out: d.vec4f,
})((input) => {
  "use gpu";

  const t = std.clamp(input.speed * 4.0, 0, 1);
  const slow = d.vec3f(0.25, 0.45, 1.0);
  const fast = d.vec3f(1.0, 0.85, 0.3);
  const color = std.mix(slow, fast, t);

  return d.vec4f(color, 1.0);
});
