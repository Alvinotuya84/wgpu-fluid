import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { fluidBindGroupLayout as layout } from "./Schemas";

/**
 * RENDER PIPELINE
 *
 * Draws the fluid dye field as a fullscreen quad.
 * Vertex shader emits 2 triangles covering the screen.
 * Fragment shader reads dye color from the grid cell at the UV coordinate.
 */

// fullscreen quad — 2 triangles, 6 vertices
const fullscreenPositions = tgpu.const(d.arrayOf(d.vec2f), [
  d.vec2f(-1, -1),
  d.vec2f(1, -1),
  d.vec2f(-1, 1),
  d.vec2f(-1, 1),
  d.vec2f(1, -1),
  d.vec2f(1, 1),
]);

export const fluidVertexShader = tgpu.vertexFn({
  in: { vid: d.builtin.vertexIndex },
  out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
  "use gpu";
  const p = fullscreenPositions.$[input.vid];
  return {
    pos: d.vec4f(p, 0, 1),
    uv: std.add(std.mul(p, 0.5), d.vec2f(0.5)), // [-1,1] → [0,1]
  };
});

export const fluidFragmentShader = tgpu.fragmentFn({
  in: { uv: d.vec2f },
  out: d.vec4f,
})((input) => {
  "use gpu";

  const gridSize = layout.$.uniforms.gridSize;

  // map UV to grid cell
  const gx = d.i32(input.uv.x * d.f32(gridSize));
  const gy = d.i32(input.uv.y * d.f32(gridSize));

  const cx = std.clamp(gx, 0, d.i32(gridSize) - 1);
  const cy = std.clamp(gy, 0, d.i32(gridSize) - 1);

  const idx = cy * d.i32(gridSize) + cx;
  const dye = layout.$.current[idx].dye;

  // tone map: slightly boost saturation for visual punch
  const boosted = std.clamp(std.mul(dye, 1.2), d.vec3f(0), d.vec3f(1));

  return d.vec4f(boosted, 1.0);
});
