import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { WORKGROUP_SIZE } from "./params";
import { fluidBindGroupLayout as layout } from "./Schemas";

/**
 * ADVECTION PASS
 *
 * Moves quantities (velocity, dye) along the velocity field using
 * semi-Lagrangian advection — trace backwards along velocity, sample
 * the quantity at that position, and write it to the current cell.
 *
 * This is what makes the fluid "flow".
 */

// safe index helper — clamp to grid. Defined at module scope (not nested
// inside advectShader's body) since TGSL doesn't support arrow functions
// declared inside another GPU function.
const safeIdx = (ix: number, iy: number): number => {
  "use gpu";
  const gridSize = layout.$.uniforms.gridSize;
  const cx = std.clamp(ix, 0, d.i32(gridSize) - 1);
  const cy = std.clamp(iy, 0, d.i32(gridSize) - 1);
  return d.u32(cy * d.i32(gridSize) + cx);
};

export const advectShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE],
})((input) => {
  "use gpu";

  const gridSize = layout.$.uniforms.gridSize;
  const dt = layout.$.uniforms.dt;

  const x = d.i32(input.gid.x);
  const y = d.i32(input.gid.y);

  // bounds check
  if (x >= d.i32(gridSize) || y >= d.i32(gridSize)) {
    return;
  }

  const idx = y * d.i32(gridSize) + x;
  const cell = layout.$.current[idx];

  // trace backwards: where did this fluid come from?
  const prevX = d.f32(x) - cell.velocity.x * dt * d.f32(gridSize);
  const prevY = d.f32(y) - cell.velocity.y * dt * d.f32(gridSize);

  // clamp to grid bounds
  const clampedX = std.clamp(prevX, 0.5, d.f32(gridSize) - 1.5);
  const clampedY = std.clamp(prevY, 0.5, d.f32(gridSize) - 1.5);

  // bilinear interpolation indices
  const x0 = d.i32(std.floor(clampedX));
  const y0 = d.i32(std.floor(clampedY));
  const x1 = x0 + 1;
  const y1 = y0 + 1;

  // interpolation weights
  const tx = clampedX - d.f32(x0);
  const ty = clampedY - d.f32(y0);

  const c00 = layout.$.current[safeIdx(x0, y0)];
  const c10 = layout.$.current[safeIdx(x1, y0)];
  const c01 = layout.$.current[safeIdx(x0, y1)];
  const c11 = layout.$.current[safeIdx(x1, y1)];

  // bilinear interpolate velocity
  const vel = std.mix(
    std.mix(c00.velocity, c10.velocity, tx),
    std.mix(c01.velocity, c11.velocity, tx),
    ty,
  );

  // bilinear interpolate dye
  const dye = std.mix(
    std.mix(c00.dye, c10.dye, tx),
    std.mix(c01.dye, c11.dye, tx),
    ty,
  );

  // write advected values, preserve pressure/divergence
  const next = layout.$.next[idx];
  next.velocity = d.vec2f(vel);
  next.dye = d.vec3f(std.mul(dye, d.vec3f(0.999))); // slight dye decay so it fades over time
  next.pressure = d.f32(cell.pressure);
  next.divergence = d.f32(cell.divergence);
});
