import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { WORKGROUP_SIZE } from "./params";
import { fluidBindGroupLayout as layout } from "./Schemas";

/**
 * SPLAT PASS
 *
 * Injects velocity and dye at the touch position.
 * Each cell within a radius of the touch gets a velocity impulse
 * and a burst of colored dye. This is the user interaction layer.
 */
export const splatShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE],
})((input) => {
  "use gpu";

  const uniforms = layout.$.uniforms;
  const gridSize = uniforms.gridSize;

  const x = d.i32(input.gid.x);
  const y = d.i32(input.gid.y);

  if (x >= d.i32(gridSize) || y >= d.i32(gridSize)) return;

  const idx = y * d.i32(gridSize) + x;
  const cell = layout.$.current[idx];
  const next = layout.$.next[idx];

  // copy existing state first — must go through a value constructor since
  // `cell.X` is a reference into the storage buffer, not a value
  next.velocity = d.vec2f(cell.velocity);
  next.dye = d.vec3f(cell.dye);
  next.pressure = d.f32(cell.pressure);
  next.divergence = d.f32(cell.divergence);

  if (uniforms.touchActive === d.u32(0)) return;

  // touch position in grid space
  const touchGX = uniforms.touchX * d.f32(gridSize);
  const touchGY = uniforms.touchY * d.f32(gridSize);

  const dx = d.f32(x) - touchGX;
  const dy = d.f32(y) - touchGY;
  const dist2 = dx * dx + dy * dy;

  // splat radius — 4 cells
  const radius = 4.0;
  const strength = std.exp(-dist2 / (radius * radius));

  if (strength < 0.001) return;

  // inject velocity outward from touch center
  next.velocity = d.vec2f(std.add(cell.velocity, std.mul(d.vec2f(dx, dy), strength * 0.3)));

  // inject dye — cycle color based on time
  const t = uniforms.time;
  const dyeColor = d.vec3f(
    std.sin(t * 0.7) * 0.5 + 0.5,
    std.sin(t * 1.3 + 2.094) * 0.5 + 0.5, // 2π/3 offset
    std.sin(t * 1.1 + 4.189) * 0.5 + 0.5, // 4π/3 offset
  );

  next.dye = d.vec3f(std.clamp(std.add(cell.dye, std.mul(dyeColor, strength)), d.vec3f(0), d.vec3f(1)));
});
