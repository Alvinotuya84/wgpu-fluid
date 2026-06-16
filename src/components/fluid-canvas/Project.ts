import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { WORKGROUP_SIZE } from "./params";
import { fluidBindGroupLayout as layout } from "./Schemas";

/**
 * DIVERGENCE PASS
 *
 * Computes how much fluid is "piling up" at each cell.
 * A divergence-free field means fluid is neither created nor destroyed.
 * We need this to feed into the pressure solver.
 */
export const divergenceShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE],
})((input) => {
  "use gpu";

  const gridSize = layout.$.uniforms.gridSize;
  const x = d.i32(input.gid.x);
  const y = d.i32(input.gid.y);

  if (x >= d.i32(gridSize) || y >= d.i32(gridSize)) return;

  const idx = y * d.i32(gridSize) + x;

  // neighbor indices with boundary clamping
  const left = std.clamp(x - 1, 0, d.i32(gridSize) - 1);
  const right = std.clamp(x + 1, 0, d.i32(gridSize) - 1);
  const down = std.clamp(y - 1, 0, d.i32(gridSize) - 1);
  const up = std.clamp(y + 1, 0, d.i32(gridSize) - 1);

  const vL = layout.$.current[down * d.i32(gridSize) + x].velocity;
  const vR = layout.$.current[up * d.i32(gridSize) + x].velocity;
  const vD = layout.$.current[y * d.i32(gridSize) + left].velocity;
  const vU = layout.$.current[y * d.i32(gridSize) + right].velocity;

  // central differences divergence: (dvx/dx + dvy/dy) / 2
  const div = (vU.x - vD.x + vR.y - vL.y) * 0.5;

  const next = layout.$.next[idx];
  next.velocity = d.vec2f(layout.$.current[idx].velocity);
  next.dye = d.vec3f(layout.$.current[idx].dye);
  next.pressure = d.f32(layout.$.current[idx].pressure);
  next.divergence = d.f32(div);
});

/**
 * JACOBI PRESSURE SOLVE PASS
 *
 * Iteratively solves for the pressure field that, when subtracted
 * from velocity, makes the fluid divergence-free (incompressible).
 * We run this ~20 times per frame — more iterations = more accurate.
 */
export const pressureShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE],
})((input) => {
  "use gpu";

  const gridSize = layout.$.uniforms.gridSize;
  const x = d.i32(input.gid.x);
  const y = d.i32(input.gid.y);

  if (x >= d.i32(gridSize) || y >= d.i32(gridSize)) return;

  const idx = y * d.i32(gridSize) + x;

  const left = std.clamp(x - 1, 0, d.i32(gridSize) - 1);
  const right = std.clamp(x + 1, 0, d.i32(gridSize) - 1);
  const down = std.clamp(y - 1, 0, d.i32(gridSize) - 1);
  const up = std.clamp(y + 1, 0, d.i32(gridSize) - 1);

  const pL = layout.$.current[y * d.i32(gridSize) + left].pressure;
  const pR = layout.$.current[y * d.i32(gridSize) + right].pressure;
  const pD = layout.$.current[down * d.i32(gridSize) + x].pressure;
  const pU = layout.$.current[up * d.i32(gridSize) + x].pressure;

  const div = layout.$.current[idx].divergence;

  // Jacobi iteration: p = (neighbors - divergence) / 4
  const newPressure = (pL + pR + pD + pU - div) * 0.25;

  const next = layout.$.next[idx];
  next.velocity = d.vec2f(layout.$.current[idx].velocity);
  next.dye = d.vec3f(layout.$.current[idx].dye);
  next.pressure = d.f32(newPressure);
  next.divergence = d.f32(div);
});

/**
 * GRADIENT SUBTRACT PASS
 *
 * Subtracts the pressure gradient from the velocity field.
 * This is what actually enforces incompressibility —
 * after this pass the fluid conserves mass.
 */
export const gradientSubtractShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE],
})((input) => {
  "use gpu";

  const gridSize = layout.$.uniforms.gridSize;
  const x = d.i32(input.gid.x);
  const y = d.i32(input.gid.y);

  if (x >= d.i32(gridSize) || y >= d.i32(gridSize)) return;

  const idx = y * d.i32(gridSize) + x;

  const left = std.clamp(x - 1, 0, d.i32(gridSize) - 1);
  const right = std.clamp(x + 1, 0, d.i32(gridSize) - 1);
  const down = std.clamp(y - 1, 0, d.i32(gridSize) - 1);
  const up = std.clamp(y + 1, 0, d.i32(gridSize) - 1);

  const pL = layout.$.current[y * d.i32(gridSize) + left].pressure;
  const pR = layout.$.current[y * d.i32(gridSize) + right].pressure;
  const pD = layout.$.current[down * d.i32(gridSize) + x].pressure;
  const pU = layout.$.current[up * d.i32(gridSize) + x].pressure;

  // pressure gradient
  const gradX = (pR - pL) * 0.5;
  const gradY = (pU - pD) * 0.5;

  const cell = layout.$.current[idx];
  const next = layout.$.next[idx];

  // subtract gradient from velocity
  next.velocity = d.vec2f(cell.velocity.x - gradX, cell.velocity.y - gradY);
  next.dye = d.vec3f(cell.dye);
  next.pressure = d.f32(cell.pressure);
  next.divergence = d.f32(cell.divergence);
});
