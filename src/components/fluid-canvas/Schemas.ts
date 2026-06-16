import tgpu from "typegpu";
import * as d from "typegpu/data";
import { GRID_SIZE } from "./params";

// Total number of cells in the grid
export const CELL_COUNT = GRID_SIZE * GRID_SIZE;

// A single fluid cell stores velocity (x, y) and dye density
export const FluidCell = d.struct({
  velocity: d.vec2f, // fluid velocity at this cell
  dye: d.vec3f, // RGB dye color at this cell
  pressure: d.f32, // pressure value used in projection step
  divergence: d.f32, // divergence of velocity field
});

export type FluidCell = d.Infer<typeof FluidCell>;

// The full fluid grid — flat array of GRID_SIZE * GRID_SIZE cells
export const FluidGrid = d.arrayOf(FluidCell, CELL_COUNT);

// Uniforms passed to every shader
export const FluidUniforms = d.struct({
  gridSize: d.u32, // GRID_SIZE (128)
  dt: d.f32, // delta time
  viscosity: d.f32,
  diffusion: d.f32,
  time: d.f32, // total elapsed time (for touch input seeding)
  touchX: d.f32, // normalized touch X [0..1]
  touchY: d.f32, // normalized touch Y [0..1]
  touchActive: d.u32, // bool — is user touching?
});

export type FluidUniforms = d.Infer<typeof FluidUniforms>;

// Bind group layout shared by all compute passes
// ping-pong: read from `current`, write to `next`
export const fluidBindGroupLayout = tgpu.bindGroupLayout({
  current: { storage: FluidGrid },
  next: { storage: FluidGrid, access: "mutable" },
  uniforms: { uniform: FluidUniforms },
});
