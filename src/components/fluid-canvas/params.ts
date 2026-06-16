// Grid resolution — 128x128 cells
export const GRID_SIZE = 128;

// How many Jacobi iterations for pressure solve (more = more accurate, more expensive)
export const JACOBI_ITERATIONS = 20;

// Fluid viscosity — higher = thicker fluid
export const VISCOSITY = 0.0001;

// How fast the dye/color diffuses
export const DIFFUSION = 0.0001;

// Time step per frame
export const DT = 0.016;

// Workgroup size for compute shaders (8x8 = 64 threads, fits most mobile GPUs)
export const WORKGROUP_SIZE = 8;
