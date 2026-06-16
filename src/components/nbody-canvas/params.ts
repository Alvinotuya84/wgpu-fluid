// Number of particles — O(n²) gravity means cost grows with the square
// of this, so keep it modest on mobile GPUs.
export const PARTICLE_COUNT = 512;

// Workgroup size for the gravity compute shader (1D — one thread per particle).
export const WORKGROUP_SIZE = 64;

// Time step per frame.
export const DT = 0.016;

// Gravitational constant — tuned for the [0,1]×[0,1] normalized sim space.
export const GRAVITY = 0.00012;

// Softening factor — prevents the force (and 1/distance³) from blowing up
// when two particles get very close to each other.
export const SOFTENING = 0.0008;

// Hard speed cap so a close encounter can't fling a particle off to infinity.
export const MAX_SPEED = 0.6;

// Strength of the attraction toward the touch point.
export const TOUCH_STRENGTH = 0.01;

// Particle quad half-size, in clip-space units (screen spans [-1, 1]).
export const PARTICLE_SIZE = 0.012;
