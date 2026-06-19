// ── Geometry ─────────────────────────────────────────────────────────────────
export const TYRE_SEGMENTS = 64;

// Radii in "circle space" units (fraction of half-canvas-height, Y-up at center)
export const TYRE_OUTER_R = 0.38;  // outer tyre edge
export const TYRE_INNER_R = 0.25;  // inner tyre / outer rim interface
export const RIM_OUTER_R  = 0.24;
export const RIM_INNER_R  = 0.10;
export const HUB_R        = 0.09;

// Total vertices: ring × 6 + ring × 6 + fan × 3
export const TYRE_SECTION_VERTS    = TYRE_SEGMENTS * 6;  // 384
export const RIM_SECTION_VERTS     = TYRE_SEGMENTS * 6;  // 384
export const HUB_SECTION_VERTS     = TYRE_SEGMENTS * 3;  // 192
export const TOTAL_TYRE_VERTS      = TYRE_SECTION_VERTS + RIM_SECTION_VERTS + HUB_SECTION_VERTS; // 960

// ── Physics ───────────────────────────────────────────────────────────────────
export const DEFAULT_ANGULAR_VELOCITY = 5.0;   // rad/s (~0.8 rev/s)
export const MAX_ANGULAR_VELOCITY     = 8.0;
export const MIN_ANGULAR_VELOCITY     = 0.2;
export const BRAKE_FRICTION           = 0.95;  // per frame multiplier while braking
export const RECOVERY_RATE            = 1.002; // per frame multiplier on release
export const HARD_BRAKE_THRESHOLD     = 1.0;   // seconds of braking before smoke

// ── Particles ─────────────────────────────────────────────────────────────────
export const MAX_PARTICLES     = 256;
export const WORKGROUP_SIZE    = 64;
export const PARTICLE_NDC_SIZE = 0.018; // half-size of each smoke quad in NDC

// ── Timing ────────────────────────────────────────────────────────────────────
export const DT = 0.016; // fixed physics timestep (≈60 fps)
