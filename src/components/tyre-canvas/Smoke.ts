import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import {
  HARD_BRAKE_THRESHOLD,
  MAX_PARTICLES,
  TYRE_OUTER_R,
  WORKGROUP_SIZE,
} from './params';
import { tyreComputeLayout as layout } from './Schemas';

/**
 * SMOKE PARTICLE PASS  (one thread per particle)
 *
 * Living particles: drift upward, lose life.
 * Dead particles: respawn at tyre contact patch when hard-braking.
 *
 * Randomness is faked via large-prime fract hashing on the per-particle
 * seed + current angle — no atomics or RNG state buffer needed.
 */
export const smokeShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})((input) => {
  'use gpu';

  const i = d.i32(input.gid.x);
  if (i >= MAX_PARTICLES) return;

  const uniforms = layout.$.uniforms;
  const physics  = layout.$.physics;
  const particle = layout.$.particles[i];
  const dt       = uniforms.dt;

  if (particle.life > 0.0) {
    // ── Update living particle ──────────────────────────────────────────────
    particle.position = d.vec2f(
      particle.position.x + particle.velocity.x * dt,
      particle.position.y + particle.velocity.y * dt,
    );
    // x-velocity damps slightly; smoke rises (+y in circle-space)
    particle.velocity = d.vec2f(
      particle.velocity.x * d.f32(0.97),
      particle.velocity.y + dt * d.f32(0.06),
    );
    particle.life = std.max(particle.life - dt * d.f32(0.45), d.f32(0));

  } else if (physics.touchDuration > HARD_BRAKE_THRESHOLD) {
    // ── Respawn as smoke puff at tyre contact patch ─────────────────────────
    // Two independent pseudo-random channels per particle:
    const base = particle.seed * d.f32(1000.0) + physics.angle;
    const r1   = std.fract(base * d.f32(127.1) + d.f32(43758.5));
    const r2   = std.fract(base * d.f32(269.5)  + d.f32(12345.6));

    particle.position = d.vec2f(
      (r1 - d.f32(0.5)) * d.f32(0.06),            // slight x scatter
      d.f32(-TYRE_OUTER_R) - d.f32(0.03),          // just below tyre bottom
    );
    particle.velocity = d.vec2f(
      (r2 - d.f32(0.5)) * d.f32(0.12),             // random horizontal drift
      d.f32(0.07) + r1 * d.f32(0.08),              // upward burst
    );
    particle.life = d.f32(1.0);
  }
});
