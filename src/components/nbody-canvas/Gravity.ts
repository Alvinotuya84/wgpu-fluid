import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
  GRAVITY,
  MAX_SPEED,
  PARTICLE_COUNT,
  SOFTENING,
  TOUCH_STRENGTH,
  WORKGROUP_SIZE,
} from "./params";
import { nbodyBindGroupLayout as layout } from "./Schemas";

/**
 * GRAVITY PASS
 *
 * Brute-force O(n²) N-body gravity: every particle attracts every other
 * particle, with a softening term so the force doesn't blow up when two
 * particles get very close. Touching the screen adds an extra attractor
 * that pulls particles toward the touch point.
 */
export const gravityShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})((input) => {
  "use gpu";

  const i = d.i32(input.gid.x);
  if (i >= PARTICLE_COUNT) return;

  const self = layout.$.current[i];

  let fx = d.f32(0);
  let fy = d.f32(0);

  for (let j = 0; j < PARTICLE_COUNT; j++) {
    if (j === i) continue;

    const other = layout.$.current[j];
    const dx = other.position.x - self.position.x;
    const dy = other.position.y - self.position.y;
    const distSq = dx * dx + dy * dy + SOFTENING;
    const invDist = std.inverseSqrt(distSq);
    const invDist3 = invDist * invDist * invDist;
    const pull = GRAVITY * other.mass * invDist3;

    fx = fx + dx * pull;
    fy = fy + dy * pull;
  }

  const uniforms = layout.$.uniforms;

  if (uniforms.touchActive !== d.u32(0)) {
    const dx = uniforms.touchX - self.position.x;
    const dy = uniforms.touchY - self.position.y;
    const distSq = dx * dx + dy * dy + SOFTENING;
    const invDist = std.inverseSqrt(distSq);
    const invDist3 = invDist * invDist * invDist;
    const pull = TOUCH_STRENGTH * invDist3;

    fx = fx + dx * pull;
    fy = fy + dy * pull;
  }

  const dt = uniforms.dt;
  const vx = self.velocity.x + fx * dt;
  const vy = self.velocity.y + fy * dt;

  // cap speed so a close encounter can't fling a particle to infinity
  const speed = std.length(d.vec2f(vx, vy));
  const scale = std.min(d.f32(1), MAX_SPEED / std.max(speed, 0.0001));
  const newVx = vx * scale;
  const newVy = vy * scale;

  // wrap around the [0,1] sim space instead of losing particles off-screen
  const newX = std.fract(self.position.x + newVx * dt);
  const newY = std.fract(self.position.y + newVy * dt);

  const next = layout.$.next[i];
  next.position = d.vec2f(newX, newY);
  next.velocity = d.vec2f(newVx, newVy);
  next.mass = d.f32(self.mass);
});
