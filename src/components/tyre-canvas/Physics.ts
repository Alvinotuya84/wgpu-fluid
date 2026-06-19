import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import * as std from 'typegpu/std';
import {
  BRAKE_FRICTION,
  MAX_ANGULAR_VELOCITY,
  MIN_ANGULAR_VELOCITY,
  RECOVERY_RATE,
} from './params';
import { tyreComputeLayout as layout } from './Schemas';

/**
 * PHYSICS PASS  (1 thread)
 *
 * Updates angularVelocity and accumulated angle based on touch state.
 * touchDuration is incremented while braking, decays on release.
 */
export const physicsShader = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [1],
})((/* input */) => {
  'use gpu';

  const uniforms = layout.$.uniforms;
  const physics  = layout.$.physics;

  const av = physics.angularVelocity;
  const td = physics.touchDuration;

  let newAV = av;
  let newTD = td;

  if (uniforms.touchActive !== d.u32(0)) {
    newAV = av * BRAKE_FRICTION;
    newTD = td + uniforms.dt;
  } else {
    newAV = std.min(av * RECOVERY_RATE, d.f32(MAX_ANGULAR_VELOCITY));
    newTD = std.max(td - uniforms.dt, d.f32(0));
  }

  // Floor at minimum so the tyre never stops completely
  if (newAV < MIN_ANGULAR_VELOCITY) {
    newAV = d.f32(MIN_ANGULAR_VELOCITY);
  }

  physics.angularVelocity = newAV;
  physics.angle           = physics.angle + newAV * uniforms.dt;
  physics.touchDuration   = newTD;
});
