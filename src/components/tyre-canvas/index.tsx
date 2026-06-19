import React, { useCallback, useRef } from 'react';
import {
  PixelRatio,
  StyleSheet,
  View,
  type GestureResponderEvent,
  type LayoutChangeEvent,
} from 'react-native';
import { Canvas, useCanvasRef, useDevice } from 'react-native-webgpu';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

import { physicsShader } from './Physics';
import {
  DEFAULT_ANGULAR_VELOCITY,
  DT,
  HUB_R,
  MAX_PARTICLES,
  RIM_INNER_R,
  RIM_OUTER_R,
  TYRE_INNER_R,
  TYRE_OUTER_R,
  TYRE_SEGMENTS,
  WORKGROUP_SIZE,
} from './params';
import {
  particleVertexShader,
  particleFragmentShader,
  tyreFragmentShader,
  tyreVertexShader,
} from './Render';
import {
  PhysicsState,
  SmokeParticle,
  SmokeParticleBuffer,
  TyreUniforms,
  TyreVertex,
  TyreVertexBuffer,
  particleRenderLayout,
  tyreComputeLayout,
  tyreRenderLayout,
} from './Schemas';
import { smokeShader } from './Smoke';

// ── useWebGPU hook ────────────────────────────────────────────────────────────

interface SceneProps {
  context: GPUCanvasContext;
  device: GPUDevice;
  presentationFormat: GPUTextureFormat;
}

type RenderScene = (timestamp: number) => void;
type Scene = (props: SceneProps) => RenderScene | Promise<RenderScene>;

function withValidate<T extends unknown[], R>(device: GPUDevice, fn: (...args: T) => R) {
  return (...args: T): R => {
    const scopes: GPUErrorFilter[] = ['validation', 'out-of-memory', 'internal'];
    for (const scope of scopes) device.pushErrorScope(scope);
    const result = fn(...args);
    for (const scope of scopes.reverse()) {
      device.popErrorScope().then((error) => {
        if (error) console.error(`GPU Error [${scope}]:`, error.message);
      });
    }
    return result;
  };
}

function useWebGPU(scene: Scene) {
  const { device } = useDevice();
  const canvasRef  = useCanvasRef();

  React.useEffect(() => {
    (async () => {
      const ref = canvasRef.current;
      if (!ref || !device) return;

      const context = ref.getContext('webgpu');
      if (!context) throw new Error('Failed to get WebGPU context.');

      const canvas = context.canvas as HTMLCanvasElement;
      canvas.width  = canvas.clientWidth  * PixelRatio.get();
      canvas.height = canvas.clientHeight * PixelRatio.get();

      const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format: presentationFormat, alphaMode: 'premultiplied' });

      const renderScene = await withValidate(device, scene)({ context, device, presentationFormat });

      let animId: number;
      const render = () => {
        renderScene(Date.now());
        context.present();
        animId = requestAnimationFrame(render);
      };
      animId = requestAnimationFrame(render);

      return () => cancelAnimationFrame(animId);
    })();
  }, [canvasRef, device, scene]);

  return canvasRef;
}

// ── CPU tyre mesh builder ─────────────────────────────────────────────────────
// Returns an array of TyreVertex objects covering 3 sections:
//   0-383  : tyre body (dark rubber with tread alternation)
//   384-767: rim      (silver)
//   768-959: hub cap  (dark center disk, fan from origin)

function buildTyreVertices(): d.Infer<typeof TyreVertex>[] {
  const TWO_PI  = Math.PI * 2;
  const N       = TYRE_SEGMENTS;
  const verts: d.Infer<typeof TyreVertex>[] = [];

  const v = (
    angle: number, radius: number,
    r: number, g: number, b: number,
  ): d.Infer<typeof TyreVertex> => ({
    baseAngle: angle,
    radius,
    r, g, b,
    _pad: 0,
  });

  // ── Section 0: Tyre ring ────────────────────────────────────────────────────
  // Tread alternates every 4 segments: dark / slightly lighter
  for (let s = 0; s < N; s++) {
    const t1 = (s     / N) * TWO_PI;
    const t2 = ((s+1) / N) * TWO_PI;
    const tread = (Math.floor(s / 4) % 2 === 0);
    const [tr, tg, tb] = tread ? [0.07, 0.07, 0.07] : [0.17, 0.17, 0.17];
    const [ir, ig, ib] = [0.11, 0.11, 0.11]; // inner edge of tyre

    // Quad as two triangles: outer1, inner1, outer2 | inner1, inner2, outer2
    verts.push(v(t1, TYRE_OUTER_R, tr, tg, tb));
    verts.push(v(t1, TYRE_INNER_R, ir, ig, ib));
    verts.push(v(t2, TYRE_OUTER_R, tr, tg, tb));
    verts.push(v(t1, TYRE_INNER_R, ir, ig, ib));
    verts.push(v(t2, TYRE_INNER_R, ir, ig, ib));
    verts.push(v(t2, TYRE_OUTER_R, tr, tg, tb));
  }

  // ── Section 1: Rim (silver ring) ────────────────────────────────────────────
  for (let s = 0; s < N; s++) {
    const t1 = (s     / N) * TWO_PI;
    const t2 = ((s+1) / N) * TWO_PI;
    // Outer rim edge brighter, inner rim edge slightly darker
    const [or_, og, ob] = [0.78, 0.78, 0.82];
    const [ir, ig, ib]  = [0.60, 0.60, 0.65];

    verts.push(v(t1, RIM_OUTER_R, or_, og, ob));
    verts.push(v(t1, RIM_INNER_R, ir,  ig, ib));
    verts.push(v(t2, RIM_OUTER_R, or_, og, ob));
    verts.push(v(t1, RIM_INNER_R, ir,  ig, ib));
    verts.push(v(t2, RIM_INNER_R, ir,  ig, ib));
    verts.push(v(t2, RIM_OUTER_R, or_, og, ob));
  }

  // ── Section 2: Hub cap (filled disk, fan from center) ───────────────────────
  for (let s = 0; s < N; s++) {
    const t1 = (s     / N) * TWO_PI;
    const t2 = ((s+1) / N) * TWO_PI;
    const [hr, hg, hb] = [0.22, 0.22, 0.24]; // dark metallic hub

    // radius=0 at center: cos/sin * 0 = (0,0) regardless of angle
    verts.push(v(0,  0,      hr, hg, hb));
    verts.push(v(t1, HUB_R,  hr, hg, hb));
    verts.push(v(t2, HUB_R,  hr, hg, hb));
  }

  return verts;
}

// ── TyreCanvas component ──────────────────────────────────────────────────────

export function TyreCanvas() {
  const touchRef = useRef({ active: false });

  const scene = useCallback(({ context, device }: SceneProps) => {
    const root = tgpu.initFromDevice({ device });

    // ── Aspect ratio (computed once; canvas size is fixed after init) ──────
    const canvas     = context.canvas as HTMLCanvasElement;
    const aspectRatio = canvas.clientWidth / canvas.clientHeight;

    // ── Uniforms buffer ────────────────────────────────────────────────────
    const uniformsBuffer = root
      .createBuffer(TyreUniforms, {
        time:        0,
        dt:          DT,
        touchActive: 0,
        aspectRatio,
      })
      .$usage('uniform');

    // ── Physics state buffer ────────────────────────────────────────────────
    const physicsBuffer = root
      .createBuffer(PhysicsState, {
        angularVelocity: DEFAULT_ANGULAR_VELOCITY,
        angle:           0,
        touchDuration:   0,
        _pad:            0,
      })
      .$usage('storage');

    // ── Smoke particles buffer (all dead at start, seeds spread uniformly) ──
    const initialParticles: d.Infer<typeof SmokeParticle>[] = Array.from(
      { length: MAX_PARTICLES },
      (_, i) => ({
        position: d.vec2f(0, 0),
        velocity: d.vec2f(0, 0),
        life:     0,
        seed:     i / MAX_PARTICLES,
        _pad0:    0,
        _pad1:    0,
      }),
    );
    const particlesBuffer = root
      .createBuffer(SmokeParticleBuffer, initialParticles)
      .$usage('storage');

    // ── Pre-baked tyre vertex buffer ────────────────────────────────────────
    const tyreVertexBuffer = root
      .createBuffer(TyreVertexBuffer, buildTyreVertices())
      .$usage('storage');

    // ── Bind groups ─────────────────────────────────────────────────────────
    const computeBindGroup = root.createBindGroup(tyreComputeLayout, {
      uniforms:  uniformsBuffer,
      physics:   physicsBuffer,
      particles: particlesBuffer,
    });

    const tyreRenderBindGroup = root.createBindGroup(tyreRenderLayout, {
      uniforms:     uniformsBuffer,
      physics:      physicsBuffer,
      tyreVertices: tyreVertexBuffer,
    });

    const particleBindGroup = root.createBindGroup(particleRenderLayout, {
      uniforms:  uniformsBuffer,
      particles: particlesBuffer,
    });

    // ── Pipelines ────────────────────────────────────────────────────────────
    const physicsPipeline = root.createComputePipeline({ compute: physicsShader });
    const smokePipeline   = root.createComputePipeline({ compute: smokeShader });

    const tyrePipeline = root.createRenderPipeline({
      vertex:    tyreVertexShader,
      fragment:  tyreFragmentShader,
      primitive: { topology: 'triangle-list' },
    });

    const smokePRPipeline = root.createRenderPipeline({
      vertex:    particleVertexShader,
      fragment:  particleFragmentShader,
      primitive: { topology: 'triangle-list' },
    });

    const smokeDispatch = Math.ceil(MAX_PARTICLES / WORKGROUP_SIZE);

    let frameTime = 0;

    // ── Render loop ───────────────────────────────────────────────────────────
    return (/* timestamp: number */) => {
      frameTime += DT;

      uniformsBuffer.write({
        time:        frameTime,
        dt:          DT,
        touchActive: touchRef.current.active ? 1 : 0,
        aspectRatio,
      });

      // Physics compute (1 thread)
      physicsPipeline.with(computeBindGroup).dispatchWorkgroups(1);

      // Smoke compute (one thread per particle)
      smokePipeline.with(computeBindGroup).dispatchWorkgroups(smokeDispatch);

      // Tyre geometry render (clear the frame)
      tyrePipeline
        .withColorAttachment({
          view:       context,
          clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1 },
          loadOp:     'clear',
          storeOp:    'store',
        })
        .with(tyreRenderBindGroup)
        .draw(960); // TOTAL_TYRE_VERTS

      // Smoke particle render (additive over tyre frame)
      smokePRPipeline
        .withColorAttachment({
          view:    context,
          loadOp:  'load',
          storeOp: 'store',
        })
        .with(particleBindGroup)
        .draw(6, MAX_PARTICLES);
    };
  }, []);

  const canvasRef = useWebGPU(scene);

  // ── Touch handlers ────────────────────────────────────────────────────────
  const layoutSizeRef = useRef({ width: 1, height: 1 });

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    layoutSizeRef.current = { width, height };
  };

  const handleTouchStart = (e: GestureResponderEvent) => {
    void e;
    touchRef.current = { active: true };
  };

  const handleTouchEnd = () => {
    touchRef.current = { active: false };
  };

  return (
    <View
      style={styles.container}
      onLayout={handleLayout}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponder={() => true}
      onResponderGrant={handleTouchStart}
      onResponderRelease={handleTouchEnd}
      onResponderTerminate={handleTouchEnd}
    >
      <Canvas ref={canvasRef} style={styles.canvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0d',
  },
  canvas: {
    flex: 1,
  },
});
