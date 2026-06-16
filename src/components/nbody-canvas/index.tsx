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

import { gravityShader } from './Gravity';
import { DT, PARTICLE_COUNT, WORKGROUP_SIZE } from './params';
import { particleFragmentShader, particleVertexShader } from './Render';
import { ParticleBuffer, NBodyUniforms, nbodyBindGroupLayout } from './Schemas';

// ─── useWebGPU hook (same as fluid-canvas) ──────────────────────────────────

interface SceneProps {
  context: GPUCanvasContext;
  device: GPUDevice;
  presentationFormat: GPUTextureFormat;
}

type RenderScene = (timestamp: number) => void;
type Scene = (props: SceneProps) => RenderScene | Promise<RenderScene>;

function withValidate<T extends unknown[], R>(
  device: GPUDevice,
  fn: (...args: T) => R,
) {
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
  const canvasRef = useCanvasRef();

  React.useEffect(() => {
    (async () => {
      const ref = canvasRef.current;
      if (!ref || !device) return;

      const context = ref.getContext('webgpu');
      if (!context) throw new Error('Failed to get WebGPU context.');

      const canvas = context.canvas as HTMLCanvasElement;
      canvas.width = canvas.clientWidth * PixelRatio.get();
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

// ─── NBodyCanvas component ───────────────────────────────────────────────────

export function NBodyCanvas() {
  // touch state — updated from RN gestures, read by the GPU each frame
  const touchRef = useRef({ x: 0.5, y: 0.5, active: false });

  const scene = useCallback(({ context, device }: SceneProps) => {
    // ── 1. Init TypeGPU root from the device we already have ───────────────
    const root = tgpu.initFromDevice({ device });

    // ── 2. Seed initial particle state ──────────────────────────────────────
    // Random positions scattered across the sim space, small random
    // velocities so the cluster isn't perfectly still on frame one.
    const initialParticles = Array.from({ length: PARTICLE_COUNT }, () => ({
      position: d.vec2f(Math.random(), Math.random()),
      velocity: d.vec2f((Math.random() - 0.5) * 0.05, (Math.random() - 0.5) * 0.05),
      mass: 0.5 + Math.random() * 1.5,
    }));

    // ── 3. Create ping-pong particle buffers ────────────────────────────────
    const bufferA = root.createBuffer(ParticleBuffer, initialParticles).$usage('storage');
    const bufferB = root.createBuffer(ParticleBuffer).$usage('storage');

    // ── 4. Uniforms buffer ────────────────────────────────────────────────
    const uniformsBuffer = root
      .createBuffer(NBodyUniforms, {
        dt: DT,
        touchX: 0.5,
        touchY: 0.5,
        touchActive: 0,
      })
      .$usage('uniform');

    // ── 5. Build bind groups (A→B and B→A for ping-pong) ─────────────────
    const bindGroups = [
      root.createBindGroup(nbodyBindGroupLayout, {
        current: bufferA,
        next: bufferB,
        uniforms: uniformsBuffer,
      }),
      root.createBindGroup(nbodyBindGroupLayout, {
        current: bufferB,
        next: bufferA,
        uniforms: uniformsBuffer,
      }),
    ];

    // ── 6. Build pipelines ───────────────────────────────────────────────
    const gravityPipeline = root.createComputePipeline({ compute: gravityShader });
    const renderPipeline = root.createRenderPipeline({
      vertex: particleVertexShader,
      fragment: particleFragmentShader,
      primitive: { topology: 'triangle-list' },
    });

    const dispatchCount = Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE);

    // ── 7. Frame state ────────────────────────────────────────────────────
    let swap = 0; // 0 = current is bufferA, 1 = current is bufferB

    // ── 8. Render loop ────────────────────────────────────────────────────
    return (timestamp: number) => {
      const touch = touchRef.current;

      uniformsBuffer.write({
        dt: DT,
        touchX: touch.x,
        touchY: touch.y,
        touchActive: touch.active ? 1 : 0,
      });

      // gravity step: read from bindGroups[swap].current, write into .next
      gravityPipeline.with(bindGroups[swap]).dispatchWorkgroups(dispatchCount);

      // render the buffer we just wrote into — that's "current" on the
      // *other* bind group
      const renderBindGroup = bindGroups[1 - swap];

      renderPipeline
        .withColorAttachment({
          view: context,
          clearValue: { r: 0, g: 0, b: 0.03, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        })
        .with(renderBindGroup)
        .draw(6, PARTICLE_COUNT);

      swap = 1 - swap;
    };
  }, []);

  const canvasRef = useWebGPU(scene);

  // ── Touch handlers ──────────────────────────────────────────────────────
  // `nativeEvent.target` is just a numeric view tag, not a measurable
  // object — measure the container ourselves via onLayout instead.
  const layoutSizeRef = useRef({ width: 1, height: 1 });

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    layoutSizeRef.current = { width, height };
  };

  const handleTouchStart = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    const { width, height } = layoutSizeRef.current;
    touchRef.current = {
      x: locationX / width,
      y: 1 - locationY / height, // flip Y — GPU Y is bottom-up
      active: true,
    };
  };

  const handleTouchMove = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    const { width, height } = layoutSizeRef.current;
    touchRef.current = {
      x: locationX / width,
      y: 1 - locationY / height,
      active: true,
    };
  };

  const handleTouchEnd = () => {
    touchRef.current = { ...touchRef.current, active: false };
  };

  return (
    <View
      style={styles.container}
      onLayout={handleLayout}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponder={() => true}
      onResponderGrant={handleTouchStart}
      onResponderMove={handleTouchMove}
      onResponderRelease={handleTouchEnd}
    >
      <Canvas ref={canvasRef} style={styles.canvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  canvas: {
    flex: 1,
  },
});
