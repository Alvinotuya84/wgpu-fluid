import React, { useCallback, useRef } from 'react';
import { PixelRatio, StyleSheet, View, type GestureResponderEvent } from 'react-native';
import { Canvas, useCanvasRef, useDevice } from 'react-native-webgpu';
import tgpu from 'typegpu';

import { advectShader } from './Advect';
import {
    DIFFUSION,
    DT,
    GRID_SIZE,
    JACOBI_ITERATIONS,
    VISCOSITY,
    WORKGROUP_SIZE,
} from './params';
import { divergenceShader, gradientSubtractShader, pressureShader } from './Project';
import { fluidFragmentShader, fluidVertexShader } from './Render';
import { FluidGrid, FluidUniforms, fluidBindGroupLayout } from './Schemas';
import { splatShader } from './Splat';

// ─── useWebGPU hook (from the example project) ─────────────────────────────

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

// ─── FluidCanvas component ──────────────────────────────────────────────────

export function FluidCanvas() {
  // touch state — updated from RN gestures, read by the GPU each frame
  const touchRef = useRef({ x: 0.5, y: 0.5, active: false });

  const scene = useCallback(({ context, device }: SceneProps) => {
    // ── 1. Init TypeGPU root from the device we already have ───────────────
    const root = tgpu.initFromDevice({ device });

    // ── 2. Create ping-pong grid buffers ────────────────────────────────────
    // (storage buffers are zero-initialized by WebGPU — no manual fill needed)
    const bufferA = root.createBuffer(FluidGrid).$usage('storage');
    const bufferB = root.createBuffer(FluidGrid).$usage('storage');

    // ── 3. Uniforms buffer ────────────────────────────────────────────────
    const uniformsBuffer = root
      .createBuffer(FluidUniforms, {
        gridSize: GRID_SIZE,
        dt: DT,
        viscosity: VISCOSITY,
        diffusion: DIFFUSION,
        time: 0,
        touchX: 0.5,
        touchY: 0.5,
        touchActive: 0,
      })
      .$usage('uniform');

    // ── 4. Build bind groups (A→B and B→A for ping-pong) ─────────────────
    const bindGroupAB = root.createBindGroup(fluidBindGroupLayout, {
      current: bufferA,
      next: bufferB,
      uniforms: uniformsBuffer,
    });

    const bindGroupBA = root.createBindGroup(fluidBindGroupLayout, {
      current: bufferB,
      next: bufferA,
      uniforms: uniformsBuffer,
    });

    // ── 5. Build compute pipelines ────────────────────────────────────────
    const advectPipeline     = root.createComputePipeline({ compute: advectShader });
    const divergencePipeline = root.createComputePipeline({ compute: divergenceShader });
    const pressurePipeline   = root.createComputePipeline({ compute: pressureShader });
    const gradientPipeline   = root.createComputePipeline({ compute: gradientSubtractShader });
    const splatPipeline      = root.createComputePipeline({ compute: splatShader });

    // ── 6. Build render pipeline ──────────────────────────────────────────
    const renderPipeline = root.createRenderPipeline({
      vertex: fluidVertexShader,
      fragment: fluidFragmentShader,
      primitive: { topology: 'triangle-list' },
    });

    // ── 7. Workgroup dispatch count ───────────────────────────────────────
    const dispatchCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);

    // ── 8. Frame state ────────────────────────────────────────────────────
    let pingPong = 0; // 0 = A→B, 1 = B→A
    let frameTime = 0;

    // helper: dispatch a compute pipeline against whichever bind group
    // matches the current ping-pong state, then flip it
    const runCompute = (pipeline: typeof advectPipeline) => {
      const bindGroup = pingPong === 0 ? bindGroupAB : bindGroupBA;
      pipeline.with(bindGroup).dispatchWorkgroups(dispatchCount, dispatchCount);
      pingPong = 1 - pingPong;
    };

    // ── 9. Render loop ────────────────────────────────────────────────────
    return (timestamp: number) => {
      frameTime += DT;

      const touch = touchRef.current;

      // update uniforms each frame
      uniformsBuffer.write({
        gridSize: GRID_SIZE,
        dt: DT,
        viscosity: VISCOSITY,
        diffusion: DIFFUSION,
        time: frameTime,
        touchX: touch.x,
        touchY: touch.y,
        touchActive: touch.active ? 1 : 0,
      });

      // splat touch input
      runCompute(splatPipeline);

      // advect velocity + dye
      runCompute(advectPipeline);

      // compute divergence
      runCompute(divergencePipeline);

      // jacobi pressure solve — multiple iterations
      for (let i = 0; i < JACOBI_ITERATIONS; i++) runCompute(pressurePipeline);

      // subtract gradient
      runCompute(gradientPipeline);

      // render — read from whichever buffer ping-pong left as "current"
      const renderBindGroup = pingPong === 0 ? bindGroupAB : bindGroupBA;

      renderPipeline
        .withColorAttachment({
          view: context,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        })
        .with(renderBindGroup)
        .draw(6); // 6 vertices = fullscreen quad
    };
  }, []);

  const canvasRef = useWebGPU(scene);

  // ── Touch handlers ──────────────────────────────────────────────────────
  const handleTouchStart = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    // @ts-ignore — nativeEvent has the view dimensions
    const { width, height } = e.nativeEvent.target || { width: 1, height: 1 };
    touchRef.current = {
      x: locationX / (width || 300),
      y: 1 - locationY / (height || 300), // flip Y — GPU Y is bottom-up
      active: true,
    };
  };

  const handleTouchMove = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    // @ts-ignore
    const { width, height } = e.nativeEvent.target || { width: 1, height: 1 };
    touchRef.current = {
      x: locationX / (width || 300),
      y: 1 - locationY / (height || 300),
      active: true,
    };
  };

  const handleTouchEnd = () => {
    touchRef.current = { ...touchRef.current, active: false };
  };

  return (
    <View
      style={styles.container}
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
