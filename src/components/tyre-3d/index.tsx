import { mat4 } from 'wgpu-matrix';
import React, { useCallback, useRef } from 'react';
import {
  PixelRatio,
  StyleSheet,
  View,
  type GestureResponderEvent,
} from 'react-native';
import { Canvas, useCanvasRef, useDevice } from 'react-native-webgpu';

import { loadWheelGLB, type MeshData } from './loader';
import { MESH_SHADER, SMOKE_COMPUTE_SHADER, SMOKE_RENDER_SHADER } from './shaders';

// ── Constants ─────────────────────────────────────────────────────────────────

const DT                    = 1 / 60;
const MAX_ANGULAR_VELOCITY  = 3.0;
const PARTICLE_COUNT        = 256;
const WORKGROUP_SIZE        = 64;
const AGGRESSIVE_THRESHOLD  = 0.8; // seconds

// Mesh colours: rubber, alloy, iron, caliper red
const MESH_COLORS: Record<string, [number, number, number]> = {
  'tire-low':     [0.06, 0.06, 0.07],
  'wheel-full':   [0.68, 0.72, 0.78],
  'brake-disc':   [0.30, 0.26, 0.24],
  'brake-caliper':[0.82, 0.12, 0.09],
};

const SPINNING_MESHES = new Set(['tire-low', 'wheel-full', 'brake-disc']);

// Normalised light direction (1, 2, 1)
const LIGHT_DIR = [1 / Math.sqrt(6), 2 / Math.sqrt(6), 1 / Math.sqrt(6)];

// ── WebGPU hook ───────────────────────────────────────────────────────────────

type SceneSetup = (ctx: GPUCanvasContext, device: GPUDevice) => (() => void) | Promise<() => void>;

function useWebGPU(setup: SceneSetup) {
  const { device } = useDevice();
  const canvasRef  = useCanvasRef();

  React.useEffect(() => {
    let animId: number;
    let active = true;

    (async () => {
      const canvas = canvasRef.current;
      if (!canvas || !device) return;

      const context = canvas.getContext('webgpu');
      if (!context) return;

      const ctx = context.canvas as HTMLCanvasElement;
      ctx.width  = ctx.clientWidth  * PixelRatio.get();
      ctx.height = ctx.clientHeight * PixelRatio.get();

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'premultiplied' });

      const renderFn = await setup(context, device);
      if (!active) return;

      const loop = () => {
        renderFn();
        context.present();
        animId = requestAnimationFrame(loop);
      };
      animId = requestAnimationFrame(loop);
    })();

    return () => {
      active = false;
      cancelAnimationFrame(animId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [device]);

  return canvasRef;
}

// ── GPU resource helpers ──────────────────────────────────────────────────────

function createBuffer(
  device: GPUDevice,
  data:   BufferSource,
  usage:  GPUBufferUsageFlags,
): GPUBuffer {
  const buf = device.createBuffer({ size: (data as ArrayBuffer).byteLength, usage, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data as ArrayBuffer));
  buf.unmap();
  return buf;
}

// ── Uniform buffer helpers ────────────────────────────────────────────────────

// FrameUniforms layout (176 bytes):
//   view[0..15]        → f32 indices  0-15  (64 bytes)
//   proj[16..31]       → f32 indices 16-31  (64 bytes)
//   lightDir[32..35]   → f32 indices 32-35  (16 bytes, w=0)
//   cameraPos[36..39]  → f32 indices 36-39  (16 bytes, w=1)
//   isBraking          → u32 at byte 160    (float index 40)
//   time               → f32 index 41
//   _p0, _p1           → f32 indices 42-43
const FRAME_BUF_SIZE = 176;

function writeFrameUniforms(
  buf:       GPUBuffer,
  queue:     GPUQueue,
  view:      Float32Array,
  proj:      Float32Array,
  camPos:    readonly [number, number, number],
  isBraking: boolean,
  time:      number,
) {
  const ab  = new ArrayBuffer(FRAME_BUF_SIZE);
  const f32 = new Float32Array(ab);
  const u32 = new Uint32Array(ab);
  f32.set(view, 0);
  f32.set(proj, 16);
  f32[32] = LIGHT_DIR[0]; f32[33] = LIGHT_DIR[1]; f32[34] = LIGHT_DIR[2]; f32[35] = 0;
  f32[36] = camPos[0];     f32[37] = camPos[1];    f32[38] = camPos[2];    f32[39] = 1;
  u32[40] = isBraking ? 1 : 0;
  f32[41] = time;
  queue.writeBuffer(buf, 0, ab);
}

// MeshUniforms layout (80 bytes):
//   model[0..15]   → f32 indices  0-15 (64 bytes)
//   baseColor[16-19] → f32 indices 16-19 (16 bytes)
const MESH_BUF_SIZE = 80;

function writeMeshUniforms(
  buf:   GPUBuffer,
  queue: GPUQueue,
  model: Float32Array,
  color: [number, number, number],
) {
  const ab  = new ArrayBuffer(MESH_BUF_SIZE);
  const f32 = new Float32Array(ab);
  f32.set(model, 0);
  f32[16] = color[0]; f32[17] = color[1]; f32[18] = color[2]; f32[19] = 1;
  queue.writeBuffer(buf, 0, ab);
}

// SmokeParams layout (16 bytes):
//   aggressive u32 offset  0
//   _p0        u32 offset  4
//   dt         f32 offset  8
//   time       f32 offset 12
const SMOKE_PARAMS_SIZE = 16;

function writeSmokeParams(
  buf:        GPUBuffer,
  queue:      GPUQueue,
  aggressive: boolean,
  dt:         number,
  time:       number,
) {
  const ab  = new ArrayBuffer(SMOKE_PARAMS_SIZE);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0] = aggressive ? 1 : 0;
  u32[1] = 0;
  f32[2] = dt;
  f32[3] = time;
  queue.writeBuffer(buf, 0, ab);
}

// ── Tyre3DCanvas component ────────────────────────────────────────────────────

export function Tyre3DCanvas() {
  // Physics state (CPU side)
  const physicsRef = useRef({ angularVelocity: 3.0, angularAngle: 0.0 });

  // Touch + orbit state
  const touchRef = useRef({
    active:      false,
    isDragging:  false,
    holdDuration: 0,
    prevX:       0,
    prevY:       0,
  });

  // Spherical orbit state for camera
  // Initial camera at (0, 1.5, 4) → radius ≈ 4.27, phi ≈ 0.357, theta = 0
  const orbitRef = useRef({ theta: 0, phi: 0.357, radius: 4.27 });

  const setup = useCallback<SceneSetup>(async (context, device) => {
    // ── 1. Load GLB meshes ──────────────────────────────────────────────────
    const meshes = await loadWheelGLB();

    // ── 2. Canvas dimensions for projection ────────────────────────────────
    const canvas = context.canvas as HTMLCanvasElement;
    const aspectRatio = canvas.clientWidth / canvas.clientHeight;

    // ── 3. Shader modules ───────────────────────────────────────────────────
    const meshModule        = device.createShaderModule({ code: MESH_SHADER });
    const smokeComputeModule = device.createShaderModule({ code: SMOKE_COMPUTE_SHADER });
    const smokeRenderModule  = device.createShaderModule({ code: SMOKE_RENDER_SHADER });

    // ── 4. Bind group layouts ───────────────────────────────────────────────
    const frameLayout = device.createBindGroupLayout({
      entries: [{
        binding:    0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer:     { type: 'uniform' },
      }],
    });

    const meshLayout = device.createBindGroupLayout({
      entries: [{
        binding:    0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer:     { type: 'uniform' },
      }],
    });

    const smokeComputeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const smokeRenderLayout1 = device.createBindGroupLayout({
      entries: [{
        binding:    0,
        visibility: GPUShaderStage.VERTEX,
        buffer:     { type: 'read-only-storage' },
      }],
    });

    // ── 5. Pipeline layouts ─────────────────────────────────────────────────
    const meshPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameLayout, meshLayout],
    });

    const smokeComputePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [smokeComputeLayout],
    });

    const smokeRenderPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameLayout, smokeRenderLayout1],
    });

    // ── 6. Depth texture ────────────────────────────────────────────────────
    const depthTexture = device.createTexture({
      size:   [canvas.width, canvas.height],
      format: 'depth24plus',
      usage:  GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const depthView = depthTexture.createView();

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // ── 7. Render pipelines ─────────────────────────────────────────────────
    const meshPipeline = device.createRenderPipeline({
      layout: meshPipelineLayout,
      vertex: {
        module:     meshModule,
        entryPoint: 'vs',
        buffers: [
          { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
          { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
        ],
      },
      fragment: {
        module:     meshModule,
        entryPoint: 'fs',
        targets: [{ format: presentationFormat }],
      },
      primitive:    { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const smokeRenderPipeline = device.createRenderPipeline({
      layout: smokeRenderPipelineLayout,
      vertex: {
        module:     smokeRenderModule,
        entryPoint: 'vs',
      },
      fragment: {
        module:     smokeRenderModule,
        entryPoint: 'fs',
        targets: [{
          format: presentationFormat,
          blend:  {
            color: { srcFactor: 'src-alpha', dstFactor: 'one',  operation: 'add' },
            alpha: { srcFactor: 'zero',      dstFactor: 'one',  operation: 'add' },
          },
        }],
      },
      primitive:    { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });

    // ── 8. Compute pipeline ─────────────────────────────────────────────────
    const smokeComputePipeline = device.createComputePipeline({
      layout:  smokeComputePipelineLayout,
      compute: { module: smokeComputeModule, entryPoint: 'main' },
    });

    // ── 9. Mesh GPU buffers ─────────────────────────────────────────────────
    interface GPUMesh {
      name:        string;
      posBuf:      GPUBuffer;
      normBuf:     GPUBuffer;
      idxBuf:      GPUBuffer;
      indexFormat: GPUIndexFormat;
      indexCount:  number;
      uniformBuf:  GPUBuffer;
      bindGroup:   GPUBindGroup;
      color:       [number, number, number];
    }

    const gpuMeshes: GPUMesh[] = meshes.map((m: MeshData) => {
      const posBuf  = createBuffer(device, m.positions.buffer as ArrayBuffer, GPUBufferUsage.VERTEX);
      const normBuf = createBuffer(device, m.normals.buffer  as ArrayBuffer, GPUBufferUsage.VERTEX);
      const idxBuf  = createBuffer(device, m.indices.buffer  as ArrayBuffer, GPUBufferUsage.INDEX);

      const uniformBuf = device.createBuffer({
        size:  MESH_BUF_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const bindGroup = device.createBindGroup({
        layout:  meshLayout,
        entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
      });

      return {
        name:        m.name,
        posBuf,
        normBuf,
        idxBuf,
        indexFormat: m.indexFormat,
        indexCount:  m.indices.length,
        uniformBuf,
        bindGroup,
        color:       MESH_COLORS[m.name] ?? [0.5, 0.5, 0.5],
      };
    });

    // ── 10. Frame uniform buffer ────────────────────────────────────────────
    const frameUniformBuf = device.createBuffer({
      size:  FRAME_BUF_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const frameBindGroup = device.createBindGroup({
      layout:  frameLayout,
      entries: [{ binding: 0, resource: { buffer: frameUniformBuf } }],
    });

    // ── 11. Smoke buffers ───────────────────────────────────────────────────
    // Particle layout: pos (vec4 = xyz+life), vel (vec4 = xyz+seed) = 32 bytes
    const particleData = new Float32Array(PARTICLE_COUNT * 8);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particleData[i * 8 + 7] = i / PARTICLE_COUNT; // seed in vel.w
    }

    const particleBuf = createBuffer(
      device,
      particleData.buffer as ArrayBuffer,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );

    const smokeParamsBuf = device.createBuffer({
      size:  SMOKE_PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const smokeComputeBindGroup = device.createBindGroup({
      layout:  smokeComputeLayout,
      entries: [
        { binding: 0, resource: { buffer: particleBuf } },
        { binding: 1, resource: { buffer: smokeParamsBuf } },
      ],
    });

    const smokeRenderBindGroup = device.createBindGroup({
      layout:  smokeRenderLayout1,
      entries: [{ binding: 0, resource: { buffer: particleBuf } }],
    });

    // ── 12. Projection matrix (computed once) ───────────────────────────────
    const projMatrix = mat4.perspective(Math.PI / 4, aspectRatio, 0.1, 100) as Float32Array;

    // Scratch matrices reused every frame
    const viewMatrix = new Float32Array(16);
    const modelSpinning = new Float32Array(16);
    const modelIdentity = mat4.identity() as Float32Array;

    let frameTime = 0;

    // ── 13. Render loop ─────────────────────────────────────────────────────
    return () => {
      frameTime += DT;
      const touch   = touchRef.current;
      const physics = physicsRef.current;
      const orbit   = orbitRef.current;

      // ── Physics update (CPU) ──────────────────────────────────────────────
      const isBraking = touch.active && !touch.isDragging;

      if (isBraking) {
        touch.holdDuration += DT;
        const aggressive = touch.holdDuration > AGGRESSIVE_THRESHOLD;
        physics.angularVelocity *= aggressive ? 0.80 : 0.94;
        physics.angularVelocity  = Math.max(physics.angularVelocity, 0);
      } else {
        physics.angularVelocity = Math.min(physics.angularVelocity * 1.003, MAX_ANGULAR_VELOCITY);
      }
      physics.angularAngle += physics.angularVelocity * DT;

      const aggressive = isBraking && touch.holdDuration > AGGRESSIVE_THRESHOLD;

      // ── Build view matrix from orbit ─────────────────────────────────────
      const { theta, phi, radius } = orbit;
      const cx = radius * Math.cos(phi) * Math.sin(theta);
      const cy = radius * Math.sin(phi);
      const cz = radius * Math.cos(phi) * Math.cos(theta);
      const camPos: [number, number, number] = [cx, cy, cz];

      mat4.lookAt([cx, cy, cz], [0, 0, 0], [0, 1, 0], viewMatrix);

      // ── Spinning model matrix ─────────────────────────────────────────────
      mat4.rotationX(physics.angularAngle, modelSpinning);

      // ── Write frame uniforms ──────────────────────────────────────────────
      writeFrameUniforms(
        frameUniformBuf, device.queue,
        viewMatrix, projMatrix as Float32Array,
        camPos, isBraking, frameTime,
      );

      // ── Write per-mesh uniforms ───────────────────────────────────────────
      for (const gm of gpuMeshes) {
        const model = SPINNING_MESHES.has(gm.name) ? modelSpinning : modelIdentity;
        writeMeshUniforms(gm.uniformBuf, device.queue, model, gm.color);
      }

      // ── Write smoke params ────────────────────────────────────────────────
      writeSmokeParams(smokeParamsBuf, device.queue, aggressive, DT, frameTime);

      // ── Encode GPU commands ───────────────────────────────────────────────
      const encoder = device.createCommandEncoder();

      // Smoke compute pass
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(smokeComputePipeline);
      computePass.setBindGroup(0, smokeComputeBindGroup);
      computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE));
      computePass.end();

      // Main render pass
      const colorView = context.getCurrentTexture().createView();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view:       colorView,
          clearValue: { r: 0.06, g: 0.06, b: 0.09, a: 1 },
          loadOp:     'clear',
          storeOp:    'store',
        }],
        depthStencilAttachment: {
          view:              depthView,
          depthClearValue:   1.0,
          depthLoadOp:       'clear',
          depthStoreOp:      'store',
        },
      });

      // Draw all 4 meshes with Phong pipeline
      renderPass.setPipeline(meshPipeline);
      renderPass.setBindGroup(0, frameBindGroup);
      for (const gm of gpuMeshes) {
        renderPass.setBindGroup(1, gm.bindGroup);
        renderPass.setVertexBuffer(0, gm.posBuf);
        renderPass.setVertexBuffer(1, gm.normBuf);
        renderPass.setIndexBuffer(gm.idxBuf, gm.indexFormat);
        renderPass.drawIndexed(gm.indexCount);
      }

      // Draw smoke particles (additive, no depth write)
      renderPass.setPipeline(smokeRenderPipeline);
      renderPass.setBindGroup(0, frameBindGroup);
      renderPass.setBindGroup(1, smokeRenderBindGroup);
      renderPass.draw(6, PARTICLE_COUNT); // 6 verts × 256 instances

      renderPass.end();
      device.queue.submit([encoder.finish()]);
    };
  }, []);

  const canvasRef = useWebGPU(setup);

  // ── Touch / orbit handlers ────────────────────────────────────────────────
  const DRAG_THRESHOLD = 8; // pixels before we switch to orbit mode
  const ORBIT_SPEED    = 0.005;

  const handleTouchStart = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    touchRef.current = {
      active:       true,
      isDragging:   false,
      holdDuration: 0,
      prevX:        locationX,
      prevY:        locationY,
    };
  };

  const handleTouchMove = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    const t  = touchRef.current;
    const dx = locationX - t.prevX;
    const dy = locationY - t.prevY;

    if (!t.isDragging && Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD) {
      t.isDragging = true;
    }

    if (t.isDragging) {
      const orbit = orbitRef.current;
      orbit.theta -= dx * ORBIT_SPEED;
      orbit.phi    = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, orbit.phi + dy * ORBIT_SPEED));
    }

    t.prevX = locationX;
    t.prevY = locationY;
  };

  const handleTouchEnd = () => {
    touchRef.current.active       = false;
    touchRef.current.isDragging   = false;
    touchRef.current.holdDuration = 0;
  };

  return (
    <View
      style={styles.container}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponder={() => true}
      onResponderGrant={handleTouchStart}
      onResponderMove={handleTouchMove}
      onResponderRelease={handleTouchEnd}
      onResponderTerminate={handleTouchEnd}
    >
      <Canvas ref={canvasRef} style={styles.canvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f17' },
  canvas:    { flex: 1 },
});
