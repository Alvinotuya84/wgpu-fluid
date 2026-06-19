import { SymbolView } from "expo-symbols";
import React, { useCallback, useRef } from "react";
import {
  PixelRatio,
  Pressable,
  StyleSheet,
  Text,
  View,
  type GestureResponderEvent,
} from "react-native";
import { Canvas, useCanvasRef, useDevice } from "react-native-webgpu";
import { mat4 } from "wgpu-matrix";

import { loadWheelGLB, type MeshData } from "./loader";
import {
  MESH_SHADER,
  SMOKE_COMPUTE_SHADER,
  SMOKE_RENDER_SHADER,
} from "./shaders";

// ── Constants ─────────────────────────────────────────────────────────────────

const DT = 1 / 60;
const MAX_ANGULAR_VELOCITY = 3.0;
const PARTICLE_COUNT = 256;
const WORKGROUP_SIZE = 64;
const AGGRESSIVE_THRESHOLD = 0.8;

// Everything spins except the brake caliper
const STATIC_MESHES = new Set(["brake-caliper"]);
// tire-low uses the tire material; everything else uses the wheel material
const TIRE_MESHES = new Set(["tire-low"]);

const LIGHT_DIR = [1 / Math.sqrt(6), 2 / Math.sqrt(6), 1 / Math.sqrt(6)];

// ── WebGPU hook ───────────────────────────────────────────────────────────────

type SceneSetup = (
  ctx: GPUCanvasContext,
  device: GPUDevice,
) => (() => void) | Promise<() => void>;

function useWebGPU(setup: SceneSetup) {
  const { device } = useDevice();
  const canvasRef = useCanvasRef();

  React.useEffect(() => {
    let animId: number;
    let active = true;

    (async () => {
      const canvas = canvasRef.current;
      if (!canvas || !device) return;

      const context = canvas.getContext("webgpu");
      if (!context) return;

      const ctx = context.canvas as HTMLCanvasElement;
      ctx.width = ctx.clientWidth * PixelRatio.get();
      ctx.height = ctx.clientHeight * PixelRatio.get();

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "premultiplied" });

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
  data: BufferSource,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buf = device.createBuffer({
    size: (data as ArrayBuffer).byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data as ArrayBuffer));
  buf.unmap();
  return buf;
}

// Creates a GPUTexture from raw PNG bytes using createImageBitmap (polyfilled by react-native-webgpu)
async function createGPUTexture(
  device: GPUDevice,
  pngBytes: Uint8Array,
): Promise<GPUTexture> {
  // react-native-webgpu polyfills createImageBitmap to accept ArrayBufferView directly
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bitmap = (await (globalThis as any).createImageBitmap(pngBytes)) as {
    width: number;
    height: number;
  };

  const texture = device.createTexture({
    size: [bitmap.width, bitmap.height],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    { source: bitmap as any },
    { texture },
    [bitmap.width, bitmap.height],
  );
  return texture;
}

// 1×1 ORM texture for materials that lack a metallicRoughness texture
function createSyntheticORM(
  device: GPUDevice,
  roughness: number,
  metallic: number,
): GPUTexture {
  const data = new Uint8Array([
    255,
    Math.round(roughness * 255),
    Math.round(metallic * 255),
    255,
  ]);
  const texture = device.createTexture({
    size: [1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, data, { bytesPerRow: 4 }, [1, 1]);
  return texture;
}

// ── Uniform buffer helpers ────────────────────────────────────────────────────

const FRAME_BUF_SIZE = 176;

function writeFrameUniforms(
  buf: GPUBuffer,
  queue: GPUQueue,
  view: Float32Array,
  proj: Float32Array,
  camPos: readonly [number, number, number],
  isBraking: boolean,
  time: number,
) {
  const ab = new ArrayBuffer(FRAME_BUF_SIZE);
  const f32 = new Float32Array(ab);
  const u32 = new Uint32Array(ab);
  f32.set(view, 0);
  f32.set(proj, 16);
  f32[32] = LIGHT_DIR[0];
  f32[33] = LIGHT_DIR[1];
  f32[34] = LIGHT_DIR[2];
  f32[35] = 0;
  f32[36] = camPos[0];
  f32[37] = camPos[1];
  f32[38] = camPos[2];
  f32[39] = 1;
  u32[40] = isBraking ? 1 : 0;
  f32[41] = time;
  queue.writeBuffer(buf, 0, ab);
}

// 64-byte model matrix only — baseColor now comes from texture
const MESH_BUF_SIZE = 64;

function writeMeshUniforms(
  buf: GPUBuffer,
  queue: GPUQueue,
  model: Float32Array,
) {
  queue.writeBuffer(buf, 0, model.buffer, model.byteOffset, 64);
}

const SMOKE_PARAMS_SIZE = 16;

function writeSmokeParams(
  buf: GPUBuffer,
  queue: GPUQueue,
  aggressive: boolean,
  dt: number,
  time: number,
) {
  const ab = new ArrayBuffer(SMOKE_PARAMS_SIZE);
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
  const physicsRef = useRef({ angularVelocity: 3.0, angularAngle: 0.0 });
  const buttonsRef = useRef({ spin: false, brake: false });
  const touchRef = useRef({
    active: false,
    isDragging: false,
    holdDuration: 0,
    prevX: 0,
    prevY: 0,
  });
  const orbitRef = useRef({ theta: 0, phi: 0.357, radius: 4.27 });

  const setup = useCallback<SceneSetup>(async (context, device) => {
    // ── 1. Load GLB meshes + embedded PNG textures ─────────────────────────
    const { meshes, textures } = await loadWheelGLB();

    // ── 2. Canvas dimensions ───────────────────────────────────────────────
    const canvas = context.canvas as HTMLCanvasElement;
    const aspectRatio = canvas.clientWidth / canvas.clientHeight;

    // ── 3. Shader modules ──────────────────────────────────────────────────
    const meshModule = device.createShaderModule({ code: MESH_SHADER });
    const smokeComputeModule = device.createShaderModule({
      code: SMOKE_COMPUTE_SHADER,
    });
    const smokeRenderModule = device.createShaderModule({
      code: SMOKE_RENDER_SHADER,
    });

    // ── 4. Bind group layouts ──────────────────────────────────────────────
    const frameLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const meshLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
      ],
    });

    // Group 2: sampler + baseColor + normalMap + ORM (one instance per material)
    const materialLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
      ],
    });

    const smokeComputeLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });

    const smokeRenderLayout1 = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    // ── 5. Pipeline layouts ────────────────────────────────────────────────
    const meshPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameLayout, meshLayout, materialLayout],
    });

    const smokeComputePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [smokeComputeLayout],
    });

    const smokeRenderPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [frameLayout, smokeRenderLayout1],
    });

    // ── 6. Depth texture ───────────────────────────────────────────────────
    const depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const depthView = depthTexture.createView();
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    // ── 7. Create GPU textures from embedded PNGs ──────────────────────────
    const [
      tireBaseColorTex,
      tireNormalTex,
      wheelBaseColorTex,
      wheelOrmTex,
      wheelNormalTex,
    ] = await Promise.all([
      createGPUTexture(device, textures.tireBaseColor),
      createGPUTexture(device, textures.tireNormal),
      createGPUTexture(device, textures.wheelBaseColor),
      createGPUTexture(device, textures.wheelMetallicRoughness),
      createGPUTexture(device, textures.wheelNormal),
    ]);

    // Tire has no metallicRoughness texture → synthetic 1×1 (roughness=0.35, metallic=0)
    const tireOrmTex = createSyntheticORM(device, 0.35, 0.0);

    const sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });

    // ── 8. Render pipelines ────────────────────────────────────────────────
    const meshPipeline = device.createRenderPipeline({
      layout: meshPipelineLayout,
      vertex: {
        module: meshModule,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
          }, // pos
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
          }, // norm
          {
            arrayStride: 8,
            attributes: [{ shaderLocation: 2, offset: 0, format: "float32x2" }],
          }, // uv
        ],
      },
      fragment: {
        module: meshModule,
        entryPoint: "fs",
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const smokeRenderPipeline = device.createRenderPipeline({
      layout: smokeRenderPipelineLayout,
      vertex: { module: smokeRenderModule, entryPoint: "vs" },
      fragment: {
        module: smokeRenderModule,
        entryPoint: "fs",
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one",
                operation: "add",
              },
              alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "less",
      },
    });

    // ── 9. Compute pipeline ────────────────────────────────────────────────
    const smokeComputePipeline = device.createComputePipeline({
      layout: smokeComputePipelineLayout,
      compute: { module: smokeComputeModule, entryPoint: "main" },
    });

    // ── 10. Mesh GPU buffers ───────────────────────────────────────────────
    interface GPUMesh {
      name: string;
      posBuf: GPUBuffer;
      normBuf: GPUBuffer;
      uvBuf: GPUBuffer;
      idxBuf: GPUBuffer;
      indexFormat: GPUIndexFormat;
      indexCount: number;
      uniformBuf: GPUBuffer;
      bindGroup: GPUBindGroup; // group 1: model matrix
    }

    const gpuMeshes: GPUMesh[] = meshes.map((m: MeshData) => {
      const posBuf = createBuffer(
        device,
        m.positions.buffer as ArrayBuffer,
        GPUBufferUsage.VERTEX,
      );
      const normBuf = createBuffer(
        device,
        m.normals.buffer as ArrayBuffer,
        GPUBufferUsage.VERTEX,
      );
      const uvBuf = createBuffer(
        device,
        m.uvs.buffer as ArrayBuffer,
        GPUBufferUsage.VERTEX,
      );
      const idxBuf = createBuffer(
        device,
        m.indices.buffer as ArrayBuffer,
        GPUBufferUsage.INDEX,
      );

      const uniformBuf = device.createBuffer({
        size: MESH_BUF_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const bindGroup = device.createBindGroup({
        layout: meshLayout,
        entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
      });

      return {
        name: m.name,
        posBuf,
        normBuf,
        uvBuf,
        idxBuf,
        indexFormat: m.indexFormat,
        indexCount: m.indices.length,
        uniformBuf,
        bindGroup,
      };
    });

    // ── 11. Material bind groups (one per material set) ────────────────────
    const tireMaterialBG = device.createBindGroup({
      layout: materialLayout,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: tireBaseColorTex.createView() },
        { binding: 2, resource: tireNormalTex.createView() },
        { binding: 3, resource: tireOrmTex.createView() },
      ],
    });

    const wheelMaterialBG = device.createBindGroup({
      layout: materialLayout,
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: wheelBaseColorTex.createView() },
        { binding: 2, resource: wheelNormalTex.createView() },
        { binding: 3, resource: wheelOrmTex.createView() },
      ],
    });

    // ── 12. Frame uniform buffer ───────────────────────────────────────────
    const frameUniformBuf = device.createBuffer({
      size: FRAME_BUF_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const frameBindGroup = device.createBindGroup({
      layout: frameLayout,
      entries: [{ binding: 0, resource: { buffer: frameUniformBuf } }],
    });

    // ── 13. Smoke buffers ──────────────────────────────────────────────────
    const particleData = new Float32Array(PARTICLE_COUNT * 8);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particleData[i * 8 + 7] = i / PARTICLE_COUNT;
    }

    const particleBuf = createBuffer(
      device,
      particleData.buffer as ArrayBuffer,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );

    const smokeParamsBuf = device.createBuffer({
      size: SMOKE_PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const smokeComputeBindGroup = device.createBindGroup({
      layout: smokeComputeLayout,
      entries: [
        { binding: 0, resource: { buffer: particleBuf } },
        { binding: 1, resource: { buffer: smokeParamsBuf } },
      ],
    });

    const smokeRenderBindGroup = device.createBindGroup({
      layout: smokeRenderLayout1,
      entries: [{ binding: 0, resource: { buffer: particleBuf } }],
    });

    // ── 14. Projection matrix (computed once) ──────────────────────────────
    const projMatrix = mat4.perspective(
      Math.PI / 4,
      aspectRatio,
      0.1,
      100,
    ) as Float32Array;
    const viewMatrix = new Float32Array(16);
    const modelSpinning = new Float32Array(16);
    const modelIdentity = mat4.identity() as Float32Array;

    let frameTime = 0;

    // ── 15. Render loop ────────────────────────────────────────────────────
    return () => {
      frameTime += DT;
      const touch = touchRef.current;
      const physics = physicsRef.current;
      const orbit = orbitRef.current;

      const buttons = buttonsRef.current;
      const touchBrake = touch.active && !touch.isDragging;
      const isBraking = touchBrake || buttons.brake;

      if (buttons.spin && !isBraking) {
        physics.angularVelocity = Math.min(
          physics.angularVelocity * 1.015,
          MAX_ANGULAR_VELOCITY * 1.5,
        );
        touch.holdDuration = 0;
      } else if (isBraking) {
        touch.holdDuration += DT;
        const aggressive = touch.holdDuration > AGGRESSIVE_THRESHOLD;
        physics.angularVelocity *= aggressive ? 0.8 : 0.94;
        physics.angularVelocity = Math.max(physics.angularVelocity, 0);
      } else {
        physics.angularVelocity = Math.min(
          physics.angularVelocity * 1.003,
          MAX_ANGULAR_VELOCITY,
        );
        touch.holdDuration = 0;
      }
      physics.angularAngle += physics.angularVelocity * DT;

      const aggressive = isBraking && touch.holdDuration > AGGRESSIVE_THRESHOLD;

      const { theta, phi, radius } = orbit;
      const cx = radius * Math.cos(phi) * Math.sin(theta);
      const cy = radius * Math.sin(phi);
      const cz = radius * Math.cos(phi) * Math.cos(theta);
      const camPos: [number, number, number] = [cx, cy, cz];

      mat4.lookAt([cx, cy, cz], [0, 0, 0], [0, 1, 0], viewMatrix);
      mat4.rotationX(physics.angularAngle, modelSpinning);

      writeFrameUniforms(
        frameUniformBuf,
        device.queue,
        viewMatrix,
        projMatrix,
        camPos,
        isBraking,
        frameTime,
      );

      for (const gm of gpuMeshes) {
        const model = STATIC_MESHES.has(gm.name)
          ? modelIdentity
          : modelSpinning;
        writeMeshUniforms(gm.uniformBuf, device.queue, model);
      }

      writeSmokeParams(smokeParamsBuf, device.queue, aggressive, DT, frameTime);

      const encoder = device.createCommandEncoder();
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(smokeComputePipeline);
      computePass.setBindGroup(0, smokeComputeBindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE),
      );
      computePass.end();

      const colorView = context.getCurrentTexture().createView();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: colorView,
            clearValue: { r: 0.06, g: 0.06, b: 0.09, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });

      renderPass.setPipeline(meshPipeline);
      renderPass.setBindGroup(0, frameBindGroup);
      for (const gm of gpuMeshes) {
        const matBG = TIRE_MESHES.has(gm.name)
          ? tireMaterialBG
          : wheelMaterialBG;
        renderPass.setBindGroup(1, gm.bindGroup);
        renderPass.setBindGroup(2, matBG);
        renderPass.setVertexBuffer(0, gm.posBuf);
        renderPass.setVertexBuffer(1, gm.normBuf);
        renderPass.setVertexBuffer(2, gm.uvBuf);
        renderPass.setIndexBuffer(gm.idxBuf, gm.indexFormat);
        renderPass.drawIndexed(gm.indexCount);
      }

      renderPass.setPipeline(smokeRenderPipeline);
      renderPass.setBindGroup(0, frameBindGroup);
      renderPass.setBindGroup(1, smokeRenderBindGroup);
      renderPass.draw(6, PARTICLE_COUNT);

      renderPass.end();
      device.queue.submit([encoder.finish()]);
    };
  }, []);

  const canvasRef = useWebGPU(setup);

  const DRAG_THRESHOLD = 8;
  const ORBIT_SPEED = 0.005;

  const handleTouchStart = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    touchRef.current = {
      active: true,
      isDragging: false,
      holdDuration: 0,
      prevX: locationX,
      prevY: locationY,
    };
  };

  const handleTouchMove = (e: GestureResponderEvent) => {
    const { locationX, locationY } = e.nativeEvent;
    const t = touchRef.current;
    const dx = locationX - t.prevX;
    const dy = locationY - t.prevY;

    if (!t.isDragging && Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD) {
      t.isDragging = true;
    }

    if (t.isDragging) {
      const orbit = orbitRef.current;
      orbit.theta -= dx * ORBIT_SPEED;
      orbit.phi = Math.max(
        -Math.PI * 0.45,
        Math.min(Math.PI * 0.45, orbit.phi + dy * ORBIT_SPEED),
      );
    }

    t.prevX = locationX;
    t.prevY = locationY;
  };

  const handleTouchEnd = () => {
    touchRef.current.active = false;
    touchRef.current.isDragging = false;
  };

  return (
    <View style={styles.container}>
      {/* Canvas + orbit gesture layer — absoluteFill, behind buttons in z-order */}
      <View
        style={StyleSheet.absoluteFill}
        onStartShouldSetResponder={() => true}
        onMoveShouldSetResponder={() => true}
        onResponderGrant={handleTouchStart}
        onResponderMove={handleTouchMove}
        onResponderRelease={handleTouchEnd}
        onResponderTerminate={handleTouchEnd}
      >
        <Canvas ref={canvasRef} style={styles.canvas} />
      </View>

      {/* Buttons are siblings of the gesture layer — onMoveShouldSetResponder
          on the gesture view never fires for touches on these Pressables */}
      <View style={styles.buttonBar}>
        <Pressable
          style={({ pressed }) => [
            styles.btn,
            styles.btnAccel,
            pressed && styles.btnPressed,
          ]}
          onPressIn={() => {
            buttonsRef.current.spin = true;
          }}
          onPressOut={() => {
            buttonsRef.current.spin = false;
          }}
        >
          <SymbolView
            name="bolt.fill"
            style={styles.btnIcon}
            tintColor="#ffffff"
            resizeMode="scaleAspectFit"
          />
          <Text style={styles.btnLabel}>Gas</Text>
        </Pressable>

        <Pressable
          style={({ pressed }) => [
            styles.btn,
            styles.btnBrake,
            pressed && styles.btnPressed,
          ]}
          onPressIn={() => {
            buttonsRef.current.brake = true;
          }}
          onPressOut={() => {
            buttonsRef.current.brake = false;
          }}
        >
          <SymbolView
            name="stop.fill"
            style={styles.btnIcon}
            tintColor="#ffffff"
            resizeMode="scaleAspectFit"
          />
          <Text style={styles.btnLabel}>Brake</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "red" },
  canvas: { flex: 1 },

  buttonBar: {
    position: "absolute",
    bottom: 44,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 32,
  },

  btn: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
    borderWidth: 2,
    backgroundColor: "rgba(12, 12, 20, 0.72)",
  },
  btnAccel: {
    borderColor: "#22C55E",
    shadowColor: "#22C55E",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 8,
    elevation: 8,
  },
  btnBrake: {
    borderColor: "#EF4444",
    shadowColor: "#EF4444",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 8,
    elevation: 8,
  },
  btnPressed: {
    opacity: 0.65,
    transform: [{ scale: 0.93 }],
  },
  btnIcon: {
    width: 32,
    height: 32,
  },
  btnLabel: {
    fontSize: 10,
    fontWeight: "600",
    color: "rgba(255,255,255,0.75)",
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
});
