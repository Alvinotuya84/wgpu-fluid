// ─── Mesh rendering (Phong lighting) ─────────────────────────────────────────
//
// Bind group 0: frame-level (view, proj, lighting) — set once per frame
// Bind group 1: per-mesh    (model matrix, base colour) — set per draw call

export const MESH_SHADER = /* wgsl */ `

struct FrameUniforms {
  view:      mat4x4<f32>,  // offset   0
  proj:      mat4x4<f32>,  // offset  64
  lightDir:  vec4<f32>,    // offset 128 (xyz = normalised light direction)
  cameraPos: vec4<f32>,    // offset 144 (xyz = world-space camera position)
  isBraking: u32,          // offset 160
  time:      f32,          // offset 164
  _p0:       f32,
  _p1:       f32,
};

struct MeshUniforms {
  model:     mat4x4<f32>,  // offset  0
  baseColor: vec4<f32>,    // offset 64
};

@group(0) @binding(0) var<uniform> frame : FrameUniforms;
@group(1) @binding(0) var<uniform> mesh  : MeshUniforms;

struct VertIn {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
};

struct VertOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       norm : vec3<f32>,
  @location(1)       wpos : vec3<f32>,
};

@vertex
fn vs(in: VertIn) -> VertOut {
  let m       = mesh.model;
  let wpos4   = m * vec4<f32>(in.position, 1.0);

  // Extract rotation (upper-left 3×3) for normal transform.
  // Valid for rotation-only model matrices (no non-uniform scale).
  let m3   = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
  let wnorm = normalize(m3 * in.normal);

  var out : VertOut;
  out.clip  = frame.proj * frame.view * wpos4;
  out.norm  = wnorm;
  out.wpos  = wpos4.xyz;
  return out;
}

@fragment
fn fs(in: VertOut) -> @location(0) vec4<f32> {
  let L   = normalize(frame.lightDir.xyz);
  let V   = normalize(frame.cameraPos.xyz - in.wpos);
  let H   = normalize(L + V);
  let N   = normalize(in.norm);

  let ambient  = 0.22;
  let diffuse  = max(dot(N, L), 0.0) * 0.65;
  let specular = pow(max(dot(N, H), 0.0), 48.0) * 0.30;

  let col = mesh.baseColor.rgb * (ambient + diffuse) + vec3<f32>(specular);
  return vec4<f32>(col, 1.0);
}
`;

// ─── Smoke particle compute  ──────────────────────────────────────────────────
//
// Bind group 0: binding 0 = particles (storage r/w)
//               binding 1 = params    (uniform)

export const SMOKE_COMPUTE_SHADER = /* wgsl */ `

struct Particle {
  pos : vec4<f32>,  // xyz = world position, w = life [0..1]; 0 = dead
  vel : vec4<f32>,  // xyz = velocity, w = per-particle seed [0..1)
};

struct SmokeParams {
  aggressive : u32,  // 0 = inactive, 1 = aggressive braking (>800 ms)
  _p0        : u32,
  dt         : f32,
  time       : f32,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform>             params    : SmokeParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= 256u) { return; }

  var p = particles[i];

  if (p.pos.w > 0.0) {
    // ── Update live particle ────────────────────────────────────────────────
    p.pos = vec4<f32>(
      p.pos.x + p.vel.x * params.dt,
      p.pos.y + p.vel.y * params.dt,
      p.pos.z + p.vel.z * params.dt,
      max(p.pos.w - params.dt / 1.5, 0.0),
    );
    // smoke rises; x/z drift damps slowly
    p.vel = vec4<f32>(
      p.vel.x * 0.985,
      p.vel.y + params.dt * 0.35,
      p.vel.z * 0.985,
      p.vel.w,
    );

  } else if (params.aggressive == 1u) {
    // ── Maybe respawn (staggered via seed+time hash) ────────────────────────
    let seed  = p.vel.w;
    let h     = fract(seed * 127.1 + params.time * 311.7 + 43758.5);
    if (h < 0.13) {
      let r1 = fract(seed * 269.5 + params.time * 183.3 + 12345.6);
      let r2 = fract(seed * 113.7 + params.time *  97.1 + 98765.4);
      let r3 = fract(seed * 457.3 + params.time * 251.9 + 54321.0);

      // Spawn at tyre contact patch (y ≈ -0.4) with upward burst
      p.pos = vec4<f32>(
        (r1 - 0.5) * 0.25,      // small x scatter
        -0.40,                    // tyre bottom in world space
        (r2 - 0.5) * 0.25,      // small z scatter
        0.9 + r3 * 0.1,         // near-full life
      );
      p.vel = vec4<f32>(
        (r1 - 0.5) * 0.6,       // sideways
        0.5 + r3 * 0.4,         // upward
        (r2 - 0.5) * 0.6,       // forward/back scatter
        seed,
      );
    }
  }

  particles[i] = p;
}
`;

// ─── Smoke particle billboard render  ────────────────────────────────────────
//
// Bind group 0: binding 0 = frame uniforms  (uniform)
// Bind group 1: binding 0 = particles       (storage, read)

export const SMOKE_RENDER_SHADER = /* wgsl */ `

struct FrameUniforms {
  view      : mat4x4<f32>,
  proj      : mat4x4<f32>,
  lightDir  : vec4<f32>,
  cameraPos : vec4<f32>,
  isBraking : u32,
  time      : f32,
  _p0       : f32,
  _p1       : f32,
};

struct Particle {
  pos : vec4<f32>,
  vel : vec4<f32>,
};

@group(0) @binding(0) var<uniform>        frame     : FrameUniforms;
@group(1) @binding(0) var<storage, read>  particles : array<Particle>;

// Six vertices for a quad (two triangles, CCW)
var<private> corners : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
  vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
);

struct SmokeOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       life : f32,
};

@vertex
fn vs(
  @builtin(vertex_index)   vid : u32,
  @builtin(instance_index) iid : u32,
) -> SmokeOut {
  let p    = particles[iid];
  let life = p.pos.w;

  if (life <= 0.0) {
    // Push dead particle off-screen — GPU still needs 6 vertices per instance
    return SmokeOut(vec4<f32>(10.0, 10.0, 0.0, 1.0), 0.0);
  }

  // Camera right/up from the view matrix (column-major: view[col][row])
  let right = vec3<f32>(frame.view[0][0], frame.view[1][0], frame.view[2][0]);
  let up    = vec3<f32>(frame.view[0][1], frame.view[1][1], frame.view[2][1]);

  let corner = corners[vid];
  // Quads grow as particles age
  let size   = 0.10 + (1.0 - life) * 0.18;
  let wpos   = p.pos.xyz + right * corner.x * size + up * corner.y * size;

  return SmokeOut(frame.proj * frame.view * vec4<f32>(wpos, 1.0), life);
}

@fragment
fn fs(in: SmokeOut) -> @location(0) vec4<f32> {
  // Fade out towards edges via soft circle fallback in screen space
  let alpha = in.life * 0.55;
  return vec4<f32>(0.72, 0.72, 0.76, alpha);
}
`;
