# wgpu-fluid

A real-time, touch-interactive fluid simulation that runs entirely on the
mobile GPU — no game engine, no Three.js, just [WebGPU](https://www.w3.org/TR/webgpu/)
compute shaders driven from React Native.

Built with [Expo](https://expo.dev), [TypeGPU](https://typegpu.com), and
[`react-native-webgpu`](https://github.com/wcandillon/react-native-webgpu).

## What it does

Touch the screen and drag — each frame, a stack of compute shaders solves an
incompressible Navier–Stokes fluid on a 128×128 grid (16,384 cells) and
renders the resulting dye field as a fullscreen quad. Every cell on the grid
updates in parallel, on-device, with no server or cloud API involved.

Per frame, the simulation runs 5 compute shaders sequentially (each one reads
the previous pass's output, via ping-pong buffers), for 24 GPU dispatches total:

1. **Splat** — injects velocity and dye at the touch point
2. **Advect** — moves velocity and dye along the velocity field
   (semi-Lagrangian advection)
3. **Divergence** — measures how much fluid is "piling up" at each cell
4. **Pressure solve** — 20 Jacobi iterations to find the pressure field
5. **Gradient subtract** — subtracts the pressure gradient from velocity to
   enforce incompressibility

...followed by a render pass that draws the dye field as a fullscreen quad.

The project also includes a second, work-in-progress demo: a brute-force
O(n²) N-body gravity simulation (`src/components/nbody-canvas/`) built on the
same compute-shader plumbing, with particles attracted to each other and to
the touch point. It's not currently wired into a screen.

## Tech stack

- [Expo](https://expo.dev) + [Expo Router](https://docs.expo.dev/router/introduction/) (file-based routing)
- [`react-native-webgpu`](https://github.com/wcandillon/react-native-webgpu) — native WebGPU bridge for React Native, by William Candillon
- [TypeGPU](https://typegpu.com) — typed WGSL/shader authoring layer, by Software Mansion
- TypeScript throughout, including inside the shaders themselves (TypeGPU's `"use gpu"` functions)

## Project structure

```
src/
  app/
    index.tsx              # Home tab
    explore.tsx             # Explore tab — the fluid sim, full-bleed
    _layout.tsx              # Tab navigator
  components/
    fluid-canvas/
      index.tsx              # React component, WebGPU setup, render loop, touch handling
      Schemas.ts              # FluidCell/FluidGrid structs + shared bind group layout
      params.ts                # Tunable constants (grid size, viscosity, dt, ...)
      Splat.ts                  # touch injection compute shader
      Advect.ts                  # advection compute shader
      Project.ts                  # divergence / pressure / gradient-subtract shaders
      Render.ts                    # fullscreen-quad vertex + fragment shaders
    nbody-canvas/
      index.tsx, Schemas.ts, Gravity.ts, Render.ts, params.ts
      # same shape as fluid-canvas, but a particle-based N-body gravity sim
```

## Getting started

This app uses a native module (`react-native-webgpu`), so it needs a custom
development build — it will **not** run inside Expo Go.

```bash
npm install

# build and run the native dev client once
npx expo run:ios       # or
npx expo run:android

# then, for subsequent runs, just start the dev server
npm run start
```

Other scripts:

```bash
npm run web      # web (requires browser WebGPU support, e.g. recent Chrome)
npm run lint      # expo lint
```

## Tuning the simulation

All the fluid-sim constants live in [`src/components/fluid-canvas/params.ts`](src/components/fluid-canvas/params.ts):

| Constant            | Default | Effect                                      |
| -------------------- | ------- | -------------------------------------------- |
| `GRID_SIZE`           | 128     | Simulation resolution (cells per side)        |
| `JACOBI_ITERATIONS`    | 20      | Pressure solve accuracy vs. cost              |
| `VISCOSITY`             | 0.0001  | Fluid thickness                                |
| `DIFFUSION`              | 0.0001  | How fast dye diffuses                          |
| `DT`                      | 0.016   | Simulation time step per frame                  |
| `WORKGROUP_SIZE`           | 8       | Compute shader workgroup size (8×8 = 64 threads) |

## Credits

- [TypeGPU](https://github.com/software-mansion/TypeGPU) by [Software Mansion](https://swmansion.com)
- [`react-native-webgpu`](https://github.com/wcandillon/react-native-webgpu) by [William Candillon](https://github.com/wcandillon)

## License

MIT — see [LICENSE](LICENSE).
