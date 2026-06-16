import {
  useConfigureContext,
  useFrame,
  useRoot,
  useUniform,
} from "@typegpu/react";
import { useMemo, useRef } from "react";
import { Canvas } from "react-native-webgpu"; // ← renamed package
import tgpu, { d } from "typegpu";

// ← moved outside component, outside pipeline — module level
const positions = tgpu.const(d.arrayOf(d.vec2f), [
  d.vec2f(0.0, 0.5),
  d.vec2f(-0.5, -0.5),
  d.vec2f(0.5, -0.5),
]);

export function AnimatedTriangle() {
  const root = useRoot();

  const time = useUniform(d.f32, 0);
  const timeRef = useRef(0);

  const pipeline = useMemo(() => {
    return root.createRenderPipeline({
      vertex: ({ $vertexIndex: vid }) => {
        "use gpu";

        const angle = time.value;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const pos = positions.$[vid];

        const rotated = d.vec2f(
          cos * pos.x - sin * pos.y,
          sin * pos.x + cos * pos.y,
        );

        return {
          $position: d.vec4f(rotated, 0, 1),
        };
      },

      fragment: () => {
        "use gpu";
        const r = (Math.sin(time.value) * 0.5 + 0.5) * 0.5;
        return d.vec4f(r, 0.114, 0.941, 1);
      },
    });
  }, [root, time]);

  const { ref, ctxRef } = useConfigureContext({ alphaMode: "premultiplied" });

  useFrame(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;

    timeRef.current += 0.01;
    time.write(timeRef.current);

    pipeline.withColorAttachment({ view: ctx }).draw(3);
    ctx.present?.();
  });

  return <Canvas ref={ref} style={{ aspectRatio: 1 }} transparent />;
}
