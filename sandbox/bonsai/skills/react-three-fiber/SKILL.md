---
name: react-three-fiber
description: Use for any 3D / WebGL / three.js scene in React (@react-three/fiber + drei): meshes, lights, animation along paths, cameras, HTML overlays.
---

# react-three-fiber essentials

- Wrap the scene in `<Canvas camera={{ position: [x, y, z], fov: 50 }}>` from `@react-three/fiber`. Give the
  canvas's parent an explicit size, e.g. `position: fixed; inset: 0` or `width: 100vw; height: 100vh`. Without
  one the canvas stays at the browser default of 300×150.
- Elements are three.js classes in camelCase:
  - `<mesh position={[x,y,z]} rotation={[rx,ry,rz]} scale={s}>` holding a geometry and a material
  - geometries: `<boxGeometry args={[w, h, d]} />`, `<cylinderGeometry args={[rTop, rBottom, h, segments]} />`,
    `<coneGeometry args={[r, h, seg]} />`, `<sphereGeometry args={[r, 32, 32]} />`, `<planeGeometry args={[w, h]} />`
  - materials: `<meshStandardMaterial color="#hex" emissive="#hex" emissiveIntensity={2} metalness roughness />`
  - lights: `<ambientLight intensity={0.4} />`, `<directionalLight position={[...]} />`, `<pointLight color distance intensity />`
  - fog: `<fog attach="fog" args={['#color', near, far]} />`, and `<color attach="background" args={['#color']} />`
  - `<group>` groups children so they can be moved together
- Animation: `const ref = useRef<THREE.Mesh>(null); useFrame((state, delta) => { ref.current!.rotation.y += delta })`.
  `useFrame` must be called inside a component rendered within `<Canvas>`. `state.clock.elapsedTime` gives the time.
- Paths: `const curve = new THREE.CatmullRomCurve3(points, true)`, then `curve.getPointAt(t)` and
  `curve.getTangentAt(t)`. To face along the path, use `obj.lookAt(pos.clone().add(tangent))`.
- drei helpers include `OrbitControls`, `Stars`, `Sky`, `Environment preset="night"`, `Float`, `Text`, `Html`,
  `PerspectiveCamera makeDefault`. Import them from `@react-three/drei`. Hooks such as `useFrame` and `useThree`
  come from `@react-three/fiber`.
- Camera control: inside the Canvas, `const { camera } = useThree()`, then in `useFrame` update `camera.position.lerp(target, 0.1)`
  and `camera.lookAt(x, y, z)`. Don't mount `OrbitControls` while doing this, because it fights the manual camera.
- Night scenes need light: use emissive materials on lamps and windows, plus a moonlit directional light and
  ambient light around 0.3–0.5. Otherwise everything renders black.
- Sharing state with HTML overlays: a module-level mutable object, written in `useFrame` and read by the panel
  through `requestAnimationFrame` or a small subscribe/notify store. Don't call React setState every frame.
- No network assets: drei `Environment preset` and `useGLTF` download files, so avoid them unless asked. Build
  models from primitives; light the scene with lights, `<Sky>` and `<Stars>`.
- Use the documented APIs; don't read library sources in node_modules.

## Minimal complete example
```tsx
import { useRef } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'

function Spinner() {
  const ref = useRef<THREE.Mesh>(null)
  useFrame((_, dt) => { if (ref.current) ref.current.rotation.y += dt })
  return <mesh ref={ref}><boxGeometry args={[1, 1, 1]} /><meshStandardMaterial color="#f4a261" /></mesh>
}

export default function App() {
  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0b1020' }}>
      <Canvas camera={{ position: [3, 2, 4], fov: 50 }}>
        <ambientLight intensity={0.4} /><directionalLight position={[3, 5, 2]} intensity={1.2} />
        <Spinner /><Stars /><OrbitControls />
      </Canvas>
    </div>
  )
}
```
