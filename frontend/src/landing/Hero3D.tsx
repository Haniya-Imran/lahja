// 3D hero: review phrases in three scripts orbit a Transformer "core"; attention arcs connect them.
import { QuadraticBezierLine } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef, type RefObject } from 'react'
import * as THREE from 'three'

const TOKENS: { t: string; s: 'pos' | 'neg' | 'neu'; ur?: boolean }[] = [
  { t: 'بہت زبردست', s: 'pos', ur: true }, { t: 'bohat acha hai', s: 'pos' }, { t: 'great quality', s: 'pos' },
  { t: 'bilkul bekaar', s: 'neg' }, { t: 'مایوس کن', s: 'neg', ur: true }, { t: 'not worth it', s: 'neg' },
  { t: 'theek hai', s: 'neu' }, { t: 'delivery on time', s: 'neu' }, { t: 'قیمت مناسب', s: 'neu', ur: true },
  { t: 'paisa wasool', s: 'pos' }, { t: 'kharab nikla', s: 'neg' },
]
const COL = { pos: '#1F9D7A', neg: '#D84B61', neu: '#7C7997' }

function Core() {
  const g = useRef<THREE.Group>(null)
  useFrame((st, dt) => {
    if (!g.current) return
    g.current.rotation.y += dt * 0.25
    g.current.rotation.x = Math.sin(st.clock.elapsedTime * 0.3) * 0.15
  })
  return (
    <group ref={g}>
      <mesh><icosahedronGeometry args={[0.95, 1]} /><meshStandardMaterial color="#3B36A8" roughness={0.35} metalness={0.25} flatShading /></mesh>
      <mesh scale={1.22}><icosahedronGeometry args={[0.95, 1]} /><meshBasicMaterial color="#1B1A3A" wireframe transparent opacity={0.25} /></mesh>
      {[0, 1, 2].map((i) => (
        <mesh key={i} rotation={[Math.PI / 2 + i * 0.6, i * 0.9, 0]}>
          <torusGeometry args={[1.55 + i * 0.18, 0.018, 8, 120]} />
          <meshStandardMaterial color={i === 1 ? '#E8A33D' : '#3B36A8'} emissive={i === 1 ? '#E8A33D' : '#3B36A8'} emissiveIntensity={0.35} />
        </mesh>
      ))}
    </group>
  )
}

function Orbit({ labels }: { labels: RefObject<(HTMLSpanElement | null)[]> }) {
  const group = useRef<THREE.Group>(null)
  const pos = useMemo(() => TOKENS.map((_, i) => {
    const a = (i / TOKENS.length) * Math.PI * 2
    const r = 2.7 + (i % 3) * 0.35
    return new THREE.Vector3(Math.cos(a) * r, Math.sin(a * 2) * 0.9 + (i % 2 ? 0.5 : -0.5), Math.sin(a) * r * 0.55)
  }), [])
  const tmp = useMemo(() => new THREE.Vector3(), [])
  useFrame((st, dt) => {
    if (!group.current) return
    group.current.rotation.y += dt * 0.08
    group.current.rotation.x += (st.pointer.y * 0.25 - group.current.rotation.x) * 0.04
    group.current.rotation.z += (-st.pointer.x * 0.12 - group.current.rotation.z) * 0.04
    group.current.updateMatrixWorld()
    // Project each token to screen space and move its DOM label (labels live outside the canvas).
    const { width, height } = st.size
    pos.forEach((p, i) => {
      const el = labels.current?.[i]
      if (!el) return
      tmp.copy(p).applyMatrix4(group.current!.matrixWorld)
      const depth = tmp.z
      tmp.project(st.camera)
      el.style.transform = `translate(-50%, -135%) translate(${((tmp.x + 1) / 2) * width}px, ${((1 - tmp.y) / 2) * height}px) scale(${(0.82 + (depth + 3) * 0.06).toFixed(3)})`
      el.style.opacity = (0.55 + Math.min(0.45, (depth + 2) * 0.12)).toFixed(2)
      el.style.zIndex = String(Math.round(depth * 10) + 50)
    })
  })
  return (
    <group ref={group}>
      <Core />
      {TOKENS.map((tk, i) => (
        <group key={i}>
          <QuadraticBezierLine start={[0, 0, 0]} end={pos[i]} mid={pos[i].clone().multiplyScalar(0.5).add(new THREE.Vector3(0, 0.9, 0))}
            color={COL[tk.s]} lineWidth={1.3} transparent opacity={0.55} dashed dashScale={8} />
          <mesh position={pos[i]}><sphereGeometry args={[0.07, 16, 16]} /><meshStandardMaterial color={COL[tk.s]} /></mesh>
        </group>
      ))}
    </group>
  )
}

export default function Hero3D() {
  const labels = useRef<(HTMLSpanElement | null)[]>([])
  return (
    <div className="h3d-wrap">
    <Canvas camera={{ position: [0, 0.6, 8.8], fov: 45 }} dpr={[1, 1.75]} gl={{ antialias: true, alpha: true }} aria-hidden="true">
      <ambientLight intensity={0.9} />
      <directionalLight position={[4, 6, 5]} intensity={1.6} />
      <pointLight position={[-4, -2, 3]} intensity={18} color="#E8A33D" />
      <Orbit labels={labels} />
    </Canvas>
    <div className="h3d-labels" aria-hidden="true">
      {TOKENS.map((tk, i) => <span key={i} ref={(el) => { labels.current[i] = el }} className={`h3d-tok ${tk.s} ${tk.ur ? 'ur' : ''}`} dir="auto">{tk.t}</span>)}
    </div>
    </div>
  )
}
