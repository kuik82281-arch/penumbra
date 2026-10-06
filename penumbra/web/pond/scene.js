// 雨池: a pond at night, in the rain. The water is clear and deep blue — you see down to its sandy, pebbled bed, with
// light dancing on it where the beam falls — and every raindrop sets off sketchy rings on the surface. Grass on the
// banks sways; cherry trees in full bloom stand on the banks, glowing softly in the night, and the wind takes their
// petals off over the pond and the banks. One warm white beam from high above shows the ripples wherever you tap;
// it can be switched off (the trees keep glowing). Above the pond hang the memories, one petal each (setMemories):
// petal-sized and lying on the water like the others, a little bigger the more they matter; under the beam they
// brighten and shimmer, and a tap reads one. Drag sideways to walk round the pond.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
const DEPTH = 1.5;
const LIGHT_Y = 32; // far above, out of the picture: only the beam comes down
const LIGHT_R = 4.4;
const RIPPLES = 72;
const TREE = new THREE.Vector3(-8.2, 0, -7.2);
const FOG = new THREE.Color(0x040507);
const PETALS = 460;
function stream(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
/** The pond's outline: a rounded, uneven shape. Negative inside. Mirrored in GLSL below. */
const pondR = (a) => 5.2 + 0.6 * Math.sin(2 * a + 0.5) + 0.35 * Math.sin(3 * a + 1.7) + 0.2 * Math.sin(5 * a);
const pondSD = (x, z) => Math.hypot(x, z) - pondR(Math.atan2(z, x));
/** The ground: banks rising gently from the water's edge, the bed sloping down to the middle. */
function groundY(x, z) {
    const d = pondSD(x, z);
    if (d >= 0)
        return 0.03 + 0.28 * smooth(0, 1.8, d) + 0.06 * Math.sin(x * 0.7) * Math.sin(z * 0.6) * smooth(0.5, 3, d);
    return -0.04 - DEPTH * smooth(0, 3.2, -d) + 0.05 * Math.sin(x * 2.1 + z * 1.3);
}
const GLSL_COMMON = /* glsl */ `
  uniform float uTime; uniform vec3 uLight; uniform float uRadius; uniform float uOn; uniform vec3 uTree; uniform float uWind;
  uniform vec3 uFogColor; uniform float uFogNear; uniform float uFogFar;
  float pondR(float a) { return 5.2 + 0.6 * sin(2.0 * a + 0.5) + 0.35 * sin(3.0 * a + 1.7) + 0.2 * sin(5.0 * a); }
  float pondSD(vec2 p) { return length(p) - pondR(atan(p.y, p.x)); }
  float litAt(vec2 p) { return smoothstep(uRadius * 1.15, uRadius * 0.25, distance(p, uLight.xz)) * uOn; }
  float treeGlow(vec2 p) { vec2 d = p - uTree.xz; return exp(-dot(d, d) / 38.0); }
  vec3 fogged(vec3 col, float depth) { return mix(col, uFogColor, smoothstep(uFogNear, uFogFar, depth)); }
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
`;
export class Pond {
    renderer;
    scene = new THREE.Scene();
    camera;
    host;
    raf = 0;
    ro;
    clock = new THREE.Clock();
    u = {
        uTime: { value: 0 },
        uLight: { value: new THREE.Vector3() },
        uRadius: { value: LIGHT_R },
        uOn: { value: 1 },
        uWind: { value: 1 },
        uTree: { value: TREE.clone().setY(9) },
        uFogColor: { value: FOG },
        uFogNear: { value: 20 },
        uFogFar: { value: 50 },
        uDrops: { value: Array.from({ length: RIPPLES }, () => new THREE.Vector4(0, 0, -100, 1)) },
    };
    blossomU = { uTime: this.u.uTime, uScale: { value: 400 }, uWind: this.u.uWind };
    dropIndex = 0;
    dropDebt = 0;
    spot;
    cone;
    pads = [];
    fish = [];
    blossomPts = [];
    petals = [];
    petalMesh = null;
    target = new THREE.Vector2(0.5, 0.8);
    pos = new THREE.Vector2(0.5, 0.8);
    on = true;
    windOn = true;
    yaw = 0.35;
    yawGoal = 0.35;
    down = null;
    raycaster = new THREE.Raycaster();
    surface = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    r = stream(808);
    small;
    onFirstTouch = () => { };
    /** A memory petal was tapped (its id), or the tap went elsewhere (null). */
    onPick = () => { };
    mems = [];
    memMesh = null;
    memGlow = null;
    picked = null;
    constructor(host) {
        this.host = host;
        this.small = Math.min(window.innerWidth, window.innerHeight) < 700;
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.small ? 1.7 : 2));
        this.renderer.setClearColor(FOG);
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        host.appendChild(this.renderer.domElement);
        this.scene.fog = new THREE.Fog(FOG, 20, 50);
        this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 120);
        this.scene.add(new THREE.AmbientLight(0x303540, 0.3));
        this.spot = new THREE.SpotLight(0xfff1dc, 260, LIGHT_Y + 8, Math.atan(LIGHT_R / LIGHT_Y) * 1.15, 0.6, 0.6);
        this.scene.add(this.spot, this.spot.target);
        const pink = new THREE.PointLight(0xff9ec4, 40, 18, 1.2);
        pink.position.set(TREE.x, 8.5, TREE.z);
        this.scene.add(pink);
        this.scene.add(this.ground(), this.water(), this.grass(this.small ? 9000 : 16000), this.rain(this.small ? 900 : 1500));
        this.cone = this.beam();
        this.scene.add(this.cone);
        this.tree();
        pink.position.copy(this.u.uTree.value); // the pink light sits in the great tree's crown
        this.lilyPads();
        this.koi();
        this.petalSystem();
        this.ro = new ResizeObserver(() => this.resize());
        this.ro.observe(host);
        this.resize();
        const el = this.renderer.domElement;
        el.addEventListener('pointerdown', this.onDown);
        el.addEventListener('pointermove', this.onMove);
        window.addEventListener('pointerup', this.onUp);
        this.raf = requestAnimationFrame(this.frame);
    }
    dispose() {
        cancelAnimationFrame(this.raf);
        this.ro.disconnect();
        window.removeEventListener('pointerup', this.onUp);
        this.scene.traverse((o) => {
            const m = o;
            m.geometry?.dispose();
            const mat = m.material;
            if (Array.isArray(mat))
                mat.forEach((x) => x.dispose());
            else
                mat?.dispose();
        });
        this.renderer.dispose();
        this.renderer.domElement.remove();
    }
    setLight(on) { this.on = on; }
    setWind(on) { this.windOn = on; }
    select(id) { this.picked = id; }
    /**
     * One petal per memory, lying on the pond like the fallen ones: placed by its id across the water, a little
     * bigger the more it matters, coloured by its kind, brightening under the beam.
     */
    setMemories(list) {
        if (this.memMesh) {
            this.scene.remove(this.memMesh);
            this.memMesh.geometry.dispose();
            this.memMesh.material.dispose();
        }
        if (this.memGlow) {
            this.scene.remove(this.memGlow);
            this.memGlow.geometry.dispose();
            this.memGlow.material.dispose();
        }
        const old = new Map(this.mems.map((m) => [m.id, m.vis]));
        const tint = { dream: 0xdcc8ff, vow: 0xff8fb4, commitment: 0xffc49e, lexicon: 0xffeef3, ritual: 0xffb8cf };
        const byTime = [...list].sort((a, b) => a.at - b.at);
        const rank = new Map(byTime.map((m, i) => [m.id, byTime.length > 1 ? i / (byTime.length - 1) : 0.5]));
        const h = (id, salt) => { let x = 2166136261 ^ salt; for (let i = 0; i < id.length; i += 1) {
            x ^= id.charCodeAt(i);
            x = Math.imul(x, 16777619);
        } return (x >>> 0) / 4294967296; };
        this.mems = list.map((m) => {
            const a = h(m.id, 1) * Math.PI * 2;
            const rad = Math.sqrt(h(m.id, 2)) * (pondR(a) - 0.9);
            void rank;
            const y = 0.016;
            const home = new THREE.Vector3(Math.cos(a) * rad, y, Math.sin(a) * rad);
            return { id: m.id, home, phase: h(m.id, 4) * Math.PI * 2, size: 1.05 + m.importance * 0.55, recalls: m.recalls, vis: old.get(m.id) ?? 0, at: home.clone(), tint: new THREE.Color(tint[m.kind] ?? 0xffc2d6) };
        });
        const n = Math.max(1, this.mems.length);
        const geo = new THREE.CircleGeometry(1, 12);
        geo.scale(0.075, 0.05, 1);
        const mesh = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), n);
        this.mems.forEach((m, i) => mesh.setColorAt(i, m.tint));
        mesh.frustumCulled = false;
        mesh.renderOrder = 5;
        this.memMesh = mesh;
        this.scene.add(mesh);
        const gpos = new Float32Array(n * 3), galpha = new Float32Array(n);
        const ggeo = new THREE.BufferGeometry();
        ggeo.setAttribute('position', new THREE.BufferAttribute(gpos, 3));
        ggeo.setAttribute('aAlpha', new THREE.BufferAttribute(galpha, 1));
        const gmat = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            uniforms: { uScale: this.blossomU.uScale },
            vertexShader: /* glsl */ `
        uniform float uScale;
        attribute float aAlpha;
        varying float vA;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = 0.55 * uScale / -mv.z;
          vA = aAlpha;
          gl_Position = projectionMatrix * mv;
        }`,
            fragmentShader: /* glsl */ `
        varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          gl_FragColor = vec4(1.0, 0.72, 0.84, smoothstep(0.5, 0.0, d) * vA * 0.32);
        }`,
        });
        const glow = new THREE.Points(ggeo, gmat);
        glow.frustumCulled = false;
        this.memGlow = glow;
        this.scene.add(glow);
    }
    stepMemories(t, dt) {
        const mesh = this.memMesh, glow = this.memGlow;
        if (!mesh || !glow)
            return;
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3(), col = new THREE.Color();
        const gpos = glow.geometry.getAttribute('position');
        const galpha = glow.geometry.getAttribute('aAlpha');
        const lx = this.pos.x, lz = this.pos.y, on = this.u.uOn.value;
        this.mems.forEach((mem, i) => {
            const d = Math.hypot(mem.home.x - lx, mem.home.z - lz);
            const lit = smooth(LIGHT_R * 1.05, LIGHT_R * 0.35, d) * on;
            mem.vis += (lit - mem.vis) * Math.min(1, dt * 2.2);
            const k = mem.phase;
            mem.at.set(mem.home.x + Math.sin(t * 0.12 + k) * 0.35, 0.016 + Math.sin(t * 1.4 + k) * 0.004, mem.home.z + Math.cos(t * 0.1 + k) * 0.35);
            e.set(-Math.PI / 2, 0, t * 0.06 + k);
            q.setFromEuler(e);
            const chosen = this.picked === mem.id;
            const pulse = mem.recalls > 0 ? 1 + 0.12 * Math.sin(t * 3 + k) : 1;
            s.setScalar(mem.size * pulse * (chosen ? 1.5 : 1));
            col.copy(mem.tint).multiplyScalar(0.3 + 0.75 * mem.vis + (chosen ? 0.2 : 0));
            mesh.setColorAt(i, col);
            m.compose(mem.at, q, s);
            mesh.setMatrixAt(i, m);
            gpos.setXYZ(i, mem.at.x, mem.at.y, mem.at.z);
            galpha.setX(i, mem.vis * (chosen ? 0.9 : mem.recalls > 0 ? 0.35 + 0.2 * Math.sin(t * 3 + k) : 0.22));
        });
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor)
            mesh.instanceColor.needsUpdate = true;
        gpos.needsUpdate = true;
        galpha.needsUpdate = true;
    }
    /** The memory petal nearest a tap, if one is showing under the beam close enough. */
    memoryAt(e) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        const v = new THREE.Vector3();
        let best = null, bestD = 26;
        for (const mem of this.mems) {
            if (mem.vis < 0.3)
                continue;
            v.copy(mem.at).project(this.camera);
            const px = (v.x * 0.5 + 0.5) * rect.width, py = (-v.y * 0.5 + 0.5) * rect.height;
            const d = Math.hypot(px - x, py - y);
            if (d < bestD) {
                bestD = d;
                best = mem.id;
            }
        }
        return best;
    }
    // ------------------------------------------------------------ the land and the water
    ground() {
        const geo = new THREE.PlaneGeometry(44, 44, 176, 176);
        geo.rotateX(-Math.PI / 2);
        const pos = geo.getAttribute('position');
        for (let i = 0; i < pos.count; i += 1)
            pos.setY(i, groundY(pos.getX(i), pos.getZ(i)));
        geo.computeVertexNormals();
        const mat = new THREE.ShaderMaterial({
            uniforms: this.u,
            vertexShader: /* glsl */ `
        varying vec3 vWorld; varying vec3 vNormal; varying float vDepth;
        void main() {
          vec4 w = modelMatrix * vec4(position, 1.0);
          vWorld = w.xyz; vNormal = normal;
          vec4 mv = viewMatrix * w; vDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
            fragmentShader: /* glsl */ `
        ${GLSL_COMMON}
        varying vec3 vWorld; varying vec3 vNormal; varying float vDepth;
        void main() {
          vec2 p = vWorld.xz;
          float lit = litAt(p);
          float glow = treeGlow(p);
          float shade = 0.55 + 0.45 * clamp(vNormal.y, 0.0, 1.0);
          vec3 col;
          if (vWorld.y < -0.005) {
            // the bed: sand and pebbles, bluer the deeper it lies
            vec2 cell = floor(p * 3.2);
            float pebble = hash(cell);
            vec2 f = fract(p * 3.2) - 0.5;
            float stone = smoothstep(0.42, 0.3, length(f + (vec2(hash(cell + 3.1), hash(cell + 7.7)) - 0.5) * 0.3)) * step(0.55, pebble);
            vec3 sand = mix(vec3(0.62, 0.58, 0.49), vec3(0.42, 0.40, 0.37), hash(floor(p * 14.0)) * 0.6);
            col = mix(sand, vec3(0.30, 0.31, 0.33) * (0.7 + 0.6 * pebble), stone);
            float deep = 1.0 - exp(-(-vWorld.y) * 1.25);
            col = mix(col, vec3(0.03, 0.12, 0.26), deep * 0.85);
            // light dancing on the bed under the beam
            float c1 = sin(p.x * 3.1 + uTime * 1.3) + sin(p.y * 2.7 - uTime * 1.1) + sin((p.x + p.y) * 2.2 + uTime * 0.9);
            float caustic = pow(clamp(1.0 - abs(c1) * 0.55, 0.0, 1.0), 5.0);
            col = col * (0.035 + 1.05 * lit + 0.35 * glow) + vec3(0.55, 0.75, 0.95) * caustic * lit * 0.35;
          } else {
            // the banks: dark earth and moss
            float n = hash(floor(p * 6.0));
            col = mix(vec3(0.035, 0.045, 0.035), vec3(0.06, 0.08, 0.05), n) * shade;
            col = col * (0.6 + 9.0 * lit) + vec3(0.5, 0.2, 0.32) * glow * 0.12;
          }
          gl_FragColor = vec4(fogged(col, vDepth), 1.0);
        }`,
        });
        return new THREE.Mesh(geo, mat);
    }
    /** The surface: deep blue and see-through, a sheen under the beam, a pink blush near the tree, and the rings. */
    water() {
        const mat = new THREE.ShaderMaterial({
            uniforms: this.u,
            transparent: true,
            depthWrite: false,
            vertexShader: /* glsl */ `
        varying vec3 vWorld; varying float vDepth;
        void main() {
          vec4 w = modelMatrix * vec4(position, 1.0);
          vWorld = w.xyz;
          vec4 mv = viewMatrix * w; vDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
            fragmentShader: /* glsl */ `
        ${GLSL_COMMON}
        uniform vec4 uDrops[${RIPPLES}];
        varying vec3 vWorld; varying float vDepth;
        float octa(vec2 d, float seed) {
          float a = atan(d.y, d.x) + seed * 6.2831;
          float k = 3.14159265 / 8.0;
          float sector = mod(a, 2.0 * k) - k;
          return length(d) * cos(k) / cos(sector) * (1.0 + 0.04 * sin(a * 3.0 + seed * 20.0));
        }
        float ring(float r, float at, float width) { return smoothstep(width, 0.0, abs(r - at)); }
        void main() {
          vec2 p = vWorld.xz;
          float d = pondSD(p);
          if (d > 0.02) discard;
          float lit = litAt(p);
          float glow = treeGlow(p);
          float lines = 0.0;
          for (int i = 0; i < ${RIPPLES}; i++) {
            vec4 drop = uDrops[i];
            float age = uTime - drop.z;
            if (age < 0.0 || age > 2.4) continue;
            vec2 v = p - drop.xy;
            if (dot(v, v) > 2.6) continue;
            float r = octa(v, drop.w);
            float fade = (1.0 - age / 2.4) * min(1.0, drop.w);
            float outer = ring(r, 0.05 + age * 0.55 * drop.w, 0.03) * fade;
            float inner = ring(r, max(0.0, age - 0.35) * 0.4 * drop.w, 0.026) * fade * step(0.35, age);
            float dot0 = ring(r, 0.03, 0.03) * smoothstep(0.25, 0.0, age);
            lines = max(lines, max(max(outer, inner), dot0));
          }
          vec3 toEye = normalize(cameraPosition - vWorld);
          float fres = pow(1.0 - clamp(toEye.y, 0.0, 1.0), 3.0);
          vec3 col = vec3(0.02, 0.09, 0.2) + vec3(0.22, 0.24, 0.24) * lit * 0.35 + vec3(0.6, 0.25, 0.4) * glow * 0.18;
          float alpha = 0.28 + 0.45 * fres;
          float seen = 0.04 + 0.96 * lit + 0.25 * glow;
          col = mix(col, vec3(0.97, 0.95, 0.9), clamp(lines * seen, 0.0, 1.0));
          alpha = max(alpha, lines * seen * 0.95);
          alpha *= smoothstep(0.02, -0.12, d);
          gl_FragColor = vec4(fogged(col, vDepth), alpha);
        }`,
        });
        const m = new THREE.Mesh(new THREE.PlaneGeometry(13, 13), mat);
        m.rotation.x = -Math.PI / 2;
        m.renderOrder = 2;
        return m;
    }
    grass(count) {
        const blade = new THREE.PlaneGeometry(1, 1, 1, 4);
        blade.translate(0, 0.5, 0);
        const geo = new THREE.InstancedBufferGeometry();
        geo.index = blade.index;
        geo.setAttribute('position', blade.getAttribute('position'));
        const offset = new Float32Array(count * 3), scale = new Float32Array(count * 2), angle = new Float32Array(count), rand = new Float32Array(count);
        const r = stream(7);
        for (let i = 0; i < count; i += 1) {
            let x = 0, z = 0;
            do {
                x = (r() - 0.5) * 32;
                z = (r() - 0.5) * 32;
            } while (pondSD(x, z) < 0.35);
            offset.set([x, groundY(x, z), z], i * 3);
            scale.set([0.05 + r() * 0.06, 0.25 + Math.pow(r(), 1.6) * 0.6], i * 2);
            angle[i] = r() * Math.PI * 2;
            rand[i] = r();
        }
        geo.setAttribute('aOffset', new THREE.InstancedBufferAttribute(offset, 3));
        geo.setAttribute('aScale', new THREE.InstancedBufferAttribute(scale, 2));
        geo.setAttribute('aAngle', new THREE.InstancedBufferAttribute(angle, 1));
        geo.setAttribute('aRand', new THREE.InstancedBufferAttribute(rand, 1));
        geo.instanceCount = count;
        const mat = new THREE.ShaderMaterial({
            uniforms: this.u,
            side: THREE.DoubleSide,
            vertexShader: /* glsl */ `
        ${GLSL_COMMON}
        attribute vec3 aOffset; attribute vec2 aScale; attribute float aAngle; attribute float aRand;
        varying float vH; varying float vLit; varying float vGlow; varying float vRand; varying float vDepth;
        void main() {
          float h = position.y;
          vec3 p = vec3(position.x * aScale.x * (1.0 - h * 0.88), h * aScale.y, 0.0);
          float c = cos(aAngle), s = sin(aAngle);
          p = vec3(p.x * c, p.y, p.x * s);
          vec3 w = p + aOffset;
          float swell = sin(uTime * 1.1 - aOffset.x * 0.22 - aOffset.z * 0.12);
          float bend = h * h * aScale.y;
          float gust = 0.1 + 0.9 * uWind;
          w.x += (0.25 + swell * 0.3 + sin(uTime * 3.0 + aRand * 12.0) * 0.12) * bend * gust;
          w.z += (0.12 + swell * 0.14) * bend * gust;
          vLit = litAt(aOffset.xz); vGlow = treeGlow(aOffset.xz);
          vH = h; vRand = aRand;
          vec4 mv = viewMatrix * vec4(w, 1.0);
          vDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
            fragmentShader: /* glsl */ `
        ${GLSL_COMMON}
        varying float vH; varying float vLit; varying float vGlow; varying float vRand; varying float vDepth;
        void main() {
          vec3 dark = mix(vec3(0.012, 0.022, 0.02), vec3(0.05, 0.085, 0.065), vH * vH) * (0.6 + 0.8 * vRand);
          vec3 lit = mix(vec3(0.10, 0.16, 0.06), vec3(0.78, 0.74, 0.55), vH * vH) * (0.75 + 0.5 * vRand);
          vec3 col = mix(dark, lit, vLit) + vec3(0.55, 0.2, 0.35) * vGlow * vH * 0.35;
          gl_FragColor = vec4(fogged(col, vDepth), 1.0);
        }`,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.frustumCulled = false;
        return mesh;
    }
    rain(count) {
        const pos = new Float32Array(count * 2 * 3);
        const end = new Float32Array(count * 2);
        const r = stream(13);
        for (let i = 0; i < count; i += 1) {
            const x = (r() - 0.5) * 30, y = r() * 14, z = (r() - 0.5) * 30;
            for (let k = 0; k < 2; k += 1) {
                pos.set([x, y, z], (i * 2 + k) * 3);
                end[i * 2 + k] = k;
            }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('aEnd', new THREE.BufferAttribute(end, 1));
        const mat = new THREE.ShaderMaterial({
            uniforms: this.u,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            vertexShader: /* glsl */ `
        ${GLSL_COMMON}
        attribute float aEnd;
        varying float vAlpha;
        void main() {
          vec3 p = position;
          float y = mod(p.y - uTime * 9.0, 14.0);
          p.y = y + aEnd * 0.5;
          float lit = smoothstep(uRadius * 1.4, uRadius * 0.3, distance(p.xz, uLight.xz)) * uOn;
          vAlpha = (0.05 + 0.9 * lit + 0.15 * treeGlow(p.xz)) * (0.2 + 0.8 * aEnd) * smoothstep(0.0, 0.6, y);
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
            fragmentShader: /* glsl */ `
        varying float vAlpha;
        void main() { gl_FragColor = vec4(0.85, 0.84, 0.82, 0.45 * vAlpha); }`,
        });
        const lines = new THREE.LineSegments(geo, mat);
        lines.frustumCulled = false;
        return lines;
    }
    beam() {
        const geo = new THREE.CylinderGeometry(0.6, LIGHT_R, LIGHT_Y, 48, 1, true);
        geo.translate(0, LIGHT_Y / 2, 0);
        const mat = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: false,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending,
            uniforms: { uOn: this.u.uOn },
            vertexShader: /* glsl */ `
        varying float vY; varying float vFacing;
        void main() {
          vY = uv.y;
          vec3 n = normalize(normalMatrix * normal);
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vFacing = abs(dot(n, normalize(-mv.xyz)));
          gl_Position = projectionMatrix * mv;
        }`,
            fragmentShader: /* glsl */ `
        uniform float uOn;
        varying float vY; varying float vFacing;
        void main() {
          float a = pow(vFacing, 2.4) * smoothstep(0.0, 0.08, vY) * (1.0 - 0.55 * vY) * uOn;
          gl_FragColor = vec4(1.0, 0.95, 0.86, a * 0.1);
        }`,
        });
        return new THREE.Mesh(geo, mat);
    }
    // ------------------------------------------------------------ the cherry tree
    /** Both cherry trees: the great one by the pond, a smaller one across the water. */
    tree() {
        const big = this.small ? 85000 : 150000;
        this.cherry(TREE, 1.75, 1234, big, true);
        this.cherry(new THREE.Vector3(5.6, 0, -9.6), 0.8, 77, Math.round(big * 0.42), false);
    }
    /**
     * A cherry tree in full bloom: a trunk dividing into upright limbs, and over them a great rounded crown — dozens of
     * fluffy clouds of blossom heaped into a dome, pale where the light falls, mauve in the shade, fine bare twigs
     * poking out at the edge and strings of flowers hanging from its rim. Self-lit (it is night), with a soft halo.
     */
    cherry(base, scale, seed, budget, main) {
        const r = stream(seed);
        const parts = [];
        const ends = [];
        const up = new THREE.Vector3(0, 1, 0);
        const limb = (a, b, ra, rb) => {
            const len = a.distanceTo(b);
            const g = new THREE.CylinderGeometry(rb, ra, len, 8, 1);
            g.translate(0, len / 2, 0);
            g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up, b.clone().sub(a).normalize()));
            g.translate(a.x, a.y, a.z);
            parts.push(g);
        };
        const grow = (start, dir, len, rad, depth) => {
            const kink = dir.clone().add(new THREE.Vector3((r() - 0.5) * 0.3, 0, (r() - 0.5) * 0.3)).normalize();
            const mid = start.clone().addScaledVector(dir, len * 0.5);
            const end = mid.clone().addScaledVector(kink, len * 0.5);
            limb(start, mid, rad, rad * 0.86);
            limb(mid, end, rad * 0.86, rad * 0.72);
            if (depth === 0) {
                ends.push(end);
                return;
            }
            const kids = depth === 4 ? 3 : 2 + (r() < 0.5 ? 1 : 0);
            for (let k = 0; k < kids; k += 1) {
                const yaw = (k / kids) * Math.PI * 2 + r() * 1.2;
                const spread = depth === 4 ? 0.42 + r() * 0.2 : 0.4 + r() * 0.35; // a vase: limbs rise and open out
                const d = new THREE.Vector3(Math.cos(yaw) * Math.sin(spread), Math.cos(spread), Math.sin(yaw) * Math.sin(spread));
                d.lerp(kink, 0.3).normalize();
                grow(end, d, len * (0.76 + r() * 0.1), rad * 0.62, depth - 1);
            }
        };
        const root = base.clone().setY(groundY(base.x, base.z) - 0.25);
        grow(root, new THREE.Vector3(0.04, 1, 0.03).normalize(), 2.6 * scale, 0.6 * scale, 4);
        const wood = new THREE.Mesh(mergeGeometries(parts), new THREE.MeshStandardMaterial({ color: 0x2a201f, roughness: 0.95, emissive: 0x1c0d0e, emissiveIntensity: 0.5 }));
        this.scene.add(wood);
        // the crown: an egg-shaped dome over the branch tips, heaped with clouds of blossom
        const tips = ends.reduce((s, e) => s.add(e), new THREE.Vector3()).multiplyScalar(1 / ends.length);
        const rx = 5.6 * scale, ry = 3.4 * scale; // wide rather than tall: the crown spreads out sideways
        const center = new THREE.Vector3(root.x + (tips.x - root.x) * 0.5, tips.y + ry * 0.15, root.z + (tips.z - root.z) * 0.5);
        const clouds = [];
        const shell = Math.round(190 * scale * scale);
        for (let k = 0; k < shell; k += 1) {
            const d = new THREE.Vector3(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
            if (d.y < -0.55)
                d.y = -0.55 + r() * 0.2; // the underside stays open, round the trunk
            d.normalize();
            const reach = 0.7 + 0.3 * Math.pow(r(), 0.5);
            const c = center.clone().add(new THREE.Vector3(d.x * rx, d.y * ry, d.z * rx).multiplyScalar(reach));
            clouds.push({ c, rad: (0.75 + r() * 0.55) * scale });
        }
        for (const e of ends)
            clouds.push({ c: e.clone().add(new THREE.Vector3(0, 0.2 * scale, 0)), rad: (0.6 + r() * 0.4) * scale });
        const rim = Math.round(46 * scale);
        for (let k = 0; k < rim; k += 1) {
            const a = (k / rim) * Math.PI * 2 + r() * 0.12;
            const c = center.clone().add(new THREE.Vector3(Math.cos(a) * rx * (1.0 + r() * 0.12), -ry * (0.38 + r() * 0.12), Math.sin(a) * rx * (1.0 + r() * 0.12)));
            clouds.push({ c, rad: (0.7 + r() * 0.4) * scale });
        }
        const strands = [];
        for (const { c } of clouds) {
            if (c.y > center.y - ry * 0.18)
                continue;
            const out = new THREE.Vector3(c.x - center.x, 0, c.z - center.z).normalize();
            const count = r() < 0.4 ? 1 : 0;
            for (let j = 0; j < count; j += 1) {
                const start = c.clone().add(new THREE.Vector3((r() - 0.5) * 0.6, -0.1, (r() - 0.5) * 0.6).multiplyScalar(scale));
                const s = [start];
                const len = (0.45 + r() * 0.8) * scale;
                const lean = (0.35 + r() * 0.5) * scale;
                for (let k = 1; k <= 9; k += 1) {
                    const f = k / 9;
                    s.push(start.clone().addScaledVector(out, lean * Math.sqrt(f)).add(new THREE.Vector3(0, -len * f, 0)));
                }
                strands.push(s);
            }
        }
        const per = Math.max(30, Math.floor((budget * 0.9) / clouds.length));
        const strandPer = 16;
        const n = clouds.length * per + strands.length * 9 * strandPer;
        const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), size = new Float32Array(n), phase = new Float32Array(n);
        const sun = new THREE.Vector3(0.35, 1, 0.55).normalize();
        const petals = [[1, 0.78, 0.86], [1, 0.85, 0.9], [0.98, 0.7, 0.81], [1, 0.92, 0.95], [0.96, 0.66, 0.79]];
        const shade = [0.46, 0.27, 0.4];
        const v = new THREE.Vector3(), gN = new THREE.Vector3();
        let i = 0;
        const put = (p, light, sz) => {
            pos.set([p.x, p.y, p.z], i * 3);
            const c = petals[Math.floor(r() * petals.length)];
            col.set([shade[0] + (c[0] - shade[0]) * light, shade[1] + (c[1] - shade[1]) * light, shade[2] + (c[2] - shade[2]) * light], i * 3);
            size[i] = sz;
            phase[i] = r() * Math.PI * 2;
            i += 1;
        };
        for (const { c, rad } of clouds) {
            for (let k = 0; k < per; k += 1) {
                v.set(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
                const depthIn = Math.pow(r(), 0.4);
                const p = c.clone().add(v.clone().multiply(new THREE.Vector3(rad, rad * 0.8, rad)).multiplyScalar(depthIn));
                gN.copy(p).sub(center).divide(new THREE.Vector3(rx, ry, rx));
                const outer = Math.min(1, gN.length());
                gN.normalize();
                const lambert = (Math.max(0, gN.dot(sun)) * 0.55 + Math.max(0, v.dot(sun)) * 0.45) * 0.8 + 0.2;
                put(p, Math.min(1, lambert * (0.35 + 0.65 * depthIn) * (0.55 + 0.45 * outer) + 0.08), (0.075 + r() * 0.075) * scale);
            }
            this.blossomPts.push(c);
        }
        for (const s of strands)
            for (let k = 1; k < s.length; k += 1)
                for (let m = 0; m < strandPer; m += 1) {
                    const taper = 1 - (k / s.length) * 0.5;
                    const p = s[k].clone().add(new THREE.Vector3((r() - 0.5) * 0.34 * taper, (r() - 0.5) * 0.2, (r() - 0.5) * 0.34 * taper).multiplyScalar(scale));
                    // under the crown: mostly in its shade, like the clouds' undersides
                    put(p, 0.22 + r() * 0.38 + 0.12 * (1 - k / s.length), (0.07 + r() * 0.06) * scale);
                }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, i * 3), 3));
        geo.setAttribute('aColor', new THREE.BufferAttribute(col.subarray(0, i * 3), 3));
        geo.setAttribute('aSize', new THREE.BufferAttribute(size.subarray(0, i), 1));
        geo.setAttribute('aPhase', new THREE.BufferAttribute(phase.subarray(0, i), 1));
        const mat = new THREE.ShaderMaterial({
            uniforms: { ...this.blossomU, uFogColor: this.u.uFogColor, uFogNear: this.u.uFogNear, uFogFar: this.u.uFogFar },
            vertexShader: /* glsl */ `
        uniform float uTime; uniform float uScale; uniform float uWind;
        attribute vec3 aColor; attribute float aSize; attribute float aPhase;
        varying vec3 vColor; varying float vDepth;
        void main() {
          vec3 p = position;
          float sway = max(0.0, p.y - 2.0) * 0.014 * (0.15 + 0.85 * uWind);
          p.x += sin(uTime * 0.9 + p.y * 0.4 + p.z * 0.2) * sway + sin(uTime * 2.3 + aPhase) * 0.012 * uWind;
          p.z += sin(uTime * 0.7 + p.x * 0.3) * sway * 0.6;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = max(1.5, aSize * uScale / -mv.z);
          vColor = aColor * (0.95 + 0.05 * sin(uTime * 1.3 + aPhase));
          vDepth = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
            fragmentShader: /* glsl */ `
        uniform vec3 uFogColor; uniform float uFogNear; uniform float uFogFar;
        varying vec3 vColor; varying float vDepth;
        void main() {
          vec2 q = gl_PointCoord - 0.5;
          float a = atan(q.y, q.x);
          if (length(q) > 0.42 + 0.06 * cos(a * 5.0)) discard;     // a soft five-lobed flower
          float centre = smoothstep(0.18, 0.0, length(q));
          gl_FragColor = vec4(mix(mix(vColor, vColor * vec3(1.0, 0.86, 0.9), centre), uFogColor, smoothstep(uFogNear, uFogFar, vDepth)), 1.0);
        }`,
        });
        const blossom = new THREE.Points(geo, mat);
        blossom.frustumCulled = false;
        this.scene.add(blossom);
        // fine bare twigs poking out of the crown, and the threads the hanging flowers hang from
        const twig = [];
        for (let k = 0; k < Math.round(70 * scale); k += 1) {
            const c = clouds[Math.floor(r() * clouds.length)].c;
            const out = c.clone().sub(center).normalize();
            const len = (0.6 + r() * 0.9) * scale;
            const tip = c.clone().addScaledVector(out, len).add(new THREE.Vector3((r() - 0.5) * 0.4, (r() - 0.2) * 0.4, (r() - 0.5) * 0.4));
            twig.push(c.x, c.y, c.z, tip.x, tip.y, tip.z);
        }
        for (const s of strands)
            for (let k = 1; k < s.length; k += 1)
                twig.push(s[k - 1].x, s[k - 1].y, s[k - 1].z, s[k].x, s[k].y, s[k].z);
        const twigGeo = new THREE.BufferGeometry();
        twigGeo.setAttribute('position', new THREE.Float32BufferAttribute(twig, 3));
        this.scene.add(new THREE.LineSegments(twigGeo, new THREE.LineBasicMaterial({ color: 0x4a3434, transparent: true, opacity: 0.75 })));
        // a soft halo behind the crown, so the tree glows in the night
        const cv = document.createElement('canvas');
        cv.width = cv.height = 128;
        const g = cv.getContext('2d');
        const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
        grad.addColorStop(0, 'rgba(255,185,212,0.3)');
        grad.addColorStop(0.55, 'rgba(255,160,200,0.09)');
        grad.addColorStop(1, 'rgba(255,160,200,0)');
        g.fillStyle = grad;
        g.fillRect(0, 0, 128, 128);
        const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
        halo.position.copy(center);
        halo.scale.setScalar(rx * 3.6);
        halo.renderOrder = -1;
        this.scene.add(halo);
        if (main)
            this.u.uTree.value.copy(center);
    }
    // ------------------------------------------------------------ life in the pond
    lilyPads() {
        const r = stream(21);
        const mat = new THREE.MeshStandardMaterial({ color: 0x6f9c7e, roughness: 0.7, side: THREE.DoubleSide });
        for (const [x, z, s] of [[-2.4, 1.6, 0.85], [2.6, -1.2, 0.65], [0.6, 3.0, 0.5], [-0.8, -2.8, 0.45], [3.0, 1.9, 0.35]]) {
            const shape = new THREE.Shape();
            const notch = r() * Math.PI * 2;
            for (let i = 0; i <= 22; i += 1) {
                const a = notch + 0.35 + (i / 22) * (Math.PI * 2 - 0.7);
                const rad = s * (0.93 + r() * 0.1);
                if (i === 0)
                    shape.moveTo(0, 0);
                shape.lineTo(Math.cos(a) * rad, Math.sin(a) * rad);
            }
            shape.lineTo(0, 0);
            const pad = new THREE.Mesh(new THREE.ShapeGeometry(shape), mat);
            pad.rotation.x = -Math.PI / 2;
            pad.position.set(x, 0.015, z);
            pad.userData.phase = r() * Math.PI * 2;
            pad.renderOrder = 3;
            this.pads.push(pad);
            this.scene.add(pad);
        }
    }
    koi() {
        const r = stream(9);
        const colors = [0xe0662a, 0xf08a3a, 0xf3efe6, 0xd9542a, 0xf6b26b];
        for (let i = 0; i < 5; i += 1) {
            const mat = new THREE.MeshStandardMaterial({ color: colors[i], roughness: 0.45, emissive: 0x1a0904 });
            const g = new THREE.Group();
            const body = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 8), mat);
            body.scale.set(2.8, 0.75, 1);
            g.add(body);
            const tail = new THREE.Mesh(new THREE.ConeGeometry(0.14, 0.28, 4), mat);
            tail.rotation.z = Math.PI / 2;
            tail.scale.set(1, 1, 0.25);
            tail.position.x = -0.4;
            g.add(tail);
            this.scene.add(g);
            this.fish.push({ mesh: g, r: 1.3 + r() * 1.9, speed: (0.14 + r() * 0.16) * (r() < 0.5 ? 1 : -1), phase: r() * Math.PI * 2, cx: (r() - 0.5) * 1.4, cz: (r() - 0.5) * 1.4, y: -0.3 - r() * 0.45 });
        }
    }
    petalSystem() {
        const geo = new THREE.CircleGeometry(1, 7);
        geo.scale(0.075, 0.05, 1);
        const mesh = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), PETALS);
        const color = new THREE.Color();
        const palette = [0xffc6d9, 0xffd9e6, 0xff9fc0, 0xffe6ef];
        for (let i = 0; i < PETALS; i += 1) {
            mesh.setColorAt(i, color.set(palette[i % palette.length]));
            this.petals.push({ p: new THREE.Vector3(), v: new THREE.Vector3(), rot: new THREE.Euler(), spin: new THREE.Vector3(), state: 3, life: this.r() * 14, size: 0.8 + this.r() * 0.5 });
        }
        mesh.frustumCulled = false;
        mesh.renderOrder = 4;
        this.petalMesh = mesh;
        this.scene.add(mesh);
    }
    launch(pt) {
        const r = this.r;
        const c = this.blossomPts[Math.floor(r() * this.blossomPts.length)];
        pt.p.set(c.x + (r() - 0.5) * 2, c.y + (r() - 0.5) * 1, c.z + (r() - 0.5) * 2);
        pt.v.set(0, 0, 0);
        pt.rot.set(r() * 6, r() * 6, r() * 6);
        pt.spin.set((r() - 0.5) * 5, (r() - 0.5) * 5, (r() - 0.5) * 5);
        pt.state = 0;
        pt.life = 0;
    }
    addDrop(x, z, t, strength) {
        this.u.uDrops.value[this.dropIndex].set(x, z, t, strength);
        this.dropIndex = (this.dropIndex + 1) % RIPPLES;
    }
    stepPetals(dt, t) {
        const mesh = this.petalMesh;
        if (!mesh)
            return;
        const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3();
        // the wind turns slowly round, gusting; off, petals only drift down
        const w = this.u.uWind.value;
        const heading = 0.6 + Math.sin(t * 0.045) * 1.6 + Math.sin(t * 0.13) * 0.5;
        const gust = (0.6 + 0.4 * Math.sin(t * 0.31) + 0.25 * Math.sin(t * 1.3)) * 1.4 * w;
        const wind = new THREE.Vector3(Math.cos(heading), 0, Math.sin(heading)).multiplyScalar(gust);
        this.petals.forEach((pt, i) => {
            let scale = pt.size;
            if (pt.state === 3) { // waiting to fall
                pt.life -= dt;
                scale = 0;
                if (pt.life <= 0) {
                    // with the wind down, only a petal now and then lets go
                    if (w > 0.3 || this.r() < 0.12)
                        this.launch(pt);
                    else
                        pt.life = 1 + this.r() * 3;
                }
            }
            else if (pt.state === 0) { // falling, tumbling on the wind
                pt.v.x += (wind.x + Math.sin(t * 2 + i) * (0.15 + 0.5 * w) - pt.v.x) * dt * 1.5;
                pt.v.z += (wind.z + Math.cos(t * 1.7 + i) * (0.15 + 0.5 * w) - pt.v.z) * dt * 1.5;
                pt.v.y = -0.55 + Math.sin(t * 3 + i) * 0.2;
                pt.p.addScaledVector(pt.v, dt);
                pt.rot.x += pt.spin.x * dt;
                pt.rot.y += pt.spin.y * dt;
                pt.rot.z += pt.spin.z * dt;
                const inPond = pondSD(pt.p.x, pt.p.z) < -0.15;
                const floor = inPond ? 0.012 : groundY(pt.p.x, pt.p.z) + 0.02;
                if (pt.p.y <= floor) {
                    pt.p.y = floor;
                    pt.state = inPond ? 1 : 2;
                    pt.life = inPond ? 14 + this.r() * 12 : 12 + this.r() * 14;
                    pt.rot.set(-Math.PI / 2, 0, this.r() * 6);
                    if (inPond)
                        this.addDrop(pt.p.x, pt.p.z, t, 0.45);
                }
                if (Math.abs(pt.p.x) > 21 || Math.abs(pt.p.z) > 21) {
                    pt.state = 3;
                    pt.life = this.r() * 4;
                }
            }
            else { // floating on the pond, or resting on the bank
                pt.life -= dt;
                if (pt.state === 1) {
                    pt.p.x += (0.08 + Math.sin(t * 0.4 + i) * 0.05) * dt;
                    pt.p.z += (0.05 + Math.cos(t * 0.3 + i) * 0.05) * dt;
                    pt.rot.z += 0.08 * dt;
                    if (pondSD(pt.p.x, pt.p.z) > -0.2) {
                        pt.p.x -= 0.08 * dt;
                        pt.p.z -= 0.05 * dt;
                    }
                    pt.p.y = 0.012 + Math.sin(t * 1.4 + i) * 0.004;
                }
                scale = pt.size * Math.min(1, pt.life / 1.5);
                if (pt.life <= 0) {
                    pt.state = 3;
                    pt.life = this.r() * 3;
                }
            }
            q.setFromEuler(pt.rot);
            s.setScalar(Math.max(0.0001, scale));
            m.compose(pt.p, q, s);
            mesh.setMatrixAt(i, m);
        });
        mesh.instanceMatrix.needsUpdate = true;
    }
    // ------------------------------------------------------------ input
    hit(e) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
        this.raycaster.setFromCamera(ndc, this.camera);
        const p = new THREE.Vector3();
        if (!this.raycaster.ray.intersectPlane(this.surface, p))
            return null;
        const v = new THREE.Vector2(p.x, p.z);
        if (v.length() > 13)
            v.setLength(13);
        return v;
    }
    onDown = (e) => { this.down = { x: e.clientX, yaw: this.yawGoal, moved: false }; };
    onMove = (e) => {
        if (!this.down)
            return;
        if (e.buttons === 0) {
            this.down = null;
            return;
        } // a press that ended elsewhere must not turn the pond
        const dx = e.clientX - this.down.x;
        if (Math.abs(dx) > 8)
            this.down.moved = true;
        if (this.down.moved)
            this.yawGoal = this.down.yaw - dx * 0.008;
    };
    onUp = (e) => {
        if (this.down && !this.down.moved) {
            const mem = this.memoryAt(e);
            this.picked = mem;
            this.onPick(mem);
            if (mem) {
                this.down = null;
                return;
            }
            const p = this.hit(e);
            if (p) {
                this.target.copy(p);
                this.onFirstTouch();
            }
        }
        this.down = null;
    };
    // ------------------------------------------------------------ frame
    resize() {
        const r = this.host.getBoundingClientRect();
        const w = Math.max(1, r.width), h = Math.max(1, r.height);
        this.renderer.setSize(w, h, false);
        this.renderer.domElement.style.width = `${w}px`;
        this.renderer.domElement.style.height = `${h}px`;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.blossomU.uScale.value = (h * this.renderer.getPixelRatio() * 0.5) / Math.tan((this.camera.fov * Math.PI) / 360);
    }
    placeCamera() {
        const portrait = this.camera.aspect < 0.75;
        const dist = portrait ? 32 : 21;
        const focus = new THREE.Vector3(portrait ? -4.4 : -2.4, portrait ? 4.8 : 3.2, portrait ? -3.4 : -2.4);
        this.camera.position.set(focus.x + Math.sin(this.yaw) * dist, portrait ? 15 : 9.5, focus.z + Math.cos(this.yaw) * dist);
        this.camera.lookAt(focus);
        this.u.uFogNear.value = dist - 4;
        this.u.uFogFar.value = dist + 30;
        const fog = this.scene.fog;
        fog.near = dist - 4;
        fog.far = dist + 30;
    }
    frame = () => {
        this.raf = requestAnimationFrame(this.frame);
        const dt = Math.min(0.05, this.clock.getDelta());
        const t = this.clock.elapsedTime;
        this.u.uTime.value = t;
        this.pos.lerp(this.target, 1 - Math.pow(0.2, dt));
        this.u.uOn.value += ((this.on ? 1 : 0) - this.u.uOn.value) * Math.min(1, dt * 3);
        this.u.uWind.value += ((this.windOn ? 1 : 0) - this.u.uWind.value) * Math.min(1, dt * 1.2);
        this.u.uLight.value.set(this.pos.x, 0, this.pos.y);
        this.spot.position.set(this.pos.x, LIGHT_Y, this.pos.y);
        this.spot.target.position.set(this.pos.x, 0, this.pos.y);
        this.spot.intensity = 260 * this.u.uOn.value;
        this.cone.position.set(this.pos.x, 0, this.pos.y);
        // rain on the pond
        this.dropDebt += dt * 26;
        const r = this.r;
        while (this.dropDebt >= 1) {
            this.dropDebt -= 1;
            let x = 0, z = 0;
            do {
                x = (r() - 0.5) * 12;
                z = (r() - 0.5) * 12;
            } while (pondSD(x, z) > -0.2);
            this.addDrop(x, z, t + r() * 0.05, 0.7 + r() * 0.6);
        }
        this.stepPetals(dt, t);
        this.stepMemories(t, dt);
        for (const pad of this.pads) {
            const ph = pad.userData.phase;
            pad.position.y = 0.015 + Math.sin(t * 1.3 + ph) * 0.006;
            pad.rotation.z = Math.sin(t * 0.2 + ph) * 0.08;
        }
        for (const f of this.fish) {
            const a = f.phase + t * f.speed;
            f.mesh.position.set(f.cx + Math.cos(a) * f.r, f.y + Math.sin(t * 0.7 + f.phase) * 0.05, f.cz + Math.sin(a) * f.r * 0.75);
            const dx = -Math.sin(a) * Math.sign(f.speed), dz = Math.cos(a) * 0.75 * Math.sign(f.speed);
            f.mesh.rotation.y = Math.atan2(-dz, dx) + Math.sin(t * 6 + f.phase) * 0.15;
        }
        this.yaw += (this.yawGoal - this.yaw) * Math.min(1, dt * 6);
        this.placeCamera();
        this.renderer.render(this.scene, this.camera);
    };
}
