/* =========================================================================
   STRATA · 地质力学真 3D 深度缓冲视口引擎 (Three.js r170 WebGL + Canvas 2D 降级)
   -------------------------------------------------------------------------
   视觉与工程升级：
   1. 透明 WebGL 画布叠加径向工作室渐变背景，配合三点柔光 PBR 材质与地面投影轮廓，
      赋予三维非正交裂隙面真实的半透明矿物石英质感与空间纵深。
   2. 严格保留真实拓扑孔洞：孔洞边界环以青玉立体管线描边，肉眼直见镂空通透效果。
   3. 左右双见证同步分屏：主视口 (连通见证 m+, t+) 与副视口 (断割见证 m-, t-)
      共享同一球面相机，旋转/平移/缩放/微距推进实时零延迟联动。
   4. S/T 最小割断口（如 0.00297 m = 2.97 mm）立体游标卡尺与平滑飞入微距聚焦。
   ========================================================================= */
"use strict";

window.StrataViewer3D = (function () {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpVec3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

  /* 克制、低饱和度的地质矿物色板 */
  const PALETTE = {
    light: {
      bgCss: "#f6f5f0",
      gridMajor: 0xc8c7be,
      gridMinor: 0xe3e2da,
      shadowColor: 0x1c2420,
      shadowOpacity: 0.055,
      nominalAnchorA: 0x167870,
      nominalAnchorB: 0x2e5888,
      nominalInterior: 0x768580,
      nominalTop1: 0xd98324,
      connPathTri: 0x0b8c60,
      connPathFrag: 0x22b583,
      connOther: 0xaab4b0,
      discCutS: 0x1677b3,
      discCutT: 0xc93b32,
      discOther: 0xaab4b0,
      edgeDefault: 0x23302c,
      edgeMuted: 0x9aa39f,
      edgeTop1: 0xb45309,
      edgeHole: 0x0d9488,
      chainTube: 0x059669,
      chainHalo: 0x34d399,
      gapLine: 0xc93b32,
      gapGlow: 0xf59e0b,
    },
    dark: {
      bgCss: "#0e1419",
      gridMajor: 0x263540,
      gridMinor: 0x17222a,
      shadowColor: 0x000000,
      shadowOpacity: 0.24,
      nominalAnchorA: 0x14b8a6,
      nominalAnchorB: 0x3b82f6,
      nominalInterior: 0x5c7180,
      nominalTop1: 0xf59e0b,
      connPathTri: 0x10b981,
      connPathFrag: 0x34d399,
      connOther: 0x3d4d59,
      discCutS: 0x38bdf8,
      discCutT: 0xf87171,
      discOther: 0x3d4d59,
      edgeDefault: 0xb8c8d4,
      edgeMuted: 0x455663,
      edgeTop1: 0xfbbf24,
      edgeHole: 0x2dd4bf,
      chainTube: 0x10b981,
      chainHalo: 0x6ee7b7,
      gapLine: 0xf87171,
      gapGlow: 0xfbbf24,
    },
  };

  function centroid3(pts) {
    if (!pts || !pts.length) return [0, 0, 0];
    let sx = 0, sy = 0, sz = 0;
    for (const p of pts) {
      sx += p[0]; sy += p[1]; sz += p[2];
    }
    const n = pts.length;
    return [sx / n, sy / n, sz / n];
  }

  function normal3(tri) {
    if (!tri || tri.length < 3) return [0, 0, 1];
    const a = tri[0], b = tri[1], c = tri[2];
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    return [nx / len, ny / len, nz / len];
  }

  function computeBounds(pts) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) {
      if (!p) continue;
      for (let d = 0; d < 3; d++) {
        if (p[d] < min[d]) min[d] = p[d];
        if (p[d] > max[d]) max[d] = p[d];
      }
    }
    if (!isFinite(min[0])) {
      min[0] = min[1] = min[2] = -1;
      max[0] = max[1] = max[2] = 1;
    }
    const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    const radius = Math.max(0.25, Math.hypot(span[0], span[1], span[2]) / 2);
    return { min, max, center, span, radius };
  }

  function niceScaleMetric(rawMeters) {
    if (!isFinite(rawMeters) || rawMeters <= 0) return { meters: 1, label: "1 m" };
    const pow = Math.pow(10, Math.floor(Math.log10(rawMeters)));
    let m = 10 * pow;
    for (const step of [1, 2, 5, 10]) {
      if (rawMeters <= step * pow) {
        m = step * pow;
        break;
      }
    }
    if (m < 0.01) {
      return { meters: m, label: (m * 1000).toFixed(m < 0.001 ? 2 : 1).replace(/\.0$/, "") + " mm" };
    }
    if (m < 1) {
      return { meters: m, label: (m * 100).toFixed(m < 0.1 ? 1 : 0).replace(/\.0$/, "") + " cm" };
    }
    return { meters: m, label: m.toFixed(m < 1 ? 2 : 1).replace(/\.0$/, "") + " m" };
  }

  function hexToCss(hexInt, alpha) {
    const r = (hexInt >> 16) & 255;
    const g = (hexInt >> 8) & 255;
    const b = hexInt & 255;
    return alpha !== undefined ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
  }

  function escHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (m) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[m]));
  }

  function create(opts) {
    const primaryCanvas = opts.primaryCanvas;
    const secondaryCanvas = opts.secondaryCanvas;
    const primaryOverlay = opts.primaryOverlay;
    const secondaryOverlay = opts.secondaryOverlay;
    const compassEl = opts.compassEl;
    const scaleEl = opts.scaleEl;
    const engineTagEl = opts.engineTagEl;

    const THREE = window.THREE || null;

    const state = {
      theme: "light",
      splitMode: false,
      autoRotate: false,
      focusedOnGap: false,
      webglActive: false,

      yaw: -0.66,
      pitch: 0.48,
      dist: 6.5,
      target: [0, 0, 0],
      bounds: computeBounds([]),
      fovDeg: 36,

      anim: null,
      rafId: 0,
      autoRotateRaf: 0,
      lastTick: 0,

      primarySpec: null,
      secondarySpec: null,
      selectedFractureId: null,
      hoverFractureId: null,

      onSelectFracture: opts.onSelectFracture || null,
      onHoverFracture: opts.onHoverFracture || null,
    };

    function createPaneBackend(canvas, overlayEl, paneKey) {
      const pane = {
        key: paneKey,
        canvas,
        overlayEl,
        w: 1,
        h: 1,
        dpr: 1,
        useWebGL: false,
        renderer: null,
        scene: null,
        camera: null,
        rootGroup: null,
        pickMeshes: [],
        ctx2d: null,
      };

      if (THREE) {
        try {
          const gl = canvas.getContext("webgl2", { antialias: true, alpha: true, preserveDrawingBuffer: true })
            || canvas.getContext("webgl", { antialias: true, alpha: true, preserveDrawingBuffer: true });
          if (gl) {
            const renderer = new THREE.WebGLRenderer({
              canvas,
              context: gl,
              antialias: true,
              alpha: true,
              preserveDrawingBuffer: true,
              powerPreference: "high-performance",
            });
            renderer.setClearColor(0x000000, 0);
            if ("outputColorSpace" in renderer && THREE.SRGBColorSpace) {
              renderer.outputColorSpace = THREE.SRGBColorSpace;
            }
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(state.fovDeg, 1, 0.002, 500);
            camera.up.set(0, 0, 1);

            /* 柔和三点科研工作室布光 */
            const hemi = new THREE.HemisphereLight(0xfffefb, 0xb8c0bc, 0.96);
            hemi.position.set(0, 0, 24);
            scene.add(hemi);

            const keyLight = new THREE.DirectionalLight(0xfffdf8, 0.92);
            keyLight.position.set(14, -16, 22);
            scene.add(keyLight);

            const rimLight = new THREE.DirectionalLight(0xd2eee6, 0.52);
            rimLight.position.set(-16, 14, 10);
            scene.add(rimLight);

            const fillLight = new THREE.DirectionalLight(0xeef2f6, 0.32);
            fillLight.position.set(0, 12, -12);
            scene.add(fillLight);

            const rootGroup = new THREE.Group();
            scene.add(rootGroup);

            pane.useWebGL = true;
            pane.renderer = renderer;
            pane.scene = scene;
            pane.camera = camera;
            pane.rootGroup = rootGroup;
          }
        } catch (_err) {
          pane.useWebGL = false;
        }
      }

      if (!pane.useWebGL) {
        pane.ctx2d = canvas.getContext("2d");
      }
      return pane;
    }

    const primaryPane = createPaneBackend(primaryCanvas, primaryOverlay, "primary");
    const secondaryPane = secondaryCanvas
      ? createPaneBackend(secondaryCanvas, secondaryOverlay, "secondary")
      : null;

    state.webglActive = primaryPane.useWebGL;
    if (engineTagEl) {
      engineTagEl.textContent = state.webglActive ? "WebGL 3D · Z-Buffer" : "Canvas 2D";
    }

    function disposeGroup(group) {
      if (!group) return;
      while (group.children.length > 0) {
        const child = group.children.pop();
        disposeGroup(child);
        if (child.geometry && typeof child.geometry.dispose === "function") {
          child.geometry.dispose();
        }
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m && m.dispose && m.dispose());
          } else if (typeof child.material.dispose === "function") {
            child.material.dispose();
          }
        }
      }
    }

    function getFractureStyle(comp, triId, spec, pal) {
      const wv = spec.witness_view || "nominal";
      const isTop1 = spec.top1FractureId === comp && spec.status === "AMBIGUOUS_WITH_VERIFIED_WITNESSES";
      const isSel = state.selectedFractureId === comp;
      const isHov = state.hoverFractureId === comp;
      const meta = (spec.fractureMeta && spec.fractureMeta[comp]) || {};
      const role = meta.role || "INTERIOR";

      let color = pal.nominalInterior;
      let edgeColor = pal.edgeDefault;
      let opacity = 0.64;
      let emissive = 0x000000;

      if (wv === "connected") {
        if (spec.pathTriangles && spec.pathTriangles.has(triId)) {
          color = pal.connPathTri;
          edgeColor = pal.connPathTri;
          opacity = 0.84;
          emissive = state.theme === "dark" ? 0x043322 : 0x03261a;
        } else if (spec.pathFractures && spec.pathFractures.has(comp)) {
          color = pal.connPathFrag;
          edgeColor = pal.connPathTri;
          opacity = 0.72;
          emissive = state.theme === "dark" ? 0x032418 : 0x021a12;
        } else {
          color = pal.connOther;
          edgeColor = pal.edgeMuted;
          opacity = 0.16;
        }
      } else if (wv === "disconnected") {
        if (spec.cutSFractures && spec.cutSFractures.has(comp)) {
          color = pal.discCutS;
          edgeColor = pal.discCutS;
          opacity = 0.74;
          emissive = state.theme === "dark" ? 0x052436 : 0x031826;
        } else if (spec.cutTFractures && spec.cutTFractures.has(comp)) {
          color = pal.discCutT;
          edgeColor = pal.discCutT;
          opacity = 0.78;
          emissive = state.theme === "dark" ? 0x300808 : 0x240606;
        } else {
          color = pal.discOther;
          edgeColor = pal.edgeMuted;
          opacity = 0.20;
        }
      } else {
        if (isTop1) {
          color = pal.nominalTop1;
          edgeColor = pal.edgeTop1;
          opacity = 0.82;
          emissive = state.theme === "dark" ? 0x331b03 : 0x241202;
        } else if (role === "ANCHOR_A") {
          color = pal.nominalAnchorA;
          edgeColor = pal.nominalAnchorA;
          opacity = 0.76;
        } else if (role === "ANCHOR_B") {
          color = pal.nominalAnchorB;
          edgeColor = pal.nominalAnchorB;
          opacity = 0.76;
        } else {
          color = pal.nominalInterior;
          edgeColor = pal.edgeDefault;
          opacity = 0.56;
        }
      }

      if (isSel) {
        opacity = Math.min(0.92, opacity + 0.14);
        emissive = state.theme === "dark" ? 0x16362c : 0x0d261e;
      } else if (isHov) {
        opacity = Math.min(0.88, opacity + 0.08);
      }

      return { color, edgeColor, opacity, emissive, isTop1, isSel, isHov, role };
    }

    function buildThreeSceneForPane(pane, spec) {
      if (!pane || !pane.useWebGL || !pane.rootGroup) return;
      const pal = PALETTE[state.theme] || PALETTE.light;
      disposeGroup(pane.rootGroup);
      pane.pickMeshes = [];
      if (!spec || !spec.triangles || !spec.triangles.length) return;

      const b = spec.bounds || state.bounds;

      // 1. 精密地质测绘基准面（圆形方位刻度环 + 细腻网格 + 地面轮廓投影）
      const bedZ = b.min[2] - b.radius * 0.14;
      const gridSpan = Math.max(2.0, Math.ceil(b.radius * 2.5));
      const gridDivs = Math.min(20, Math.max(8, Math.round(gridSpan * 2)));
      const gridHelper = new THREE.GridHelper(gridSpan, gridDivs, pal.gridMajor, pal.gridMinor);
      gridHelper.rotation.x = Math.PI / 2;
      gridHelper.position.set(b.center[0], b.center[1], bedZ);
      if (gridHelper.material) {
        gridHelper.material.transparent = true;
        gridHelper.material.opacity = state.theme === "dark" ? 0.28 : 0.42;
      }
      pane.rootGroup.add(gridHelper);

      // 外围测绘圆环
      const ringRad = gridSpan * 0.56;
      const circlePts = [];
      for (let i = 0; i <= 72; i++) {
        const a = (i / 72) * Math.PI * 2;
        circlePts.push(new THREE.Vector3(b.center[0] + Math.cos(a) * ringRad, b.center[1] + Math.sin(a) * ringRad, bedZ + 0.001));
      }
      pane.rootGroup.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(circlePts),
          new THREE.LineBasicMaterial({ color: pal.gridMajor, transparent: true, opacity: state.theme === "dark" ? 0.32 : 0.48 })
        )
      );

      // 地面柔光轮廓投影（让悬空的裂隙网络拥有真实的空间重量感）
      const shadowCoords = [];
      for (const tri of spec.triangles) {
        const v0 = tri.vertsT[0], v1 = tri.vertsT[1], v2 = tri.vertsT[2];
        shadowCoords.push(
          v0[0], v0[1], bedZ + 0.002,
          v1[0], v1[1], bedZ + 0.002,
          v2[0], v2[1], bedZ + 0.002
        );
      }
      if (shadowCoords.length > 0) {
        const sGeom = new THREE.BufferGeometry();
        sGeom.setAttribute("position", new THREE.Float32BufferAttribute(shadowCoords, 3));
        const sMat = new THREE.MeshBasicMaterial({
          color: pal.shadowColor,
          transparent: true,
          opacity: pal.shadowOpacity,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        pane.rootGroup.add(new THREE.Mesh(sGeom, sMat));
      }

      // 极简三维坐标轴指示针
      const axisOrigin = new THREE.Vector3(
        b.center[0] - gridSpan * 0.48,
        b.center[1] - gridSpan * 0.48,
        bedZ + 0.006
      );
      const axisLen = gridSpan * 0.12;
      const addAxis = (dx, dy, dz, hex) => {
        const g = new THREE.BufferGeometry().setFromPoints([
          axisOrigin,
          new THREE.Vector3(axisOrigin.x + dx, axisOrigin.y + dy, axisOrigin.z + dz),
        ]);
        pane.rootGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.7 })));
      };
      addAxis(axisLen, 0, 0, 0xc93b32);
      addAxis(0, axisLen, 0, 0x0b8c60);
      addAxis(0, 0, axisLen * 0.8, 0x2563eb);

      // 2. 裂隙多边形 PBR 矿物材质与轮廓描边
      const byComp = new Map();
      for (const tri of spec.triangles) {
        if (!byComp.has(tri.component)) byComp.set(tri.component, []);
        byComp.get(tri.component).push(tri);
      }

      byComp.forEach((tris, comp) => {
        for (const tri of tris) {
          const style = getFractureStyle(comp, tri.id, spec, pal);
          const geom = new THREE.BufferGeometry();
          const v0 = tri.vertsT[0], v1 = tri.vertsT[1], v2 = tri.vertsT[2];
          geom.setAttribute(
            "position",
            new THREE.Float32BufferAttribute([v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]], 3)
          );
          geom.computeVertexNormals();

          const mat = new THREE.MeshStandardMaterial({
            color: style.color,
            emissive: style.emissive,
            roughness: 0.44,
            metalness: 0.08,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: style.opacity,
            depthWrite: style.opacity > 0.45,
            polygonOffset: true,
            polygonOffsetFactor: 1,
            polygonOffsetUnits: 1,
          });
          const mesh = new THREE.Mesh(geom, mat);
          mesh.userData = { fractureId: comp, triangleId: tri.id };
          pane.rootGroup.add(mesh);
          pane.pickMeshes.push(mesh);

          // 极细内部三角网格线（测绘结构线）
          const wireGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(...v0),
            new THREE.Vector3(...v1),
            new THREE.Vector3(...v2),
            new THREE.Vector3(...v0),
          ]);
          const wireMat = new THREE.LineBasicMaterial({
            color: style.edgeColor,
            transparent: true,
            opacity: style.opacity > 0.45 ? 0.12 : 0.05,
          });
          pane.rootGroup.add(new THREE.Line(wireGeom, wireMat));
        }

        // 外边界立体轮廓线
        const sampleStyle = getFractureStyle(comp, "", spec, pal);
        const segs = (spec.boundaryEdges && spec.boundaryEdges.get(comp)) || [];
        if (segs.length > 0) {
          const lineCoords = [];
          for (const [a, bPt] of segs) {
            lineCoords.push(a[0], a[1], a[2], bPt[0], bPt[1], bPt[2]);
          }
          const bGeom = new THREE.BufferGeometry();
          bGeom.setAttribute("position", new THREE.Float32BufferAttribute(lineCoords, 3));
          const bMat = new THREE.LineBasicMaterial({
            color: sampleStyle.isTop1 ? pal.edgeTop1 : sampleStyle.edgeColor,
            transparent: true,
            opacity: sampleStyle.isTop1 ? 0.96 : Math.min(0.88, sampleStyle.opacity + 0.18),
          });
          pane.rootGroup.add(new THREE.LineSegments(bGeom, bMat));
        }

        // 保留真实拓扑孔洞：青玉立体管线内环
        const holeRings = (spec.holeRingsByComp && spec.holeRingsByComp.get(comp)) || [];
        for (const hRing of holeRings) {
          if (hRing.length < 3) continue;
          const pts = hRing.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
          pts.push(pts[0].clone());
          const curve = new THREE.CatmullRomCurve3(pts, true, "centripetal", 0.01);
          const tubeRadius = Math.max(0.011, b.radius * 0.007);
          const tubeGeom = new THREE.TubeGeometry(curve, Math.max(24, hRing.length * 4), tubeRadius, 8, true);
          const tubeMat = new THREE.MeshStandardMaterial({
            color: pal.edgeHole,
            emissive: 0x0f766e,
            roughness: 0.25,
            metalness: 0.2,
          });
          pane.rootGroup.add(new THREE.Mesh(tubeGeom, tubeMat));
        }
      });

      // 3. 连通见证双层发光翠绿通道与接触节点
      if (spec.witness_view === "connected" && spec.pathPolyline3d && spec.pathPolyline3d.length >= 2) {
        const pathPts = spec.pathPolyline3d.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
        const coreRad = Math.max(0.014, b.radius * 0.0095);
        const haloRad = coreRad * 1.95;

        for (let i = 0; i < pathPts.length - 1; i++) {
          const pA = pathPts[i], pB = pathPts[i + 1];
          if (pA.distanceTo(pB) < 1e-6) continue;
          const segCurve = new THREE.LineCurve3(pA, pB);

          const coreGeom = new THREE.TubeGeometry(segCurve, 8, coreRad, 10, false);
          const coreMat = new THREE.MeshStandardMaterial({
            color: pal.chainTube,
            emissive: 0x065f46,
            roughness: 0.2,
            metalness: 0.15,
          });
          pane.rootGroup.add(new THREE.Mesh(coreGeom, coreMat));

          const haloGeom = new THREE.TubeGeometry(segCurve, 8, haloRad, 10, false);
          const haloMat = new THREE.MeshBasicMaterial({
            color: pal.chainHalo,
            transparent: true,
            opacity: 0.24,
            depthWrite: false,
          });
          pane.rootGroup.add(new THREE.Mesh(haloGeom, haloMat));
        }

        const sphereRad = Math.max(0.022, b.radius * 0.014);
        const sphereGeom = new THREE.SphereGeometry(sphereRad, 14, 14);
        const sphereMat = new THREE.MeshStandardMaterial({
          color: 0x34d399,
          emissive: 0x047857,
          roughness: 0.18,
          metalness: 0.25,
        });
        for (const cp of spec.contactPoints || []) {
          if (!cp.point) continue;
          const sm = new THREE.Mesh(sphereGeom, sphereMat);
          sm.position.set(cp.point[0], cp.point[1], cp.point[2]);
          pane.rootGroup.add(sm);
        }
      }

      // 4. 断割见证 S/T 最小割断口 CAD 精密立体游标卡尺
      if (spec.witness_view === "disconnected" && spec.cutGap && spec.cutGap.from_pt && spec.cutGap.to_pt) {
        const pFrom = new THREE.Vector3(...spec.cutGap.from_pt);
        const pTo = new THREE.Vector3(...spec.cutGap.to_pt);
        const pMid = pFrom.clone().add(pTo).multiplyScalar(0.5);
        const gapDist = pFrom.distanceTo(pTo);

        const markerRad = state.focusedOnGap
          ? Math.max(0.00032, Math.min(0.0038, gapDist * 0.2))
          : Math.max(0.014, b.radius * 0.011);
        const sGeom = new THREE.SphereGeometry(markerRad, 16, 16);
        const matS = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, roughness: 0.2 });
        const matT = new THREE.MeshStandardMaterial({ color: 0xf87171, emissive: 0xc93b32, roughness: 0.2 });

        const mFrom = new THREE.Mesh(sGeom, matS);
        mFrom.position.copy(pFrom);
        pane.rootGroup.add(mFrom);

        const mTo = new THREE.Mesh(sGeom, matT);
        mTo.position.copy(pTo);
        pane.rootGroup.add(mTo);

        if (gapDist > 1e-6) {
          const calCurve = new THREE.LineCurve3(pFrom, pTo);
          const calRad = state.focusedOnGap
            ? Math.max(0.00016, gapDist * 0.08)
            : Math.max(0.0055, b.radius * 0.0045);
          pane.rootGroup.add(
            new THREE.Mesh(
              new THREE.TubeGeometry(calCurve, 6, calRad, 10, false),
              new THREE.MeshBasicMaterial({ color: pal.gapGlow })
            )
          );
        }

        const haloRad = state.focusedOnGap
          ? Math.max(0.0022, gapDist * 1.45)
          : Math.max(0.075, b.radius * 0.052);
        const haloMesh = new THREE.Mesh(
          new THREE.RingGeometry(haloRad * 0.85, haloRad, 36),
          new THREE.MeshBasicMaterial({
            color: pal.gapLine,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.88,
          })
        );
        haloMesh.position.copy(pMid);
        haloMesh.userData = { isGapHalo: true };
        pane.rootGroup.add(haloMesh);
      }

      // 5. 端点 A / B 精密立体标桩
      if (spec.endpoints) {
        const addPylon = (pt, hex, emHex) => {
          if (!pt) return;
          const g = new THREE.Group();
          g.position.set(pt[0], pt[1], pt[2]);
          const r = Math.max(0.04, b.radius * 0.023);
          g.add(
            new THREE.Mesh(
              new THREE.OctahedronGeometry(r, 0),
              new THREE.MeshStandardMaterial({ color: hex, emissive: emHex, roughness: 0.22, metalness: 0.3 })
            )
          );
          const ring = new THREE.Mesh(
            new THREE.RingGeometry(r * 1.32, r * 1.56, 28),
            new THREE.MeshBasicMaterial({ color: hex, side: THREE.DoubleSide, transparent: true, opacity: 0.65 })
          );
          ring.userData = { isGapHalo: true };
          g.add(ring);
          pane.rootGroup.add(g);
        };
        addPylon(spec.endpoints.A, 0x0f766e, 0x115e59);
        addPylon(spec.endpoints.B, 0x1d4ed8, 0x1e3a8a);
      }
    }

    function computeCameraEye() {
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);
      const cy = Math.cos(state.yaw);
      const sy = Math.sin(state.yaw);
      return [
        state.target[0] + state.dist * cp * sy,
        state.target[1] - state.dist * cp * cy,
        state.target[2] + state.dist * sp,
      ];
    }

    function projectPoint3D(pt, pane) {
      const w = pane.w || 1;
      const h = pane.h || 1;
      if (pane.useWebGL && pane.camera) {
        const v = new THREE.Vector3(pt[0], pt[1], pt[2]);
        const camDist = v.distanceTo(pane.camera.position);
        v.project(pane.camera);
        return {
          x: (v.x * 0.5 + 0.5) * w,
          y: (-v.y * 0.5 + 0.5) * h,
          depth: camDist,
          visible: v.z >= -1.0 && v.z <= 1.0,
        };
      }
      const dx = pt[0] - state.target[0];
      const dy = pt[1] - state.target[1];
      const dz = pt[2] - state.target[2];
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = dx * cy + dy * sy;
      const y1 = -dx * sy + dy * cy;
      const z2 = y1 * sp + dz * cp;
      const depth = state.dist - (y1 * cp - dz * sp);
      const safeDepth = Math.max(0.01, depth);
      const focal = (h * 0.5) / Math.tan((state.fovDeg * 0.5 * Math.PI) / 180);
      return {
        x: w * 0.5 + (x1 * focal) / safeDepth,
        y: h * 0.5 - (z2 * focal) / safeDepth,
        depth: safeDepth,
        visible: depth > 0.01,
      };
    }

    function renderCanvas2DFallback(pane, spec) {
      const ctx = pane.ctx2d;
      if (!ctx) return;
      const pal = PALETTE[state.theme] || PALETTE.light;
      ctx.save();
      ctx.scale(pane.dpr, pane.dpr);
      ctx.clearRect(0, 0, pane.w, pane.h);
      if (!spec || !spec.triangles) {
        ctx.restore();
        return;
      }

      const drawList = spec.triangles.map((tri) => {
        const pts = tri.vertsT.map((p) => projectPoint3D(p, pane));
        const depth = (pts[0].depth + pts[1].depth + pts[2].depth) / 3;
        return { tri, pts, depth };
      });
      drawList.sort((a, b) => b.depth - a.depth);

      for (const item of drawList) {
        const st = getFractureStyle(item.tri.component, item.tri.id, spec, pal);
        ctx.beginPath();
        ctx.moveTo(item.pts[0].x, item.pts[0].y);
        ctx.lineTo(item.pts[1].x, item.pts[1].y);
        ctx.lineTo(item.pts[2].x, item.pts[2].y);
        ctx.closePath();
        ctx.fillStyle = hexToCss(st.color, st.opacity);
        ctx.fill();
        ctx.strokeStyle = hexToCss(st.edgeColor, 0.28);
        ctx.lineWidth = 0.65;
        ctx.stroke();
      }
      ctx.restore();
    }

    function renderOverlayAnnotations(pane, spec) {
      const el = pane.overlayEl;
      if (!el) return;
      if (!spec || !spec.triangles || !spec.triangles.length) {
        el.innerHTML = "";
        return;
      }
      const pins = [];
      const cx = (x) => clamp(x, 38, Math.max(42, pane.w - 42));
      const cy = (y) => clamp(y, 26, Math.max(30, pane.h - 24));

      // 端点 A / B 精简测绘标牌
      if (spec.endpoints && spec.endpoints.A) {
        const qA = projectPoint3D(spec.endpoints.A, pane);
        if (qA.visible) {
          pins.push(
            `<div class="cad-pin pin-ep-a" style="left:${cx(qA.x)}px;top:${cy(qA.y - 14)}px">` +
            `<i class="p-node"></i><span>A</span></div>`
          );
        }
      }
      if (spec.endpoints && spec.endpoints.B) {
        const qB = projectPoint3D(spec.endpoints.B, pane);
        if (qB.visible) {
          pins.push(
            `<div class="cad-pin pin-ep-b" style="left:${cx(qB.x)}px;top:${cy(qB.y - 14)}px">` +
            `<i class="p-node"></i><span>B</span></div>`
          );
        }
      }

      // 真实拓扑孔洞标识
      if (spec.holeProbes && spec.holeProbes.length) {
        for (const h of spec.holeProbes) {
          const pt = (h.probes || [])[0];
          if (!pt) continue;
          const qH = projectPoint3D(pt, pane);
          if (qH.visible) {
            pins.push(
              `<div class="cad-pin pin-hole" style="left:${cx(qH.x)}px;top:${cy(qH.y)}px" title="真实拓扑孔洞保留（面积 ${Number(h.preserved_hole_area_m2 || 0).toFixed(4)} m²）">` +
              `<i class="p-ring"></i><span>真孔洞 ×${h.preserved_hole_count || 1}</span></div>`
            );
          }
        }
      }

      // #1 优先复核临界裂隙标注（仅单视口名义态显示，分屏时不遮挡对比画面）
      if (
        !state.splitMode &&
        spec.witness_view === "nominal" &&
        spec.top1FractureId &&
        spec.status === "AMBIGUOUS_WITH_VERIFIED_WITNESSES" &&
        spec.compCentroids
      ) {
        const cPt = spec.compCentroids.get(spec.top1FractureId);
        if (cPt) {
          const qT = projectPoint3D(cPt, pane);
          const meta = (spec.fractureMeta && spec.fractureMeta[spec.top1FractureId]) || {};
          const shortLbl = meta.short_label || spec.top1FractureId.slice(0, 10);
          if (qT.visible) {
            pins.push(
              `<button type="button" class="cad-pin pin-top1" data-action="vp-pick-fracture" data-fracture="${escHtml(spec.top1FractureId)}" ` +
              `style="left:${cx(qT.x)}px;top:${cy(qT.y - 16)}px">` +
              `<span class="p-idx">#1</span><span>${escHtml(shortLbl)}</span></button>`
            );
          }
        }
      }

      // 断割见证 S/T 最小割断口交互聚焦标尺
      if (spec.witness_view === "disconnected" && spec.cutGap && spec.cutGap.from_pt && spec.cutGap.to_pt) {
        const g = spec.cutGap;
        const mid = lerpVec3(g.from_pt, g.to_pt, 0.5);
        const qG = projectPoint3D(mid, pane);
        if (qG.visible) {
          const distM = Number(g.distance_m || 0);
          const mmTxt = distM < 0.05
            ? `${(distM * 1000).toFixed(2)} mm (${distM.toFixed(5)} m)`
            : `${distM.toFixed(4)} m`;
          pins.push(
            `<button type="button" class="cad-pin pin-gap ${state.focusedOnGap ? "is-focused" : ""}" ` +
            `data-action="focus-gap" data-testid="pin-focus-gap-${pane.key}" ` +
            `style="left:${cx(qG.x)}px;top:${cy(qG.y + 18)}px" title="点击推进 3D 微距镜头聚焦断口">` +
            `<i class="p-pulse"></i><span>S/T 最小割断口 ${escHtml(mmTxt)}</span>` +
            `<span class="p-cta">${state.focusedOnGap ? "复位" : "微距 ↗"}</span></button>`
          );
        }
      }

      el.innerHTML = pins.join("");
    }

    function updateCompassAndScale() {
      if (compassEl) {
        const degYaw = ((-state.yaw * 180) / Math.PI).toFixed(1);
        const degPitch = ((state.pitch * 180) / Math.PI).toFixed(0);
        compassEl.innerHTML = `
          <svg viewBox="0 0 56 56" width="44" height="44" aria-hidden="true">
            <circle cx="28" cy="28" r="23" class="cmp-ring"/>
            <g transform="rotate(${degYaw} 28 28)">
              <polygon points="28,8 31.5,28 28,24.5 24.5,28" class="cmp-n"/>
              <polygon points="28,48 31.5,28 28,31.5 24.5,28" class="cmp-s"/>
            </g>
            <circle cx="28" cy="28" r="1.8" class="cmp-dot"/>
          </svg>
          <span class="cmp-readout">${degPitch}°</span>`;
      }

      if (scaleEl && primaryPane.w > 20) {
        const fovRad = (state.fovDeg * Math.PI) / 180;
        const metersPerPixel = (2 * state.dist * Math.tan(fovRad * 0.5)) / Math.max(1, primaryPane.h);
        const nice = niceScaleMetric(metersPerPixel * 82);
        const barPx = clamp(Math.round(nice.meters / Math.max(1e-9, metersPerPixel)), 34, 130);
        const half = Math.round(barPx / 2);
        scaleEl.innerHTML =
          `<span class="sc-num">0</span>` +
          `<span class="sc-track" style="width:${barPx}px"><i class="sc-fill" style="width:${half}px"></i></span>` +
          `<span class="sc-num">${escHtml(nice.label)}</span>`;
      }
    }

    function renderSinglePane(pane, spec) {
      if (!pane || pane.w <= 2 || pane.h <= 2) return;
      const eye = computeCameraEye();
      if (pane.useWebGL && pane.renderer && pane.camera && pane.scene) {
        pane.camera.aspect = pane.w / Math.max(1, pane.h);
        pane.camera.near = Math.max(0.0005, state.dist * 0.008);
        pane.camera.far = Math.max(50, state.dist * 18 + state.bounds.radius * 12);
        pane.camera.updateProjectionMatrix();
        pane.camera.position.set(eye[0], eye[1], eye[2]);
        pane.camera.lookAt(state.target[0], state.target[1], state.target[2]);

        pane.rootGroup.traverse((obj) => {
          if (obj.userData && obj.userData.isGapHalo) {
            obj.lookAt(pane.camera.position);
          }
        });
        pane.renderer.render(pane.scene, pane.camera);
      } else {
        renderCanvas2DFallback(pane, spec);
      }
      renderOverlayAnnotations(pane, spec);
    }

    function renderNow() {
      state.rafId = 0;
      renderSinglePane(primaryPane, state.primarySpec);
      if (state.splitMode && secondaryPane) {
        renderSinglePane(secondaryPane, state.secondarySpec);
      }
      updateCompassAndScale();
    }

    function requestRender() {
      if (!state.rafId) {
        state.rafId = window.requestAnimationFrame(renderNow);
      }
    }

    function resizePane(pane) {
      if (!pane || !pane.canvas) return;
      const rect = pane.canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      pane.dpr = Math.min(2, window.devicePixelRatio || 1);
      pane.w = Math.max(1, Math.round(rect.width));
      pane.h = Math.max(1, Math.round(rect.height));
      if (pane.useWebGL && pane.renderer) {
        pane.renderer.setPixelRatio(pane.dpr);
        pane.renderer.setSize(pane.w, pane.h, false);
      } else {
        pane.canvas.width = Math.round(pane.w * pane.dpr);
        pane.canvas.height = Math.round(pane.h * pane.dpr);
      }
    }

    function resize() {
      resizePane(primaryPane);
      if (state.splitMode && secondaryPane) {
        resizePane(secondaryPane);
      }
      requestRender();
    }

    function prepareSpecGeometry(rawSpec) {
      if (!rawSpec) return null;
      const tVal = Number(rawSpec.t_value || 0);
      const interpAlpha = rawSpec.interpAlpha;
      const ampFactor = Number(rawSpec.ampFactor || 1);

      const tris = (rawSpec.triangles || []).map((tri, idx) => {
        const v = tri.v || [0, 0, 0];
        let vertsT;
        if (
          typeof interpAlpha === "number" &&
          rawSpec.connTriangles &&
          rawSpec.discTriangles &&
          rawSpec.connTriangles[idx] &&
          rawSpec.discTriangles[idx]
        ) {
          const cTri = rawSpec.connTriangles[idx];
          const dTri = rawSpec.discTriangles[idx];
          const tConn = Number(rawSpec.connT || 0);
          const tDisc = Number(rawSpec.discT || 0);
          vertsT = (tri.vertices || []).map((p0, vi) => {
            const vc = cTri.v || v;
            const vd = dTri.v || v;
            const pc0 = (cTri.vertices && cTri.vertices[vi]) || p0;
            const pd0 = (dTri.vertices && dTri.vertices[vi]) || p0;
            const pConn = [
              pc0[0] + tConn * vc[0] * ampFactor,
              pc0[1] + tConn * vc[1] * ampFactor,
              pc0[2] + tConn * vc[2] * ampFactor,
            ];
            const pDisc = [
              pd0[0] + tDisc * vd[0] * ampFactor,
              pd0[1] + tDisc * vd[1] * ampFactor,
              pd0[2] + tDisc * vd[2] * ampFactor,
            ];
            return lerpVec3(pConn, pDisc, interpAlpha);
          });
        } else {
          vertsT = (tri.vertices || []).map((p) => [
            p[0] + tVal * v[0] * ampFactor,
            p[1] + tVal * v[1] * ampFactor,
            p[2] + tVal * v[2] * ampFactor,
          ]);
        }
        return {
          id: tri.id,
          component: tri.component,
          label: tri.label || "",
          vertsT,
          centroid: centroid3(vertsT),
          normal: normal3(vertsT),
        };
      });

      const allPts = [];
      if (rawSpec.endpoints && rawSpec.endpoints.A) allPts.push(rawSpec.endpoints.A);
      if (rawSpec.endpoints && rawSpec.endpoints.B) allPts.push(rawSpec.endpoints.B);
      for (const tri of tris) {
        for (const p of tri.vertsT) allPts.push(p);
      }

      const byComp = new Map();
      const compPts = new Map();
      for (const tri of tris) {
        if (!byComp.has(tri.component)) byComp.set(tri.component, []);
        byComp.get(tri.component).push(tri);
        if (!compPts.has(tri.component)) compPts.set(tri.component, []);
        compPts.get(tri.component).push(...tri.vertsT);
      }
      const compCentroids = new Map();
      compPts.forEach((pts, comp) => compCentroids.set(comp, centroid3(pts)));

      const boundaryEdges = new Map();
      const vkey = (p) => `${Math.round(p[0] * 1e4)},${Math.round(p[1] * 1e4)},${Math.round(p[2] * 1e4)}`;
      byComp.forEach((cTris, comp) => {
        const counts = new Map();
        const segs = new Map();
        for (const tri of cTris) {
          const vs = tri.vertsT;
          const pairs = [[vs[0], vs[1]], [vs[1], vs[2]], [vs[2], vs[0]]];
          for (const [a, bPt] of pairs) {
            const ka = vkey(a), kb = vkey(bPt);
            const ek = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
            counts.set(ek, (counts.get(ek) || 0) + 1);
            if (!segs.has(ek)) segs.set(ek, [a, bPt]);
          }
        }
        const boundary = [];
        counts.forEach((cnt, ek) => {
          if (cnt === 1) boundary.push(segs.get(ek));
        });
        boundaryEdges.set(comp, boundary);
      });

      const holeRingsByComp = new Map();
      if (rawSpec.fractureContours) {
        Object.keys(rawSpec.fractureContours).forEach((comp) => {
          const fc = rawSpec.fractureContours[comp];
          const v = fc.v || [0, 0, 0];
          const rings = (fc.hole_rings || []).map((ring) =>
            ring.map((p) => [
              p[0] + tVal * v[0] * ampFactor,
              p[1] + tVal * v[1] * ampFactor,
              p[2] + tVal * v[2] * ampFactor,
            ])
          );
          if (rings.length > 0) holeRingsByComp.set(comp, rings);
        });
      }

      return {
        cluster_id: rawSpec.cluster_id,
        mode_id: rawSpec.mode_id || "m0",
        t_value: tVal,
        witness_view: rawSpec.witness_view || "nominal",
        status: rawSpec.status || "UNKNOWN",
        endpoints: rawSpec.endpoints || null,
        triangles: tris,
        bounds: computeBounds(allPts),
        boundaryEdges,
        holeRingsByComp,
        compCentroids,
        fractureMeta: rawSpec.fractureMeta || {},
        pathFractures: new Set(rawSpec.pathFractures || []),
        pathTriangles: new Set(rawSpec.pathTriangles || []),
        pathPolyline3d: rawSpec.pathPolyline3d || [],
        contactPoints: rawSpec.contactPoints || [],
        cutSFractures: new Set(rawSpec.cutSFractures || []),
        cutTFractures: new Set(rawSpec.cutTFractures || []),
        cutGap: rawSpec.cutGap || null,
        top1FractureId: rawSpec.top1FractureId || null,
        holeProbes: rawSpec.holeProbes || [],
      };
    }

    function animateCameraTo(goal, durationMs = 500) {
      const start = {
        yaw: state.yaw,
        pitch: state.pitch,
        dist: state.dist,
        target: state.target.slice(),
      };
      const t0 = performance.now();

      function step(now) {
        const u = clamp((now - t0) / Math.max(1, durationMs), 0, 1);
        const ease = 1 - Math.pow(1 - u, 3);
        if (goal.yaw !== undefined) state.yaw = lerp(start.yaw, goal.yaw, ease);
        if (goal.pitch !== undefined) state.pitch = lerp(start.pitch, goal.pitch, ease);
        if (goal.dist !== undefined) state.dist = lerp(start.dist, goal.dist, ease);
        if (goal.target !== undefined) state.target = lerpVec3(start.target, goal.target, ease);
        renderNow();
        if (u < 1) state.anim = window.requestAnimationFrame(step);
        else state.anim = null;
      }

      if (state.anim) window.cancelAnimationFrame(state.anim);
      state.anim = window.requestAnimationFrame(step);
    }

    function defaultCamDist() {
      const isCompact = primaryPane && primaryPane.w > 0 && primaryPane.w < 540;
      return state.bounds.radius * (isCompact ? 3.25 : 2.82);
    }

    function setScenes(primaryRawSpec, secondaryRawSpec, optsSet = {}) {
      const prevCluster = state.primarySpec ? state.primarySpec.cluster_id : null;
      const nextCluster = primaryRawSpec ? primaryRawSpec.cluster_id : null;
      const keepCamera = !!optsSet.keepCamera && prevCluster === nextCluster;

      state.splitMode = !!optsSet.splitMode && !!secondaryRawSpec;
      state.primarySpec = prepareSpecGeometry(primaryRawSpec);
      state.secondarySpec = state.splitMode ? prepareSpecGeometry(secondaryRawSpec) : null;

      resize();

      if (state.primarySpec) {
        state.bounds = state.primarySpec.bounds;
        if (!keepCamera) {
          state.focusedOnGap = false;
          state.target = state.bounds.center.slice();
          state.yaw = -0.66;
          state.pitch = 0.48;
          state.dist = defaultCamDist();
        }
      }

      buildThreeSceneForPane(primaryPane, state.primarySpec);
      if (state.splitMode && secondaryPane) {
        buildThreeSceneForPane(secondaryPane, state.secondarySpec);
      }
      requestRender();
    }

    function focusGap(cutGap, animate = true) {
      const gap = cutGap
        || (state.primarySpec && state.primarySpec.cutGap)
        || (state.secondarySpec && state.secondarySpec.cutGap);
      if (!gap || !gap.from_pt || !gap.to_pt) return false;

      if (state.focusedOnGap) {
        fitView(animate);
        return false;
      }

      state.focusedOnGap = true;
      buildThreeSceneForPane(primaryPane, state.primarySpec);
      if (state.splitMode && secondaryPane) {
        buildThreeSceneForPane(secondaryPane, state.secondarySpec);
      }

      const mid = lerpVec3(gap.from_pt, gap.to_pt, 0.5);
      const gapDist = Math.hypot(
        gap.to_pt[0] - gap.from_pt[0],
        gap.to_pt[1] - gap.from_pt[1],
        gap.to_pt[2] - gap.from_pt[2]
      );
      const targetDist = clamp(Math.max(gapDist * 28, state.bounds.radius * 0.22), 0.04, state.bounds.radius * 0.8);
      const dx = gap.to_pt[0] - gap.from_pt[0];
      const dy = gap.to_pt[1] - gap.from_pt[1];
      const perpYaw = Math.atan2(dy, dx) - 0.35;

      const goal = { target: mid, dist: targetDist, yaw: perpYaw, pitch: 0.36 };
      if (animate) animateCameraTo(goal, 540);
      else {
        state.target = goal.target;
        state.dist = goal.dist;
        state.yaw = goal.yaw;
        state.pitch = goal.pitch;
        requestRender();
      }
      return true;
    }

    function fitView(animate = true) {
      state.focusedOnGap = false;
      buildThreeSceneForPane(primaryPane, state.primarySpec);
      if (state.splitMode && secondaryPane) {
        buildThreeSceneForPane(secondaryPane, state.secondarySpec);
      }
      const goal = {
        target: state.bounds.center.slice(),
        dist: defaultCamDist(),
        yaw: -0.66,
        pitch: 0.48,
      };
      if (animate) animateCameraTo(goal, 420);
      else {
        state.target = goal.target;
        state.dist = goal.dist;
        state.yaw = goal.yaw;
        state.pitch = goal.pitch;
        requestRender();
      }
    }

    function setTopView(animate = true) {
      state.focusedOnGap = false;
      animateCameraTo({
        target: state.bounds.center.slice(),
        dist: state.bounds.radius * 2.8,
        yaw: 0,
        pitch: 1.51,
      }, animate ? 400 : 1);
    }

    function setFrontView(animate = true) {
      state.focusedOnGap = false;
      animateCameraTo({
        target: state.bounds.center.slice(),
        dist: state.bounds.radius * 2.75,
        yaw: 0,
        pitch: 0.08,
      }, animate ? 400 : 1);
    }

    function setAutoRotate(enabled) {
      state.autoRotate = !!enabled;
      if (state.autoRotate && !state.autoRotateRaf) {
        state.lastTick = performance.now();
        const loop = (now) => {
          if (!state.autoRotate) {
            state.autoRotateRaf = 0;
            return;
          }
          const dt = Math.min(0.1, (now - state.lastTick) / 1000);
          state.lastTick = now;
          state.yaw += dt * 0.38;
          renderNow();
          state.autoRotateRaf = window.requestAnimationFrame(loop);
        };
        state.autoRotateRaf = window.requestAnimationFrame(loop);
      }
      return state.autoRotate;
    }

    function setTheme(theme) {
      state.theme = theme === "dark" ? "dark" : "light";
      buildThreeSceneForPane(primaryPane, state.primarySpec);
      if (state.splitMode && secondaryPane) {
        buildThreeSceneForPane(secondaryPane, state.secondarySpec);
      }
      requestRender();
    }

    function setSelectedFracture(fid) {
      if (state.selectedFractureId === fid) return;
      state.selectedFractureId = fid;
      buildThreeSceneForPane(primaryPane, state.primarySpec);
      if (state.splitMode && secondaryPane) {
        buildThreeSceneForPane(secondaryPane, state.secondarySpec);
      }
      requestRender();
    }

    function pickFractureAt(pane, clientX, clientY) {
      if (!pane || !pane.canvas) return null;
      const rect = pane.canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;

      if (pane.useWebGL && THREE && pane.camera && pane.pickMeshes.length) {
        const ndc = new THREE.Vector2(
          (x / Math.max(1, rect.width)) * 2 - 1,
          -(y / Math.max(1, rect.height)) * 2 + 1
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(ndc, pane.camera);
        const hits = raycaster.intersectObjects(pane.pickMeshes, false);
        if (hits.length > 0 && hits[0].object && hits[0].object.userData) {
          return hits[0].object.userData.fractureId || null;
        }
      }
      return null;
    }

    function attachPointerControls(pane) {
      if (!pane || !pane.canvas) return;
      const canvas = pane.canvas;
      const activePointers = new Map();
      let dragStart = null;
      let pinchStartDist = 0;
      let pinchStartCamDist = 0;
      let pinchStartMid = null;
      let pinchStartTarget = null;
      let lastTapTime = 0;
      let lastTapPos = null;

      function beginSingleDrag(pt, button, shift, alreadyMoved) {
        dragStart = {
          x: pt.x,
          y: pt.y,
          moved: !!alreadyMoved,
          button: button || 0,
          shift: !!shift,
          yaw: state.yaw,
          pitch: state.pitch,
          target: state.target.slice(),
        };
      }

      function beginTwoFingerGesture() {
        const pts = Array.from(activePointers.values());
        if (pts.length < 2) return;
        pinchStartDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
        pinchStartCamDist = state.dist;
        pinchStartMid = { x: (pts[0].x + pts[1].x) * 0.5, y: (pts[0].y + pts[1].y) * 0.5 };
        pinchStartTarget = state.target.slice();
        dragStart = null;
      }

      canvas.addEventListener("pointerdown", (e) => {
        if (state.anim) {
          window.cancelAnimationFrame(state.anim);
          state.anim = null;
        }
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (activePointers.size === 1) {
          beginSingleDrag({ x: e.clientX, y: e.clientY }, e.button, e.shiftKey, false);
        } else if (activePointers.size === 2) {
          beginTwoFingerGesture();
        }
      });

      canvas.addEventListener("pointermove", (e) => {
        if (!activePointers.has(e.pointerId)) {
          if (e.pointerType !== "touch") {
            const fid = pickFractureAt(pane, e.clientX, e.clientY);
            if (fid !== state.hoverFractureId) {
              state.hoverFractureId = fid;
              canvas.style.cursor = fid ? "pointer" : "grab";
              if (state.onHoverFracture) state.onHoverFracture(fid);
            }
          }
          return;
        }

        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (activePointers.size >= 2) {
          const pts = Array.from(activePointers.values());
          const curDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
          const curMid = { x: (pts[0].x + pts[1].x) * 0.5, y: (pts[0].y + pts[1].y) * 0.5 };
          if (pinchStartDist > 8 && curDist > 8) {
            state.dist = clamp(pinchStartCamDist * (pinchStartDist / curDist), 0.02, state.bounds.radius * 22);
          }
          if (pinchStartMid && pinchStartTarget) {
            const dx = curMid.x - pinchStartMid.x;
            const dy = curMid.y - pinchStartMid.y;
            const panScale = state.dist * 0.0019;
            const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
            state.target[0] = pinchStartTarget[0] - dx * cy * panScale;
            state.target[1] = pinchStartTarget[1] - dx * sy * panScale;
            state.target[2] = pinchStartTarget[2] + dy * panScale;
          }
          requestRender();
          return;
        }

        if (!dragStart) return;
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;
        const moveThresh = e.pointerType === "touch" ? 7 : 4;
        if (Math.hypot(dx, dy) > moveThresh) dragStart.moved = true;

        if (dragStart.button === 2 || dragStart.shift) {
          const panScale = state.dist * 0.0018;
          const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
          state.target[0] = dragStart.target[0] - dx * cy * panScale;
          state.target[1] = dragStart.target[1] - dx * sy * panScale;
          state.target[2] = dragStart.target[2] + dy * panScale;
        } else {
          const rotSpeed = e.pointerType === "touch" ? 0.0082 : 0.0072;
          state.yaw = dragStart.yaw - dx * rotSpeed;
          state.pitch = clamp(dragStart.pitch + dy * (rotSpeed * 0.92), -1.48, 1.52);
        }
        requestRender();
      });

      const endPointer = (e) => {
        const wasSingle = activePointers.size === 1;
        if (activePointers.has(e.pointerId)) activePointers.delete(e.pointerId);

        // 双指抬起一指时无缝重锚单指旋转基准，杜绝镜头跳变
        if (activePointers.size === 1) {
          const rem = Array.from(activePointers.values())[0];
          beginSingleDrag(rem, 0, false, true);
          pinchStartDist = 0;
          pinchStartMid = null;
          pinchStartTarget = null;
          return;
        }

        if (wasSingle && dragStart && !dragStart.moved && e.type === "pointerup" && dragStart.button === 0) {
          const fid = pickFractureAt(pane, e.clientX, e.clientY);
          if (fid && state.onSelectFracture) {
            state.onSelectFracture(fid);
          } else if (e.pointerType === "touch") {
            const now = performance.now();
            if (lastTapPos && now - lastTapTime < 300 && Math.hypot(e.clientX - lastTapPos.x, e.clientY - lastTapPos.y) < 32) {
              fitView(true);
              lastTapTime = 0;
              lastTapPos = null;
            } else {
              lastTapTime = now;
              lastTapPos = { x: e.clientX, y: e.clientY };
            }
          }
        }
        if (activePointers.size === 0) {
          dragStart = null;
          pinchStartDist = 0;
          pinchStartMid = null;
          pinchStartTarget = null;
        }
      };

      canvas.addEventListener("pointerup", endPointer);
      canvas.addEventListener("pointercancel", endPointer);
      canvas.addEventListener("contextmenu", (e) => e.preventDefault());

      canvas.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = Math.exp(clamp(e.deltaY, -160, 160) * 0.0013);
          state.dist = clamp(state.dist * factor, 0.02, state.bounds.radius * 22);
          requestRender();
        },
        { passive: false }
      );
    }

    attachPointerControls(primaryPane);
    if (secondaryPane) attachPointerControls(secondaryPane);
    window.addEventListener("resize", resize);
    if (typeof window.ResizeObserver === "function" && primaryPane.canvas && primaryPane.canvas.parentElement) {
      const stageWrap = primaryPane.canvas.closest(".vp-stage-wrap") || primaryPane.canvas.parentElement;
      const ro = new window.ResizeObserver(() => resize());
      ro.observe(stageWrap);
    }

    function screenshotDataURL() {
      renderNow();
      // Composite WebGL transparent canvas onto solid theme background for clean PNG export
      const out = document.createElement("canvas");
      out.width = primaryCanvas.width;
      out.height = primaryCanvas.height;
      const ctx = out.getContext("2d");
      const pal = PALETTE[state.theme] || PALETTE.light;
      ctx.fillStyle = pal.bgCss;
      ctx.fillRect(0, 0, out.width, out.height);
      ctx.drawImage(primaryCanvas, 0, 0);
      return out.toDataURL("image/png");
    }

    function getCameraState() {
      return {
        yaw: Number(state.yaw.toFixed(4)),
        pitch: Number(state.pitch.toFixed(4)),
        dist: Number(state.dist.toFixed(4)),
        target: state.target.map((v) => Number(v.toFixed(4))),
        focusedOnGap: state.focusedOnGap,
        splitMode: state.splitMode,
        webglActive: state.webglActive,
      };
    }

    return {
      setScenes,
      resize,
      fit: fitView,
      topView: setTopView,
      frontView: setFrontView,
      focusGap,
      setAutoRotate,
      setTheme,
      setSelectedFracture,
      screenshot: screenshotDataURL,
      getCameraState,
      isWebGL: () => state.webglActive,
    };
  }

  return { create };
})();
