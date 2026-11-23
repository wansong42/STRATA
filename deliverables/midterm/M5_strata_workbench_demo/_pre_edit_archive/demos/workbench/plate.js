/* W162: revision_c plate_c.js 原样承载（零改动）。数据源已从沙盒内存改为 GET /api/scene：
   core_api.js 只把服务给定的 mesh3d 三角面还原为边界环、scatter3d 点云直接作为显示点，
   不合成几何、不重投影。 */
/* =========================================================================
   STRATA · 纸本仪器版三维证据渲染器（候选 C 专用 · 浅色测绘语言）
   纯 canvas2D 软件渲染，离线零依赖。与 plate3d.js 同一投影/拾取数学，
   视觉重绘：裱版纸面 + 主/次发丝网格 + 矿物色线稿 + 方形纸签，无辉光无渐变。
   交互：左键旋转 / 右键或 Shift 平移 / 滚轮缩放 / 单击拾取 / 悬停高亮
   ========================================================================= */
"use strict";
window.PlateC = (function () {
  const TAU = Math.PI * 2;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  const PAPER = "#fdfcf9";
  const INK = "36,40,34";
  /* 淡彩测绘：语义色一律降饱和，图版不出现高纯度色块 */
  const C = {
    /* 主设守则：图版只用「墨 + 矿绿」两族。语义色降到接近墨的中性，
       靠线宽/虚实/点形区分状态，不靠色相。 */
    surface: "#2f6350",
    pending: "#866a37",
    conflict: "#93483a",
    unknown: "#6f6b60",
    fragment: "#8b877c",
    point: "#403e37",
    relYes: "#3d6b52",
    relNo: "#9b4433",
    relUnknown: "#6f6b60",
    protect: "#7c786c",
    accent: "#245c47",
    grid: "rgba(36,40,34,",
    label: "#232720",
    chip: "#fdfcf9",
    chipLine: "#d4d5cc",
  };
  /* 面的定值淡彩。手选而不是运行时混色：
     运行时 paperTint 会因色相明度差算出偏淡的结果（实测三个面都接近纸白，
     洞与碎片又都是浅灰，三者糊成一片）。这里直接给确定的淡彩，可核对。 */
  /* 同色系的明度微调：用于"凹陷"这类需要比基底略深一档的地方。 */
  function darken(hex, k) {
    const n = parseInt(hex.slice(1), 16);
    const f = (v) => Math.max(0, Math.round(v * (1 - k)));
    return "rgb(" + f((n >> 16) & 255) + "," + f((n >> 8) & 255) + "," + f(n & 255) + ")";
  }

  /* 面的定值淡彩：与纸底（#fcfbf8）之间保持"看得出、不刺眼"的差距。
     三个值都调到比纸底只暗 18–30 级 —— 再深就读作色块，
     再淡就与纸底分不开（实测过的两处失败边界）。 */
  const PASTEL = {
    reviewed: "#e2ece5",   /* 已复核：淡薄荷 */
    pending:  "#f0eadb",   /* 待复核：淡砂 */
    conflict: "#efe1da",   /* 冲突：淡陶 */
    unknown:  "#e9e8e0",
    accent:   "#d9e7de",   /* 选中：略深一档的薄荷 */
  };

  const MONO = '"Cascadia Mono","SF Mono",Consolas,monospace';
  const SANS = '"MiSans","HarmonyOS Sans SC","Noto Sans SC","Segoe UI",system-ui,sans-serif';

  function centroid(ring) {
    const n = ring.length || 1;
    return [0, 1, 2].map((d) => ring.reduce((s, p) => s + p[d], 0) / n);
  }
  function normalOf(ring) {
    if (ring.length < 3) return [0, 0, 1];
    const a = ring[0], b = ring[1], c = ring[2];
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const L = Math.hypot(...n) || 1;
    return n.map((x) => x / L);
  }
  function boundsOf(points) {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (const p of points) for (let d = 0; d < 3; d++) { min[d] = Math.min(min[d], p[d]); max[d] = Math.max(max[d], p[d]); }
    if (!points.length) { min.fill(-1); max.fill(1); }
    return { min, max, center: centroid([min, max]), radius: Math.max(1, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2) };
  }
  /* 投影点既可能是 [x,y]（世界坐标）也可能是 {x,y}（已投影）。
     原实现只按数组解构，传投影点时会抛 TypeError，导致悬停/拾取整条链路失效。 */
  const px = (q) => (Array.isArray(q) ? q[0] : q.x);
  const py = (q) => (Array.isArray(q) ? q[1] : q.y);
  function pointInRing(x, y, ring) {
    if (!Array.isArray(ring) || ring.length < 3) return false;
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = px(ring[i]), yi = py(ring[i]);
      const xj = px(ring[j]), yj = py(ring[j]);
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
    }
    return inside;
  }
  function pointInFace(x, y, face) {
    if (!pointInRing(x, y, face.outer)) return false;
    return !(face.holes || []).some((h) => pointInRing(x, y, h));
  }
  function niceScale(raw) {
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (raw <= m * pow) return m * pow;
    return 10 * pow;
  }
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${clamp(a, 0, 1).toFixed(3)})`;
  }
  /* 把状态色以 t 的比例混进纸色，得到不透明的浅纸色。
     半透明填充在多个面重叠时会累加成浑色块（实图上是灰蒙蒙一片）；
     不透明浅色才会像"叠起来的纸"，前层正常遮住后层。 */
  /* 颜料模型：把状态色归一成"同一明度的低饱和墨"，再按 inkAmount 混进纸色。
     直接混纯色会让绿/棕/红三种面呈现完全不同的视觉重量（实测：绿过重、棕近乎消失）；
     归一后三个面权重一致，才像同一套矿物颜料。 */
  function paperTint(hex, inkAmount) {
    const n = parseInt(hex.slice(1), 16);
    let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const l0 = (mx + mn) / 2 / 255;
    // 归一：饱和度收到 0.30，明度收到 0.52
    const S = 0.34, L = 0.50;
    let h = 0;
    if (mx !== mn) {
      const d = mx - mn;
      if (mx === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
      else if (mx === g) h = ((b - r) / d + 2) / 6;
      else h = ((r - g) / d + 4) / 6;
    }
    const q = L < 0.5 ? L * (1 + S) : L + S - L * S;
    const pch = 2 * L - q;
    const hue2 = (t) => {
      t = (t + 1) % 1;
      if (t < 1 / 6) return pch + (q - pch) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return pch + (q - pch) * (2 / 3 - t) * 6;
      return pch;
    };
    const nr = Math.round(hue2(h + 1 / 3) * 255);
    const ng = Math.round(hue2(h) * 255);
    const nb = Math.round(hue2(h - 1 / 3) * 255);
    const pr = 253, pg = 252, pb = 248;
    const mix = (pp, cc) => Math.round(pp * (1 - inkAmount) + cc * inkAmount);
    void l0;
    return "rgb(" + mix(pr, nr) + "," + mix(pg, ng) + "," + mix(pb, nb) + ")";
  }
  function reviewColor(review) {
    return review === "conflict" ? C.conflict : review === "pending" ? C.pending : review === "unknown" ? C.unknown : C.surface;
  }

  function create(canvas, opts = {}) {
    const ctx = canvas.getContext("2d");
    const st = {
      project: null, selected: null, hover: null, review: {}, previewIds: new Set(),
      layers: { surfaces: true, fragments: true, points: true, relations: true, protected: false, grid: false },
      yaw: -0.62, pitch: 0.62, dist: 0, pan: [0, 0], target: [0, 0, 0], radius: 3,
      w: 0, h: 0, dpr: 1, focal: 0, raf: 0, fitRaf: 0,
      onPick: opts.onPick || null, onHover: opts.onHover || null,
      onFurniture: opts.onFurniture || null,
      drawables: [], bounds: null, relCount: 0, pointCount: 0, auto: true, _scSig: "",
    };

    function resize() {
      const r = canvas.getBoundingClientRect();
      st.dpr = Math.min(2, window.devicePixelRatio || 1);
      st.w = Math.max(1, Math.round(r.width)); st.h = Math.max(1, Math.round(r.height));
      canvas.width = Math.round(st.w * st.dpr); canvas.height = Math.round(st.h * st.dpr);
      st.focal = Math.min(st.w, st.h) * 1.55;
      /* 自动取景状态下，随容器尺寸重算 pan/dist，保证任何窗口都居中不贴边 */
      if (st.auto && st.evidence && st.evidence.length) {
        const fr = frame(st.yaw, st.pitch);
        if (fr) { st.dist = fr.dist; st.pan = fr.pan; }
      }
      requestRender();
    }
    function setProject(project) {
      st.project = project;
      st.relCount = ((project && project.objects) || []).filter((o) => o.kind === "relation").length;
      st.pointCount = ((project && project.points) || []).length;
      const all = ((project && project.points) || []).slice();
      (project?.objects || []).forEach((o) => {
        /* 保护区只是地面范围：纳入取景会把证据压小 */
        if (o.kind === "protected") return;
        if (o.exterior) all.push(...o.exterior);
        if (o.triangle) all.push(...o.triangle);
        if (o.center) all.push(o.center);
      });
      st.bounds = boundsOf(all);
      st.target = st.bounds.center.slice();
      st.radius = st.bounds.radius;
      st.evidence = all;
      st.auto = true;
      /* 首屏即为空 dist：直接按投影足迹定机位，别等用户点"重置视角" */
      if (!st.dist) {
        const fr = frame(st.yaw, st.pitch);
        if (fr) { st.dist = fr.dist; st.pan = fr.pan; }
        else st.dist = st.radius * 2.72;
      }
      requestRender();
    }

    /* 回到标准机位：带 320ms 缓动，交给自动取景重新计算。 */
    function fit() {
      if (!st.bounds) return;
      if (st.fitRaf) cancelAnimationFrame(st.fitRaf);
      st.auto = true;
      const from = { yaw: st.yaw, pitch: st.pitch, dist: st.dist, pan: st.pan.slice(), target: st.target.slice() };
      const fr = frame(-0.62, 0.62);
      const to = fr
        ? { yaw: -0.62, pitch: 0.62, dist: fr.dist, pan: fr.pan, target: st.bounds.center.slice() }
        : { yaw: -0.62, pitch: 0.62, dist: st.radius * 2.72, pan: [0, 0], target: st.bounds.center.slice() };
      const t0 = performance.now();
      const step = () => {
        const t = clamp((performance.now() - t0) / 320, 0, 1);
        const e = 1 - Math.pow(1 - t, 3);
        st.yaw = from.yaw + (to.yaw - from.yaw) * e;
        st.pitch = from.pitch + (to.pitch - from.pitch) * e;
        st.dist = from.dist + (to.dist - from.dist) * e;
        st.pan = [from.pan[0] + (to.pan[0] - from.pan[0]) * e, from.pan[1] + (to.pan[1] - from.pan[1]) * e];
        st.target = [from.target[0] + (to.target[0] - from.target[0]) * e, from.target[1] + (to.target[1] - from.target[1]) * e, from.target[2] + (to.target[2] - from.target[2]) * e];
        render();
        st.fitRaf = t < 1 ? requestAnimationFrame(step) : 0;
      };
      st.fitRaf = requestAnimationFrame(step);
    }

    /* 真实投影下的屏幕包围盒。frame() 用它做迭代校正：
       解析估计假设"统一缩放"，但透视会把近处顶点放大，
       于是实际足迹比估计更宽 —— 这正是证据贴左边缘的原因。 */
    function projectedBounds(pts, dist, yaw, pitch, pan) {
      const p0 = pan ? pan[0] : 0, p1 = pan ? pan[1] : 0;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const tx = st.target[0], ty = st.target[1], tz = st.target[2];
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of pts) {
        const x = p[0] - tx, y = p[1] - ty, z = p[2] - tz;
        const xr = x * cy - y * sy, yr = x * sy + y * cy;
        const depth = Math.max(0.2, dist - (yr * sp + z * cp));
        const sc = st.focal / depth;
        const sx = st.w / 2 + p0 + xr * sc;
        const sy2 = st.h / 2 + p1 - (yr * cp - z * sp) * sc;
        if (sx < x0) x0 = sx; if (sx > x1) x1 = sx;
        if (sy2 < y0) y0 = sy2; if (sy2 > y1) y1 = sy2;
      }
      return { x0, x1, y0, y1 };
    }

    /* 版面安全区：题注、工具、图例和比例尺都有固定位置，证据只占领中央。
       图例与比例尺已移到 DOM 图例栏，画布只需给右下角的三轴留净空。
       与按整幅画布百分比缩放相比，这种构图在窄长视口也不会把家具顶进证据里。 */
    function composition() {
      const compact = st.w < 620;
      const left = compact ? 26 : 52;
      const right = compact ? 70 : 86;
      const top = compact ? 74 : 88;
      const bottom = compact ? 92 : 112;
      const w = Math.max(120, st.w - left - right);
      const h = Math.max(120, st.h - top - bottom);
      return { cx: left + w / 2, cy: top + h / 2, w, h, fill: compact ? 0.92 : 0.95 };
    }

    /* 光学取景：解析估初值 → 真实投影迭代校正 → 对准安全区中心。 */
    function frame(yaw, pitch) {
      const pts = st.evidence && st.evidence.length ? st.evidence : null;
      if (!pts) return null;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const tx = st.target[0], ty = st.target[1], tz = st.target[2];
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of pts) {
        const x = p[0] - tx, y = p[1] - ty, z = p[2] - tz;
        const xr = x * cy - y * sy, yr = x * sy + y * cy;
        const vx = xr, vy = yr * cp - z * sp;
        if (vx < x0) x0 = vx; if (vx > x1) x1 = vx;
        if (vy < y0) y0 = vy; if (vy > y1) y1 = vy;
      }
      const dx = Math.max(1e-6, x1 - x0), dy = Math.max(1e-6, y1 - y0);
      const comp = composition();
      let dist = clamp(st.focal * Math.max(dx / (comp.w * comp.fill), dy / (comp.h * comp.fill)),
                       st.radius * 1.2, st.radius * 16);

      for (let i = 0; i < 4; i++) {
        const bb = projectedBounds(pts, dist, yaw, pitch);
        const k = Math.max((bb.x1 - bb.x0) / (comp.w * comp.fill), (bb.y1 - bb.y0) / (comp.h * comp.fill));
        if (!isFinite(k) || k <= 0) break;
        if (Math.abs(k - 1) < 0.015) break;
        dist = clamp(dist * k, st.radius * 1.2, st.radius * 24);
      }

      const bb = projectedBounds(pts, dist, yaw, pitch);
      return {
        dist,
        pan: [comp.cx - (bb.x0 + bb.x1) / 2, comp.cy - (bb.y0 + bb.y1) / 2],
      };
    }

    function view(p) {
      const x = p[0] - st.target[0], y = p[1] - st.target[1], z = p[2] - st.target[2];
      const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw);
      const x1 = x * cy - y * sy, y1 = x * sy + y * cy, z1 = z;
      const cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
      return [x1, y1 * cp - z1 * sp, y1 * sp + z1 * cp];
    }
    function proj(p) {
      const v = view(p);
      const depth = st.dist - v[2];
      const s = st.focal / Math.max(0.2, depth);
      return { x: st.w / 2 + st.pan[0] + v[0] * s, y: st.h / 2 + st.pan[1] - v[1] * s, depth, s };
    }
    function fog(depth) {
      const near = st.dist - st.radius, far = st.dist + st.radius;
      return clamp(1 - 0.3 * ((depth - near) / Math.max(0.001, far - near)), 0.74, 1);
    }
    function requestRender() {
      if (st.raf) return;
      st.raf = requestAnimationFrame(() => { st.raf = 0; render(); });
    }

    function collect() {
      const D = [];
      const p = st.project;
      if (!p) return D;
      const byId = new Map(p.objects.map((o) => [o.id, o]));
      if (st.layers.grid) D.push({ type: "grid", rank: 0, depth: st.dist + st.radius * 5 });
      for (const o of p.objects) {
        if (o.kind === "surface" && st.layers.surfaces) {
          const ring = o.exterior || [];
          if (ring.length < 3) continue;
          const pts = ring.map(proj);
          const depth = pts.reduce((s, q) => s + q.depth, 0) / pts.length;
          const n = normalOf(ring);
          const nv = view([n[0] + st.target[0], n[1] + st.target[1], n[2] + st.target[2]]);
          D.push({
            type: "surface", id: o.id, rank: 2, depth, obj: o,
            outer: pts, holes: (o.holes || []).map((h) => h.map(proj)),
            review: st.review[o.id] || o.review || "reviewed",
            facing: Math.abs(nv[2]) / (Math.hypot(...nv) || 1), fog: fog(depth), centroid: proj(centroid(ring)),
          });
        } else if (o.kind === "fragment" && st.layers.fragments) {
          const ring = o.triangle || [];
          if (ring.length < 3) continue;
          const pts = ring.map(proj);
          /* 找出把这个碎片列为成员的父面，稍后把碎片裁进父面轮廓内。
             不裁的话，碎片会从面里伸出去，读作"毛刺"（实图上确认过）。 */
          const parents = p.objects.filter((q) => q.kind === "surface" &&
            (q.member_fragment_ids || []).includes(o.id) && (q.exterior || []).length >= 3);
          const clip = parents.map((q) => q.exterior.map(proj));
          D.push({ type: "fragment", id: o.id, rank: 3, depth: pts.reduce((s, q) => s + q.depth, 0) / 3, outer: pts, clip, fog: fog(pts[0].depth) });
        } else if (o.kind === "protected" && (st.layers.protected || st.selected === o.id)) {
          const c = o.center || [0, 0, 0], s = (o.size || [1, 1, 1]).map((v) => v / 2);
          /* 只取 X/Y 足迹、压到地面上：保护区是"范围"，不是立体笼子。
             按盒子的真实高度投影，矩形会被甩到高处并横穿证据（读作脚手架）。 */
          const gz = st.bounds ? st.bounds.min[2] - st.radius * 0.015 : c[2] - s[2];
          const foot3 = [
            [c[0] - s[0], c[1] - s[1], gz],
            [c[0] + s[0], c[1] - s[1], gz],
            [c[0] + s[0], c[1] + s[1], gz],
            [c[0] - s[0], c[1] + s[1], gz],
          ];
          const pf = foot3.map(proj);
          const dmid = pf.reduce((a, q) => a + q.depth, 0) / 4;
          D.push({ type: "protect", id: o.id, rank: 1, depth: dmid, foot: pf, fog: fog(dmid) });
        } else if (o.kind === "relation" && st.layers.relations) {
          const a = byId.get(o.from), b = byId.get(o.to);
          if (!a || !b || !a.exterior || !b.exterior) continue;
          const pa = proj(centroid(a.exterior)), pb = proj(centroid(b.exterior));
          D.push({ type: "relation", id: o.id, rank: 4, depth: (pa.depth + pb.depth) / 2, a: pa, b: pb, state: o.state });
        }
      }
      if (st.layers.points) {
        const sel = st.selected ? byId.get(st.selected) : null;
        const selIdx = new Set(sel?.support_indices || []);
        (p.points || []).forEach((pt, i) => {
          const q = proj(pt);
          D.push({ type: "point", rank: 5, depth: q.depth, q, highlighted: selIdx.has(i), fog: fog(q.depth) });
        });
      }
      /* 先按图层 rank（底→顶），层内再按深度远→近。
         原先只按平均深度排序，碎片会被大面片随机盖住（时隐时现）。 */
      D.sort((a, b) => (a.rank - b.rank) || (b.depth - a.depth));
      return D;
    }

    /* 底面基准：一块贴着数据足迹的"图纸垫"，3×3 极淡格线 + 一圈实边。
       旧版把大网铺满全图，斜线横穿画面像划痕；这里收窄到数据足迹附近，
       内部线降到近乎纹理，只有外框读得出是基准面。 */
    function drawGrid() {
      const b = st.bounds; if (!b) return;
      const z = b.min[2] - st.radius * 0.06;
      const step = niceScale(st.radius * 0.62);
      const cx = b.center[0], cy = b.center[1];
      const n = 2;
      const x0 = cx - step * n, x1 = cx + step * n;
      const y0 = cy - step * n, y1 = cy + step * n;
      ctx.lineWidth = 1;
      const line = (a, c, alpha) => {
        ctx.strokeStyle = C.grid + (alpha * fog((a.depth + c.depth) / 2)).toFixed(3) + ")";
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); ctx.stroke();
      };
      /* 内部格线：只留过中心的两条，其余不画，避免"划痕感" */
      for (let i = -n; i <= n; i++) {
        const x = cx + i * step, y = cy + i * step;
        if (i !== 0) continue;
        line(proj([x, y0, z]), proj([x, y1, z]), 0.05);
        line(proj([x0, y, z]), proj([x1, y, z]), 0.05);
      }
      /* 外框：一圈实边，读作基准面边界 */
      const cc = [proj([x0, y0, z]), proj([x1, y0, z]), proj([x1, y1, z]), proj([x0, y1, z])];
      ctx.strokeStyle = C.grid + (0.11 * fog(cc[0].depth)).toFixed(3) + ")";
      ctx.beginPath();
      cc.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.closePath(); ctx.stroke();
      /* 四条边的中点刻度：制图基准的记号，也让框不像随便画的四边形 */
      const mids = [
        [proj([cx, y0, z]), proj([cx, y1, z])],
        [proj([x0, cy, z]), proj([x1, cy, z])],
      ];
      for (const pair of mids) {
        for (const q of [pair[0], pair[1]]) {
          ctx.fillStyle = C.grid + (0.2 * fog(q.depth)).toFixed(3) + ")";
          ctx.fillRect(q.x - 1.5, q.y - 1.5, 3, 3);
        }
      }
    }

    function ringPath(outer, holes) {
      const path = new Path2D();
      for (const ring of [outer, ...holes]) {
        ring.forEach((q, i) => (i ? path.lineTo(q.x, q.y) : path.moveTo(q.x, q.y)));
        path.closePath();
      }
      return path;
    }

    function drawSurface(d) {
      const sel = d.id === st.selected, hov = d.id === st.hover;
      const col = sel ? C.accent : reviewColor(d.review);
      const key = sel ? "accent" : (d.review === "pending" ? "pending" : d.review === "conflict" ? "conflict" : d.review === "unknown" ? "unknown" : "reviewed");
      /* 朝向只影响极轻的明度，不能像 3D 渐变一样抢走线稿的主次。 */
      const fill = darken(PASTEL[key] || PASTEL.reviewed, 0.012 + (1 - clamp(d.facing, 0, 1)) * 0.024);

      const outer = new Path2D();
      d.outer.forEach((q, i) => (i ? outer.lineTo(q.x, q.y) : outer.moveTo(q.x, q.y)));
      outer.closePath();

      /* evenodd 路径：洞的位置不涂色，纸底自然透出 —— 这就是"窗口"。
         洞不再补任何填充或描边：一旦补上，它就变成"贴在面上的一张卡片"。 */
      const face = new Path2D();
      d.outer.forEach((q, i) => (i ? face.lineTo(q.x, q.y) : face.moveTo(q.x, q.y)));
      face.closePath();
      for (const h of (d.holes || [])) {
        h.forEach((q, i) => (i ? face.lineTo(q.x, q.y) : face.moveTo(q.x, q.y)));
        face.closePath();
      }
      ctx.fillStyle = fill;
      ctx.fill(face, "evenodd");

      /* 外轮廓 + 洞的内环，用同一支笔。
         多边形带洞时，洞的边界必须也用该多边形自己的轮廓线描一遍。
         少了这一笔，洞里透出的纸底就成了"贴在面上的一张白纸"；
         有了它，同一个白块才读作"面上挖的洞"。 */
      ctx.lineWidth = sel ? 1.8 : hov ? 1.45 : 1.1;
      ctx.strokeStyle = hexA(col, (sel ? 0.96 : hov ? 0.9 : 0.8) * d.fog);
      ctx.stroke(outer);
      if (d.holes && d.holes.length) {
        /* 洞的内环必须比外轮廓细得多、淡得多。
           与外轮廓同粗时，洞里那块纸底会被读成"贴上去的白卡片"；
           细淡的内环才读作"面上挖的窗口"。 */
        ctx.save();
        ctx.lineWidth = 0.85;
        ctx.strokeStyle = hexA(col, 0.45 * d.fog);
        for (const h of d.holes) {
          const holeRing = new Path2D();
          h.forEach((q, i) => (i ? holeRing.lineTo(q.x, q.y) : holeRing.moveTo(q.x, q.y)));
          holeRing.closePath();
          ctx.stroke(holeRing);
        }
        ctx.restore();
      }

      if (st.previewIds.has(d.id)) {
        ctx.save();
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = hexA(C.accent, 0.95);
        ctx.stroke(outer);
        ctx.restore();
      }
      ctx.fillStyle = hexA(col, 0.85 * d.fog);
      for (const q of d.outer) { ctx.beginPath(); ctx.arc(q.x, q.y, sel ? 1.9 : 1.25, 0, TAU); ctx.fill(); }
    }

    /* 碎片：面内的细分边界，只描细线、不填色。
       面已有淡彩，碎片若再填浅色就成了"面上的补丁"；
       只描线才读作"这个面由几块拼成"，这是制图里表达内部分界的写法。 */
    function drawFragment(d) {
      const sel = d.id === st.selected;
      const path = new Path2D();
      d.outer.forEach((q, i) => (i ? path.lineTo(q.x, q.y) : path.moveTo(q.x, q.y)));
      path.closePath();

      /* 裁进父面：碎片是"这个面由几块拼成"的内部分界线，
         不裁就会从面里伸出去，读作毛刺。 */
      ctx.save();
      if (d.clip && d.clip.length) {
        const clipPath = new Path2D();
        for (const ring of d.clip) {
          ring.forEach((q, i) => (i ? clipPath.lineTo(q.x, q.y) : clipPath.moveTo(q.x, q.y)));
          clipPath.closePath();
        }
        ctx.clip(clipPath);
      }
      /* 极淡中性墨洗：碎片是"这个面由几块拼成"的内部分块，
         只有线会像划痕，填一层几乎看不出的墨洗才有"块"的体积感，
         且因为是中性灰，不会与任何面色打架。 */
      ctx.fillStyle = "rgba(31,30,25,.035)";
      ctx.fill(path);
      ctx.lineWidth = sel ? 1.3 : 0.95;
      ctx.strokeStyle = sel ? hexA(C.accent, 0.95) : "rgba(74,70,58,.42)";
      ctx.stroke(path);
      ctx.restore();
    }

    /* 保护区：地面上的一圈细虚线，不填色。
       填充会盖住证据；粗虚线会读成脚手架。细虚线 + 直角才是"圈定范围"。 */
    function drawProtect(d) {
      const sel = d.id === st.selected;
      const foot = d.foot;
      if (!foot || foot.length < 4) return;
      const path = new Path2D();
      foot.forEach((q, i) => (i ? path.lineTo(q.x, q.y) : path.moveTo(q.x, q.y)));
      path.closePath();
      ctx.save();
      ctx.setLineDash(sel ? [] : [4, 5]);
      ctx.lineWidth = sel ? 1.3 : 0.9;
      ctx.strokeStyle = sel ? hexA(C.accent, 0.9) : "rgba(118,112,96,.28)";
      ctx.stroke(path);
      ctx.restore();
    }

    function drawRelation(d) {
      const sel = d.id === st.selected;
      const emph = sel || d.id === st.hover;
      const dense = st.relCount > 64;
      const col = d.state === "unknown" ? C.relUnknown : d.state === "yes" ? C.relYes : C.relNo;
      /* 密集域自适应：unknown 关系退为 faint 发丝（仍可拾取），yes/no 保持中等；选中/悬停始终全强度 */
      const faint = dense && !emph && d.state === "unknown";
      ctx.lineWidth = faint ? 0.8 : sel ? 1.8 : dense && !emph ? 1 : 1.25;
      ctx.setLineDash(d.state === "unknown" ? [2, 5] : []);
      ctx.strokeStyle = hexA(col, faint ? 0.16 : sel ? 0.85 : dense && !emph ? 0.34 : 0.5);
      ctx.beginPath(); ctx.moveTo(d.a.x, d.a.y); ctx.lineTo(d.b.x, d.b.y); ctx.stroke();
      ctx.setLineDash([]);
      if (faint) return;
      const mx = (d.a.x + d.b.x) / 2, my = (d.a.y + d.b.y) / 2;
      /* unknown 已由虚线表达；再多一枚"?"就成了浮在图上的 UI 徽章。
         制图写法：端点空心小圈。 */
      if (d.state === "unknown") {
        ctx.beginPath(); ctx.arc(mx, my, 2.6, 0, TAU);
        ctx.fillStyle = PAPER; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = hexA(col, 0.38); ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(mx, my, 2.2, 0, TAU);
        ctx.fillStyle = hexA(col, dense && !emph ? 0.4 : 0.62); ctx.fill();
      }
    }

    function drawPoint(d) {
      if (d.highlighted) {
        ctx.beginPath(); ctx.arc(d.q.x, d.q.y, 5, 0, TAU);
        ctx.fillStyle = hexA(C.accent, 0.12); ctx.fill();
        ctx.beginPath(); ctx.arc(d.q.x, d.q.y, 2.1, 0, TAU);
        ctx.fillStyle = hexA(C.accent, 0.9); ctx.fill();
        return;
      }
      const dense = st.pointCount > 2000;
      const r = dense ? 1.5 : 2.2;
      ctx.fillStyle = hexA(C.point, (dense ? 0.5 : 0.66) * d.fog);
      ctx.fillRect(d.q.x - r / 2, d.q.y - r / 2, r, r);
    }

    /* 标注：纸签 + 引线。
       定位不取质心（会压在证据上），而取对象投影轮廓的上沿：
       默认贴在轮廓外，空间不足时翻到下方。引线短、只连到轮廓边，
       像制图里的序号标注，而不是浮在图上的一张卡片。 */
    function drawLabel(d) {
      const text = d.id;
      ctx.font = "500 10.5px " + MONO;
      const tw = ctx.measureText(text).width;
      const padX = 8, h = 20;
      const w = tw + padX * 2 + 4;

      /* 轮廓上下沿与顶点横坐标 */
      let top = Infinity, bot = -Infinity, anchorX = d.centroid.x;
      const ring = d.outer && d.outer.length ? d.outer : [d.centroid];
      for (const q of ring) {
        if (q.y < top) { top = q.y; anchorX = q.x; }
        if (q.y > bot) bot = q.y;
      }
      const gap = 9;
      let y, anchorY;
      if (top - gap - h >= 10) { y = top - gap - h; anchorY = top; }
      else if (bot + gap + h <= st.h - 12) { y = bot + gap; anchorY = bot; }
      else { y = clamp(top - gap - h, 10, st.h - h - 12); anchorY = top; }

      let x = anchorX - w / 2;
      x = clamp(x, 12, Math.max(12, st.w - w - 12));

      /* 引线：纸签下沿 → 轮廓顶点。极短、极淡，不穿过证据 */
      const lineX = clamp(anchorX, x + 6, x + w - 6);
      ctx.beginPath();
      if (y < anchorY) { ctx.moveTo(lineX, y + h); ctx.lineTo(lineX, anchorY - 3); }
      else { ctx.moveTo(lineX, y); ctx.lineTo(lineX, anchorY + 3); }
      ctx.strokeStyle = `rgba(${INK},.18)`; ctx.lineWidth = 1; ctx.stroke();

      ctx.fillStyle = C.chip;
      roundRect(x, y, w, h, 4); ctx.fill();
      ctx.strokeStyle = C.chipLine; ctx.lineWidth = 1;
      roundRect(x, y, w, h, 4); ctx.stroke();
      ctx.fillStyle = C.accent; ctx.fillRect(x + 7, y + h / 2 - 1.5, 3, 3);
      ctx.fillStyle = C.label; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(text, x + padX + 5, y + h / 2 + .5);
    }

    /* 坐标三轴：放进右下角净空区，整簇不越过 composition() 的 right/bottom。
       位置与长度是跟着那组安全区反算的，改安全区就要一起改这里。 */
    function drawTriad() {
      const compact = st.w < 620;
      const L = compact ? 24 : 34, LOFF = compact ? 9 : 13;
      const ox = st.w - (compact ? 40 : 54);
      const oy = st.h - (compact ? 58 : 74);
      const axes = [[1, 0, 0, "#6d7168", "X"], [0, 1, 0, "#245c47", "Y"], [0, 0, 1, "#6d7168", "Z"]];

      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.beginPath(); ctx.arc(ox, oy, 2.6, 0, TAU);
      ctx.fillStyle = PAPER; ctx.fill();
      ctx.strokeStyle = `rgba(${INK},.42)`; ctx.lineWidth = 1; ctx.stroke();

      for (const [ax, ay, az, col, name] of axes) {
        const v = view([ax + st.target[0], ay + st.target[1], az + st.target[2]]);
        const dx = v[0], dy = -v[1];
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const ex = ox + ux * L, ey = oy + uy * L;
        ctx.strokeStyle = hexA(col, 0.86); ctx.lineWidth = 1.35;
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.fillStyle = hexA(col, 0.82);
        ctx.fillRect(ex - 1.6, ey - 1.6, 3.2, 3.2);
        const nx = Math.abs(ux) < 0.36 ? 9 : 0;
        ctx.font = "600 " + (compact ? 9.5 : 10.5) + "px " + SANS;
        /* 轴字母是画布上唯一的文字，必须给满不透明：0.92 压到纸底只有 4.1:1 */
        ctx.fillStyle = col;
        ctx.fillText(name, ox + ux * (L + LOFF) + nx, oy + uy * (L + LOFF));
      }
    }

    /* 比例尺：在相邻三个数量级里挑最接近目标长度的 1 / 2 / 5 刻度。
       实际描线交给 DOM 图例栏，画布只报"多少像素 = 多少米"，
       于是比例尺再也不会因为固定范围无命中而横跨半张图。 */
    function scaleInfo() {
      const worldPerPx = st.dist / st.focal;
      /* 窄视口给短尺：图例栏那一行还要放图例，长尺会把它们挤到第二行 */
      const targetPx = st.w < 620 ? 66 : 92;
      const raw = Math.max(1e-9, worldPerPx * targetPx);
      const exp = Math.floor(Math.log10(raw));
      let meters = Math.pow(10, exp);
      let bestDelta = Infinity;
      for (let e = exp - 1; e <= exp + 1; e++) {
        for (const m of [1, 2, 5]) {
          const value = m * Math.pow(10, e);
          const delta = Math.abs(value / worldPerPx - targetPx);
          if (delta < bestDelta) { bestDelta = delta; meters = value; }
        }
      }
      return { px: Math.max(28, Math.round(meters / worldPerPx)), meters, label: formatMeters(meters) };
    }

    function formatMeters(m) {
      const unit = st.project?.unit;
      const suffix = unit === "m" ? " m" : unit === "source_coordinate_units_unverified" ? " u（未核）" : "";
      return (m >= 1 ? m : +m.toFixed(2)) + suffix;
    }
    function roundRect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }

    function render() {
      ctx.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
      ctx.clearRect(0, 0, st.w, st.h);
      ctx.fillStyle = PAPER; ctx.fillRect(0, 0, st.w, st.h);
      if (!st.project) return;

      /* 自动取景只使用中央安全区，覆盖层永远拥有自己的净空。 */
      if (st.auto && st.evidence && st.evidence.length && st.w > 8) {
        const comp = composition();
        for (let i = 0; i < 6; i++) {
          const bb = projectedBounds(st.evidence, st.dist, st.yaw, st.pitch, st.pan);
          const bw = Math.max(1, bb.x1 - bb.x0), bh = Math.max(1, bb.y1 - bb.y0);
          st.pan[0] += comp.cx - (bb.x0 + bb.x1) / 2;
          st.pan[1] += comp.cy - (bb.y0 + bb.y1) / 2;
          const k = Math.max(bw / (comp.w * comp.fill), bh / (comp.h * comp.fill));
          if (!isFinite(k) || k <= 0) break;
          if (Math.abs(k - 1) < 0.01) break;
          st.dist = clamp(st.dist * k, st.radius * 1.2, st.radius * 24);
        }
      }

      st.drawables = collect();
      let labelDrawable = null;
      for (const d of st.drawables) {
        if (d.type === "grid") drawGrid();
        else if (d.type === "surface") { drawSurface(d); if (d.id === st.selected) labelDrawable = d; }
        else if (d.type === "fragment") drawFragment(d);
        else if (d.type === "protect") drawProtect(d);
        else if (d.type === "relation") drawRelation(d);
        else if (d.type === "point") drawPoint(d);
      }
      if (labelDrawable) drawLabel(labelDrawable);
      drawTriad();
      /* 比例尺交给图例栏：只在刻度真的变了才重绘 DOM，避免每帧写样式。
         空态没有项目也没有机位，此时不报数（除零会算出 NaN 宽度）。 */
      if (st.project && st.dist > 0 && st.focal > 0) {
        const si = scaleInfo();
        const ssig = si.px + "|" + si.label;
        if (st.onFurniture && ssig !== st._scSig) { st._scSig = ssig; st.onFurniture(si); }
      }
    }

    function pick(x, y) {
      for (let i = st.drawables.length - 1; i >= 0; i--) {
        const d = st.drawables[i];
        if (d.type === "surface" && pointInFace(x, y, d)) return d.id;
        if (d.type === "fragment" && pointInRing(x, y, d.outer)) return d.id;
        if (d.type === "relation") {
          const dx = d.b.x - d.a.x, dy = d.b.y - d.a.y;
          const L2 = dx * dx + dy * dy || 1;
          const t = clamp(((x - d.a.x) * dx + (y - d.a.y) * dy) / L2, 0, 1);
          if (Math.hypot(x - (d.a.x + dx * t), y - (d.a.y + dy * t)) < 6) return d.id;
        }
        if (d.type === "protect" && pointInRing(x, y, d.foot)) return d.id;
      }
      return null;
    }

    let drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false };
    });
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        st.auto = false;
        if (drag.pan) { st.pan[0] += dx; st.pan[1] += dy; }
        else { st.yaw += dx * 0.006; st.pitch = clamp(st.pitch + dy * 0.005, -1.4, 1.4); }
        drag.x = e.clientX; drag.y = e.clientY;
        requestRender();
      } else {
        const id = pick(x, y);
        if (id !== st.hover) { st.hover = id; canvas.style.cursor = id ? "pointer" : "grab"; if (st.onHover) st.onHover(id); requestRender(); }
      }
    });
    canvas.addEventListener("pointerup", (e) => {
      const wasDrag = drag; drag = null;
      if (wasDrag && !wasDrag.moved && e.button === 0) {
        const r = canvas.getBoundingClientRect();
        if (st.onPick) st.onPick(pick(e.clientX - r.left, e.clientY - r.top));
      }
    });
    canvas.addEventListener("pointerleave", () => { if (st.hover) { st.hover = null; if (st.onHover) st.onHover(null); requestRender(); } });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      st.auto = false;
      st.dist = clamp(st.dist * (e.deltaY > 0 ? 1.1 : 0.9), st.radius * 0.6, st.radius * 12);
      requestRender();
    }, { passive: false });

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    return {
      setProject, fit, requestRender,
      setSelected(id) { st.selected = id; requestRender(); },
      setReview(map) { st.review = map || {}; requestRender(); },
      setHover(id) { st.hover = id; requestRender(); },
      setPreviewIds(ids) { st.previewIds = new Set(ids || []); requestRender(); },
      setLayers(l) { Object.assign(st.layers, l); requestRender(); },
      getLayers() { return { ...st.layers }; },
      screenshot() { render(); return canvas.toDataURL("image/png"); },
      destroy() { ro.disconnect(); if (st.raf) cancelAnimationFrame(st.raf); if (st.fitRaf) cancelAnimationFrame(st.fitRaf); },
      _state: st,
    };
  }
  return { create };
})();