/* =========================================================================
   STRATA · 真实服务数据层（core_api.js）
   W162：把 revision_c 视觉壳从「同步内存沙盒」整体换到「真实 WorkbenchService」。
   边界（硬约束）：
     - 唯一数据源是同源 /api/*；本文件不合成几何、不本地算 content_hash、
       不产出任何「保存成功 / 导出 PASS」回执。
     - 页面状态（S）只由服务响应赋值；每个动作完成后重新 GET /api/state，
       保证 DOM 显示值与同刻服务状态同源同刻。
     - 对外暴露只读 window.__W162_STATE__，字段与 DOM 同源。
   ========================================================================= */
"use strict";
window.SandboxCore = (function () {
  /* 文案原样取自 revision_c 原型 seed.js 的 copy 段；展示态相关文案已随展示态一并移除。 */
  const COPY = {
    new_synthetic: "新建合成项目",
    load_internal: "载入内部 PF02",
    import_json: "导入项目 JSON",
    save: "保存",
    close: "关闭",
    reopen: "重开",
    reopen_label: "重开已保存项目",
    preview: "生成预览（不改项目）",
    commit: "提交新版本",
    cancel_preview: "取消预览",
    undo: "撤销",
    redo: "重做",
    solve: "规则求解",
    learned: "学习模式",
    cancel_job: "取消任务",
    export: "导出并由新进程读回",
    fit: "重置视角",
    shot: "保存当前三维截图",
    retry_save: "重试保存",
    view_log: "查看操作日志",
    dev_chip: "开发竖切 · 非发布版",
    dirty: "有未保存修改",
    clean: "已保存",
    unit_unverified: "单位未核",
    real_quality_undefined: "真实质量 undefined",
    preview_unchanged: "预览未改项目",
    save_failed_title: "保存失败 · 项目未损坏",
    save_failed_code: "PROJECT_STORE_WRITE_ERROR",
    save_failed_reason: "保存被后端拒绝；当前会话可继续编辑。",
    save_failed_rollback: "旧 head 已保留（previous_head_preserved=true），当前会话可继续编辑。",
    empty_title: "项目已关闭",
    empty_hint: "从登记资料或已保存版本打开项目；页面一切读数来自真实服务。",
    boundary_note: "合成域 ≠ 真实露头域；配对增益栏 ≠ 绝对误差水平栏；真实质量不自动定义。"
  };

  /* ui_common.mountChrome 只在 ?chrome=1 时读 SEED；本页不提供展示态，故为空表。 */
  const SEED = { sandbox: { label: "STRATA", round: "", states: [], variants: [] } };

  const listeners = new Set();
  const detailCache = new Map();
  const evidenceCache = new Map();
  const recentRequests = [];
  const logRows = [];
  let revision = 0;
  let lastState = null;
  let geo = { points: [], surfaces: new Map(), protected: new Map(), summary: null };

  const S = {
    project_open: false, project_id: null, project_version: 0, content_hash: "",
    read_only_legacy: false, bundle_id: null, dirty: false, saved_state: null,
    source: "synthetic", source_domain: null, selectedId: null, review: {},
    preview: null, saveError: null, saveReceipt: null, exportReceipt: null,
    library: [], jobs: [], log: logRows, activeJobId: null, project: null,
    catalog: { entries: [] }, scene: null, exportFiles: [], scenario: null,
    evidence_snapshot_id: null, evidence_counts: null, ready: false,
    /* W172 OBS-D：区分「state 尚未读到 / 读取失败 / 真空库」。ready 在三次重试全失败后仍置 true，
       不能单独区分加载中与失败；state_loaded 只在 applyState 成功落地时置 true。 */
    state_loaded: false,
    portable: null, portableStatus: null, portableDetail: null,
    portableReceipt: null, portableError: null,
    /* M5：贯通可辨识性与主动坍缩状态（全部来自同源 /api/identifiability/*，零前端伪造） */
    ident: {
      cases: [], total_cases: 0, site_counts: {}, status_counts: {},
      m2_run_id: null, m3_verdict_sha256: null, not_claimed: [],
      cases_loaded: false, loading_cases: false, cases_error: null,
      selected_cluster_id: null, current_case: null,
      loading_case: false, case_error: null,
      witness_view: "nominal",
      selected_fracture_id: null,
      applying_check: false,
      collapse_state: null,
      check_history: []
    }
  };

  /* ---------------- 小工具 ---------------- */
  function nowIso() { return new Date().toISOString(); }
  function errText(value) { return value === null || value === undefined ? "" : String(value); }

  function logOperation(operation, expected, actual, level) {
    logRows.push({ measured_at: nowIso(), level: level || "ok", operation: operation, expected: expected, actual: actual });
    if (logRows.length > 200) logRows.splice(0, logRows.length - 200);
  }

  function record(path, method, outcome, ms) {
    recentRequests.push({ path: path, method: method, outcome: outcome, ms: ms, at: nowIso() });
    if (recentRequests.length > 80) recentRequests.splice(0, recentRequests.length - 80);
  }

  function ApiError(code, message, path, status) {
    const err = new Error(message || code);
    err.name = "ApiError";
    err.code = code;
    err.path = path;
    err.status = status || 0;
    return err;
  }

  async function api(path, options) {
    const opts = options || {};
    const method = opts.method || "GET";
    const started = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
    let response;
    try {
      response = await fetch(path, {
        method: method,
        cache: "no-store",
        credentials: "same-origin",
        headers: opts.body === undefined ? {} : { "Content-Type": "application/json" },
        body: opts.body === undefined ? undefined : JSON.stringify(opts.body)
      });
    } catch (err) {
      record(path, method, "NETWORK_ERROR");
      logOperation(method + " " + path, "到达本地服务", "NETWORK_ERROR · " + errText(err && err.message), "err");
      throw ApiError("NETWORK_ERROR", errText(err && err.message) || "网络请求失败", path, 0);
    }
    const ms = Math.round(((typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now()) - started);
    record(path, method, response.status, ms);
    logOperation(method + " " + path, "HTTP 200", "HTTP " + response.status + " · " + ms + "ms", response.ok ? "ok" : "err");
    let data = null;
    try {
      data = await response.json();
    } catch (err) {
      throw ApiError("RESPONSE_NOT_JSON", "响应不是合法 JSON", path, response.status);
    }
    if (!response.ok) {
      const info = (data && data.error) || {};
      throw ApiError(info.code || ("HTTP_" + response.status), info.message || ("HTTP " + response.status), path, response.status);
    }
    return data;
  }

  function fail(operation, err) {
    const code = (err && err.code) || "UNEXPECTED_ERROR";
    const message = (err && err.message) || errText(err);
    /* W171 W170C-04：expected 不再硬编码「后端接受」。HTTP>=400 才是后端拒绝；
       NETWORK_ERROR/RESPONSE_NOT_JSON 等非后端拒绝记「请求失败」。英文原文仍留在 actual 详情。 */
    const expected = (err && err.status >= 400) ? "后端拒绝" : "请求失败";
    logOperation(operation, expected, code + " · " + message, "err");
    emit();
    return { ok: false, code: code, message: message, path: err && err.path };
  }

  /* ---------------- 几何：只做形状搬运，不做几何合成 ----------------
     /api/scene 的真实响应是 Plotly 轨迹（mesh3d 三角面 / scatter3d 点线）。
     plate.js 需要「外环 + 洞」的纸面多边形，因此这里只从**服务给定的三角面**
     还原边界环：边只被一个三角引用即为边界。坐标一个不改，顺序由服务数据决定。 */
  function ringsFromMesh(xs, ys, zs, ii, jj, kk) {
    const seen = new Map();
    const verts = [];
    const remap = new Array(xs.length);
    for (let n = 0; n < xs.length; n++) {
      const key = xs[n].toFixed(6) + "|" + ys[n].toFixed(6) + "|" + zs[n].toFixed(6);
      let index = seen.get(key);
      if (index === undefined) {
        index = verts.length;
        seen.set(key, index);
        verts.push([xs[n], ys[n], zs[n]]);
      }
      remap[n] = index;
    }
    const counts = new Map();
    for (let t = 0; t < ii.length; t++) {
      const a = remap[ii[t]], b = remap[jj[t]], c = remap[kk[t]];
      if (a === b || b === c || a === c) continue;
      const pairs = [[a, b], [b, c], [c, a]];
      for (let p = 0; p < pairs.length; p++) {
        const u = pairs[p][0], v = pairs[p][1];
        const key = u < v ? u + ":" + v : v + ":" + u;
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    }
    const neighbors = new Map();
    counts.forEach(function (count, key) {
      if (count !== 1) return;
      const parts = key.split(":");
      const u = Number(parts[0]), v = Number(parts[1]);
      if (!neighbors.has(u)) neighbors.set(u, []);
      if (!neighbors.has(v)) neighbors.set(v, []);
      neighbors.get(u).push(v);
      neighbors.get(v).push(u);
    });
    const visited = new Set();
    const rings = [];
    neighbors.forEach(function (list, start) {
      if (visited.has(start)) return;
      const ring = [];
      let prev = -1, cur = start;
      for (let guard = 0; guard < 200000; guard++) {
        visited.add(cur);
        ring.push(verts[cur]);
        const options = neighbors.get(cur) || [];
        let next = -1;
        for (let i = 0; i < options.length; i++) {
          const cand = options[i];
          if (cand !== prev && !visited.has(cand)) { next = cand; break; }
        }
        if (next < 0) break;
        prev = cur; cur = next;
        if (cur === start) break;
      }
      if (ring.length >= 3) rings.push(ring);
    });
    if (!rings.length) return { exterior: [], holes: [], center: null };
    const areaOf = function (ring) {
      let acc = [0, 0, 0];
      for (let i = 1; i < ring.length - 1; i++) {
        const p = ring[0], q = ring[i], r = ring[i + 1];
        const ux = q[0] - p[0], uy = q[1] - p[1], uz = q[2] - p[2];
        const vx = r[0] - p[0], vy = r[1] - p[1], vz = r[2] - p[2];
        acc[0] += uy * vz - uz * vy;
        acc[1] += uz * vx - ux * vz;
        acc[2] += ux * vy - uy * vx;
      }
      return 0.5 * Math.sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    };
    let exterior = rings[0], best = areaOf(rings[0]);
    const holes = [];
    for (let i = 1; i < rings.length; i++) {
      const area = areaOf(rings[i]);
      if (area > best) { if (exterior) holes.push(exterior); exterior = rings[i]; best = area; }
      else holes.push(rings[i]);
    }
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < verts.length; i++) { cx += verts[i][0]; cy += verts[i][1]; cz += verts[i][2]; }
    const n = Math.max(1, verts.length);
    return { exterior: exterior, holes: holes, center: [cx / n, cy / n, cz / n] };
  }

  function applyScene(payload) {
    const points = [];
    const surfaces = new Map();
    const protectedRings = new Map();
    const traces = (payload && payload.traces) || [];
    for (let t = 0; t < traces.length; t++) {
      const trace = traces[t];
      const meta = trace.meta || {};
      if (trace.type === "mesh3d" && meta.layer === "surfaces" && meta.object_id) {
        const rings = ringsFromMesh(trace.x || [], trace.y || [], trace.z || [], trace.i || [], trace.j || [], trace.k || []);
        surfaces.set(meta.object_id, rings);
      } else if (trace.type === "scatter3d" && meta.layer === "points") {
        const x = trace.x || [], y = trace.y || [], z = trace.z || [];
        for (let i = 0; i < x.length; i++) {
          if (z[i] === null) continue;
          points.push([x[i], y[i], z[i]]);
        }
      } else if (trace.type === "scatter3d" && meta.layer === "protected" && meta.object_id) {
        const x = trace.x || [], y = trace.y || [], z = trace.z || [];
        const ring = [];
        for (let i = 0; i < x.length; i++) if (z[i] !== null) ring.push([x[i], y[i], z[i]]);
        if (ring.length >= 3) protectedRings.set(meta.object_id, ring);
      }
    }
    return { points: points, surfaces: surfaces, protected: protectedRings, summary: payload };
  }

  function reviewForSurface(status) {
    if (status === "human_reviewed") return "reviewed";
    if (status === "conflict") return "conflict";
    return "pending";
  }

  function buildProject(state, geometry) {
    if (!state || !state.project_open) return null;
    const counts = state.counts || {};
    const objects = [];
    const items = state.objects || [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === "surface") {
        const shape = geometry.surfaces.get(item.id) || { exterior: [], holes: [], center: null };
        objects.push({
          kind: "surface", id: item.id, label: item.label, version: item.version,
          interpretation_status: item.status, status: item.status,
          review: reviewForSurface(item.status), provenance: item.provenance,
          member_fragment_ids: item.members || [], geometry: item.geometry,
          exterior: shape.exterior, holes: shape.holes, center: shape.center
        });
      } else if (item.kind === "fragment") {
        objects.push({
          kind: "fragment", id: item.id, label: item.label, version: item.version,
          observation_id: item.observation_id, support_indices: item.support_indices || [],
          geometry: item.geometry, review: "reviewed"
        });
      } else if (item.kind === "relation") {
        objects.push({
          kind: "relation", id: item.id, label: item.label, version: item.version,
          from: item.src_id, to: item.dst_id, state: item.state,
          unknown_reason: item.unknown_reason, relation_type: item.relation_type,
          review: item.state === "unknown" ? "unknown" : "reviewed"
        });
      } else if (item.kind === "protected") {
        const ring = geometry.protected.get(item.id) || [];
        let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
        for (let r = 0; r < ring.length; r++) {
          minx = Math.min(minx, ring[r][0]); maxx = Math.max(maxx, ring[r][0]);
          miny = Math.min(miny, ring[r][1]); maxy = Math.max(maxy, ring[r][1]);
        }
        const has = ring.length > 0;
        objects.push({
          kind: "protected", id: item.id, label: item.label, version: 1, roi_id: item.roi_id,
          center: has ? [(minx + maxx) / 2, (miny + maxy) / 2, 0] : [0, 0, 0],
          size: has ? [Math.max(maxx - minx, 1e-6), Math.max(maxy - miny, 1e-6), 0] : [1e-6, 1e-6, 0],
          review: "reviewed"
        });
      }
    }
    const observation = (state.observations || [])[0] || {};
    const summary = geometry.summary || {};
    return {
      project_id: state.project_id,
      project_version: state.project_version,
      content_hash: state.content_hash,
      unit: observation.unit || "UNKNOWN",
      coordinate_frame: observation.coordinate_frame || "UNKNOWN",
      license: observation.license || "UNKNOWN",
      evidence_policy: state.geometry_policy || "UNRECORDED",
      real_quality: state.real_quality || "undefined",
      counts: {
        observations: counts.observations || 0,
        fragments: counts.fragments || 0,
        surfaces: counts.surfaces || 0,
        relations: counts.relations || 0,
        protected: counts.protected_regions || 0,
        display_points: Number(summary.raw_display_point_count || geometry.points.length || 0)
      },
      objects: objects,
      points: geometry.points
    };
  }

  /* ---------------- 状态落地 ---------------- */
  function applyState(state) {
    /* W172 OBS-D：任何成功落地的 state 都带权威 library（含 closeProject 的部分 state 与后续重开）。 */
    S.state_loaded = true;
    lastState = state;
    S.project_open = !!state.project_open;
    S.project_id = state.project_id || null;
    S.project_version = Number(state.project_version || 0);
    S.content_hash = errText(state.content_hash);
    S.read_only_legacy = !!state.read_only_legacy;
    S.schema_version = state.schema_version || null;
    S.source_id = state.source_id || null;
    S.source_domain = state.source_domain || null;
    S.source = state.source_domain === "internal_real" ? "internal" : "synthetic";
    S.bundle_id = state.bundle_id || null;
    S.dirty = !!state.dirty;
    S.saved_state = state.saved_state || null;
    S.library = (state.library || []).map(function (item) {
      return { bundle_id: item.bundle_id, project_version: item.project_version, content_hash: item.content_hash, saved_at: item.measured_at || "" };
    });
    S.jobs = state.jobs || [];
    const running = S.jobs.filter(function (job) { return job.status === "running"; });
    S.activeJobId = running.length ? running[running.length - 1].job_id : null;
    S.evidence_snapshot_id = state.evidence_snapshot_id || null;
    S.evidence_counts = state.evidence_status_counts || null;
    /* W163：便携视图摘要只来自 GET /api/state.portable（服务端 open_package 的
       同一份读数），页面不自行推断包状态。完整明细由 /api/portable/status 提供。 */
    S.portable = state.portable || null;
    S.can_undo = !!state.can_undo;
    S.can_redo = !!state.can_redo;
    S.preview = state.live_preview ? {
      token: state.live_preview.token, base_version: state.live_preview.base_version,
      action: state.live_preview.action, target_ids: state.live_preview.target_ids || [],
      diff: state.live_preview.diff || {}, warnings: state.live_preview.warnings || [],
      created_at: state.live_preview.created_at, expires_in_seconds: state.live_preview.expires_in_seconds
    } : null;
    if (!S.project_open) {
      S.selectedId = null;
      detailCache.clear();
      evidenceCache.clear();
    }
    S.project = buildProject(state, geo);
    S.review = {};
    if (S.project) {
      for (let i = 0; i < S.project.objects.length; i++) {
        const obj = S.project.objects[i];
        S.review[obj.id] = obj.review;
      }
      if (S.selectedId && !S.review.hasOwnProperty(S.selectedId)) S.selectedId = null;
    }
  }

  async function loadScene() {
    if (!S.project_open) { geo = { points: [], surfaces: new Map(), protected: new Map(), summary: null }; S.project = null; return; }
    const payload = await api("/api/scene");
    geo = applyScene(payload);
    S.scene = payload;
    S.project = buildProject(lastState, geo);
  }

  async function sync(options) {
    const opts = options || {};
    const state = await api("/api/state");
    applyState(state);
    if (S.project_open && opts.scene !== false) await loadScene();
    emit();
    return state;
  }

  function emit() {
    revision += 1;
    listeners.forEach(function (fn) {
      try { fn(); } catch (err) { /* 订阅者异常不影响数据层 */ }
    });
  }

  function subscribe(fn) {
    if (typeof fn === "function") listeners.add(fn);
    return function () { listeners.delete(fn); };
  }

  /* ---------------- 只读外部状态（与 DOM 同源同刻） ---------------- */
  function snapshot() {
    return {
      project_open: S.project_open,
      project_version: S.project_version,
      content_hash: S.content_hash,
      read_only_legacy: S.read_only_legacy,
      bundle_id: S.bundle_id,
      selected_object_id: S.selectedId,
      evidence_snapshot_id: S.evidence_snapshot_id,
      evidence_status_counts: S.evidence_counts ? JSON.parse(JSON.stringify(S.evidence_counts)) : null,
      last_save_receipt: S.saveReceipt ? JSON.parse(JSON.stringify(S.saveReceipt)) : null,
      last_export_receipt: S.exportReceipt ? JSON.parse(JSON.stringify(S.exportReceipt)) : null,
      portable: S.portable ? JSON.parse(JSON.stringify(S.portable)) : null,
      recent_request_paths: recentRequests.map(function (item) { return item.path; }),
      revision: revision
    };
  }
  Object.defineProperty(window, "__W162_STATE__", { get: snapshot, enumerable: true, configurable: false });
  /* W163 证据面：便携视图的完整服务端读数（open_package 文档）与最近导出回执。 */
  Object.defineProperty(window, "__W163_PORTABLE__", {
    get: function () {
      return {
        summary: S.portable ? JSON.parse(JSON.stringify(S.portable)) : null,
        detail: S.portableDetail ? JSON.parse(JSON.stringify(S.portableDetail)) : null,
        status: S.portableStatus ? JSON.parse(JSON.stringify(S.portableStatus)) : null,
        receipt: S.portableReceipt ? JSON.parse(JSON.stringify(S.portableReceipt)) : null,
        error: S.portableError ? JSON.parse(JSON.stringify(S.portableError)) : null
      };
    },
    enumerable: true, configurable: false
  });

  /* M5 只读外部状态（与 DOM 同源，供自动化冒烟核对） */
  Object.defineProperty(window, "__M5_IDENT_STATE__", {
    get: function () {
      const id = S.ident;
      const cur = id.current_case;
      const effStatus = id.collapse_state ? id.collapse_state.resulting_status : (cur ? cur.status : null);
      return {
        cases_loaded: id.cases_loaded,
        total_cases: id.total_cases,
        selected_cluster_id: id.selected_cluster_id,
        case_status: cur ? cur.status : null,
        effective_status: effStatus,
        provenance_mode: id.collapse_state ? id.collapse_state.provenance_mode : (cur ? cur.provenance_mode : null),
        witness_view: id.witness_view,
        selected_fracture_id: id.selected_fracture_id,
        collapsed: !!id.collapse_state,
        check_history_count: id.check_history.length,
        input_sha256: cur ? cur.input_sha256 : null,
        m3_judge_verdict: cur ? cur.m3_judge_verdict : null,
        cases_error: id.cases_error ? Object.assign({}, id.cases_error) : null,
        case_error: id.case_error ? Object.assign({}, id.case_error) : null
      };
    },
    enumerable: true, configurable: false
  });
  function normalizeDetail(detail, id) {
    const raw = (detail && detail.object) || {};
    const kindRaw = (detail && detail.kind) || "";
    const kind = kindRaw === "fragments" ? "fragment" : kindRaw === "surfaces" ? "surface"
      : kindRaw === "relations" ? "relation" : kindRaw === "protected_regions" ? "protected" : kindRaw;
    const known = S.project && S.project.objects.filter(function (o) { return o.id === id; })[0];
    const base = known ? JSON.parse(JSON.stringify(known)) : { id: id, kind: kind, version: raw.version || 1 };
    base.kind = base.kind || kind;
    base.raw = raw;
    return base;
  }

  async function selectObject(id) {
    if (!id) { S.selectedId = null; emit(); return { ok: true }; }
    S.selectedId = id;
    emit();
    try {
      const results = await Promise.all([
        api("/api/object?id=" + encodeURIComponent(id)),
        api("/api/evidence?id=" + encodeURIComponent(id))
      ]);
      const detail = results[0], evidence = results[1];
      const uiObject = normalizeDetail(detail, id);
      detailCache.set(id, uiObject);
      evidenceCache.set(id, {
        object: uiObject,
        source_files: evidence.source_files || [],
        relation_evidence: evidence.relation_evidence || null,
        status: evidence.status || null,
        basis_current: evidence.basis_current,
        support: evidence.support || null,
        point_preview_policy: evidence.point_preview_policy || null,
        note: evidence.note || "",
        raw: evidence
      });
      emit();
      return { ok: true };
    } catch (err) {
      return fail("select_object", err);
    }
  }

  function getObject(id) {
    if (!id) return null;
    if (detailCache.has(id)) {
      const cached = detailCache.get(id);
      const known = S.project && S.project.objects.filter(function (o) { return o.id === id; })[0];
      return known ? Object.assign({}, known, { raw: cached.raw }) : cached;
    }
    if (!S.project) return null;
    return S.project.objects.filter(function (o) { return o.id === id; })[0] || null;
  }

  function getEvidence(id) {
    if (!id) return null;
    return evidenceCache.get(id) || null;
  }

  /* W169 F1：SELECT_SUPPORT 的默认载荷必须是后端收得下的合法子集骨架。
     后端（edit_engine SELECT_SUPPORT）要求 indices 非空、唯一、>= MIN_SUPPORT_POINTS(3)，
     且是这一片当前支持点索引；空数组必拒 BAD_SUPPORT。
     取「前一半」而不是 3 个点：真实 ROI 数据上 3 点子集拟合出的平面会触发
     SELECT_SUPPORT_BREAKS_SURFACE_COPLANARITY（证据 fixes/api_probe），
     半集与全集均 200；这与 SPLIT_SURFACE 骨架同口径。 */
  const SUPPORT_SKELETON_MIN = 3;
  function selectSupportSkeleton(obj) {
    const support = (obj && Array.isArray(obj.support_indices)) ? obj.support_indices : [];
    const take = support.length >= 2 * SUPPORT_SKELETON_MIN
      ? Math.max(SUPPORT_SKELETON_MIN, Math.floor(support.length / 2))
      : support.length;
    const picked = [];
    for (let i = 0; i < support.length && picked.length < take; i += 1) {
      const value = Number(support[i]);
      if (Number.isFinite(value) && picked.indexOf(value) === -1) picked.push(value);
    }
    return { indices: picked };
  }

  function defaultPayload(action, obj) {
    const kind = obj && obj.kind;
    if (action === "SET_INTERPRETATION") return { interpretation_status: "human_reviewed" };
    if (action === "SET_RELATION") return { state: "yes" };
    if (action === "SET_UNKNOWN") return { unknown_reason: "OCCLUDED" };
    if (action === "SELECT_SUPPORT") return selectSupportSkeleton(obj);
    if (action === "ADD_FRAGMENT") {
      let supObj = obj;
      if (obj && obj.kind === "surface" && Array.isArray(obj.member_fragment_ids) && obj.member_fragment_ids.length) {
        supObj = getObject(obj.member_fragment_ids[0]) || obj;
      }
      const obsId = (supObj && supObj.observation_id) || (obj && obj.observation_id)
        || (lastState && Array.isArray(lastState.observations) && lastState.observations[0] && lastState.observations[0].observation_id)
        || "";
      if (S.source === "synthetic") {
        let baseFrag = (supObj && supObj.kind === "fragment") ? supObj : null;
        if (!baseFrag && S.project && Array.isArray(S.project.objects)) {
          const firstFrag = S.project.objects.find(function (o) { return o.kind === "fragment"; });
          if (firstFrag) baseFrag = getObject(firstFrag.id) || firstFrag;
        }
        const raw = (baseFrag && baseFrag.raw) || {};
        const sr = raw.support_ref || {};
        const targetSurf = (obj && obj.kind === "surface")
          ? obj.id
          : ((S.project && Array.isArray(S.project.objects) && (S.project.objects.find(function (o) { return o.kind === "surface"; }) || {}).id) || "surf_a");
        const suffix = String(Date.now()).slice(-6);
        return {
          surface_id: targetSurf,
          fragment: {
            fragment_id: "frag_ui_" + (S.project_version || 1) + "_" + suffix,
            observation_id: raw.observation_id || obsId || "obs_main",
            support_ref: {
              observation_id: sr.observation_id || raw.observation_id || obsId || "obs_main",
              source_sha256: sr.source_sha256 || "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              storage_uri: sr.storage_uri || "synthetic://obs_main/support",
              index_kind: sr.index_kind || "original_point_index",
              indices: [20, 21, 22],
              roi_id: sr.roi_id || "roi_main",
              decimation_lineage: {}
            },
            plane_local: raw.plane_local || { normal: [0.0, 0.0, 1.0], offset: 0.0 },
            finite_regions_local: raw.finite_regions_local || [{ exterior: [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], holes: [] }],
            residuals: Array.isArray(raw.residuals) && raw.residuals.length ? raw.residuals : [0.0, 0.1, -0.1],
            quality: raw.quality || { rmse: 0.08165, synthetic: true },
            candidate_generator: raw.candidate_generator || { candidate_batch_id: "batch_main", generator_config_hash: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" },
            version: 1
          }
        };
      }
      const skel = { indices: selectSupportSkeleton(supObj).indices };
      if (obsId) skel.observation_id = obsId;
      if (obj && obj.kind === "surface") skel.surface_id = obj.id;
      return skel;
    }
    if (action === "PROTECT_REGION") return { roi_id: "roi_w162" };
    if (action === "EDIT_BOUNDARY") return { region_index: 0 };
    if (kind === "relation") return { state: "unknown", unknown_reason: "OCCLUDED" };
    return {};
  }

  /* ---------------- 八动作 + undo/redo/cancel ---------------- */
  async function adoptState(state, operation) {
    applyState(state);
    /* W169 D04：控件/版本/状态即时反馈，不等 loadScene（真实 ROI 上实测 ~12s）。
       undo 后后端 can_redo 立即为 true，重做按钮据此即时 enabled，消除 ~13.6s 控件冻结；
       场景几何在 loadScene 完成后再 emit 一次同步。refresh 由 sig 门控且幂等，双 emit 安全。 */
    emit();
    if (S.project_open) await loadScene();
    emit();
    logOperation(operation, "服务返回新状态", "v" + S.project_version + " · " + (S.content_hash || "").slice(0, 12) + "… · read_only=" + S.read_only_legacy);
  }

  async function openCatalog(catalogId) {
    try {
      const res = await api("/api/projects/open", { method: "POST", body: { catalog_id: catalogId } });
      await adoptState(res.result, "open_catalog:" + catalogId);
      return { ok: true, state: res.result };
    } catch (err) { return fail("open_catalog:" + catalogId, err); }
  }

  async function openSaved(bundleId) {
    if (!bundleId) return { ok: false, code: "BUNDLE_ID_REQUIRED", message: "没有可重开的保存版本" };
    try {
      const res = await api("/api/projects/open-saved", { method: "POST", body: { bundle_id: bundleId } });
      await adoptState(res.result, "open_saved:" + bundleId);
      return { ok: true, state: res.result };
    } catch (err) { return fail("open_saved", err); }
  }

  /* W169 D03 根因修复：重开恢复不再唯一依赖易失内存态。
     boot 窗口内（init 的 /api/state applyState 之前）或 init 的 /api/state 失败时，
     S.bundle_id/S.library 可能为空，旧逻辑直接短路 BUNDLE_ID_REQUIRED（零网络）→ 永久卡 v0，
     而 bundle 在盘完好（受控故障注入已证：强制 sync 即恢复 v2）。
     这里在无可解析 id 时「按需重新拉取」服务状态/库再开，使恢复路径与 init 成败解耦。 */
  function resolveBundleId() {
    if (S.bundle_id) return S.bundle_id;
    return (S.library && S.library.length) ? S.library[0].bundle_id : null;
  }
  async function reopenSaved(bundleId) {
    let id = bundleId || resolveBundleId();
    if (!id) {
      //  按需重新拉取状态：init 可能未完成，或 /api/state 曾瞬时失败
      try { await sync({ scene: false }); } catch (err) { /* 下面再试库端点兜底 */ }
      id = resolveBundleId();
    }
    if (!id) {
      //  仍无：直接拉库端点（GET /api/library → {library:[...]}），不依赖 state 快照
      try {
        const res = await api("/api/library");
        const lib = (res && (res.library || (res.result && res.result.library))) || [];
        if (lib.length) {
          S.library = lib.map(function (item) {
            return { bundle_id: item.bundle_id, project_version: item.project_version, content_hash: item.content_hash, saved_at: item.measured_at || "" };
          });
          id = S.library[0].bundle_id;
          emit();
        }
      } catch (err) { /* 真无已保存版本 */ }
    }
    if (!id) return { ok: false, code: "NO_SAVED_BUNDLE", message: "没有可重开的保存版本 · 请先保存当前项目，或从登记资料打开数据集" };
    return openSaved(id);
  }

  /* 兼容 prototype 的 CORE.reopenProject(bundleId) 调用面：走稳健恢复路径 */
  async function reopenProject(bundleId) { return reopenSaved(bundleId); }

  async function migrate() {
    try {
      const res = await api("/api/projects/migrate", { method: "POST", body: {} });
      await adoptState(res.result, "migrate");
      return { ok: true, state: res.result };
    } catch (err) { return fail("migrate", err); }
  }

  async function closeProject() {
    try {
      const res = await api("/api/projects/close", { method: "POST", body: {} });
      applyState({ project_open: false, library: res.result.library || [] });
      S.project = null;
      emit();
      return { ok: true };
    } catch (err) { return fail("close", err); }
  }

  async function importProject(text) {
    try {
      const document_ = JSON.parse(text);
      const res = await api("/api/projects/import", {
        method: "POST",
        body: { project: document_, declared_bytes: new TextEncoder().encode(text).length }
      });
      await adoptState(res.result, "import");
      return { ok: true };
    } catch (err) { return fail("import", err); }
  }

  async function preview(spec) {
    const body = {
      action: spec.action,
      target_ids: spec.targets || [],
      payload: spec.payload || {},
      base_version: Number(spec.base_version || S.project_version),
      hardness: spec.hardness || "hard"
    };
    try {
      const res = await api("/api/edit/preview", { method: "POST", body: body });
      const payload = res.result || {};
      if (!payload.preview) throw ApiError("PREVIEW_MISSING", "服务未返回预览令牌", "/api/edit/preview", 200);
      await sync({ scene: false });
      return { ok: true, preview: S.preview || payload.preview };
    } catch (err) { return fail("edit_preview", err); }
  }

  async function commit() {
    if (!S.preview || !S.preview.token) return { ok: false, code: "NO_LIVE_PREVIEW", message: "没有待提交的预览" };
    const token = S.preview.token;
    const baseVersion = S.preview.base_version;
    try {
      const res = await api("/api/edit/commit", { method: "POST", body: { token: token, base_version: baseVersion } });
      const payload = res.result || {};
      await adoptState(payload.state, "commit");
      S.preview = null;
      if (payload.transaction) {
        logOperation("commit_receipt", "新版本 +1", "v" + payload.state.project_version + " · changed=" + ((payload.transaction.changed_ids || []).length));
      }
      emit();
      return { ok: true, version: S.project_version };
    } catch (err) { return fail("edit_commit", err); }
  }

  async function cancelPreview() {
    const token = S.preview && S.preview.token;
    if (!token) return { ok: false, code: "NO_LIVE_PREVIEW", message: "没有可取消的预览" };
    try {
      const res = await api("/api/edit/cancel", { method: "POST", body: { token: token } });
      S.preview = null;
      await sync({ scene: false });
      return { ok: true, result: res.result };
    } catch (err) { return fail("edit_cancel", err); }
  }

  async function undoRedo(kind) {
    try {
      const res = await api("/api/edit/" + kind, { method: "POST", body: { base_version: S.project_version } });
      await adoptState((res.result || {}).state, kind);
      return { ok: true };
    } catch (err) { return fail("edit_" + kind, err); }
  }

  async function saveProject() {
    try {
      const res = await api("/api/projects/save", { method: "POST", body: {} });
      const payload = res.result || {};
      const receipt = payload.receipt || null;
      if (!payload.saved || !receipt) {
        S.saveError = {
          code: ((payload.error || {}).code) || "SAVE_FAILED",
          message: ((payload.error || {}).message) || "保存被后端拒绝",
          at: nowIso()
        };
        S.saveReceipt = null;
        logOperation("save", "saved=true 且回执 verified", "saved=false · " + S.saveError.code, "err");
        await sync({ scene: false });
        return { ok: false, code: S.saveError.code, message: S.saveError.message };
      }
      S.saveError = null;
      S.saveReceipt = {
        saved: true, verified: !!receipt.verified,
        project_version: payload.project_version,
        content_hash: payload.content_hash,
        file_count: receipt.file_count,
        bundle_id: payload.bundle_id,
        version_dir: receipt.version_dir || null,
        evidence_snapshot_id: payload.evidence ? payload.evidence.snapshot_id : null,
        at: payload.measured_at || nowIso()
      };
      await sync({ scene: false });
      return { ok: true, receipt: S.saveReceipt };
    } catch (err) {
      S.saveError = { code: (err && err.code) || "SAVE_FAILED", message: (err && err.message) || "", at: nowIso() };
      S.saveReceipt = null;
      return fail("save", err);
    }
  }

  async function exportProject() {
    try {
      const res = await api("/api/export", { method: "POST", body: {} });
      const payload = res.result || {};
      const readback = payload.readback || {};
      const current = !!payload.content_hash && payload.content_hash === S.content_hash
        && Number(payload.project_version) === Number(S.project_version);
      S.exportReceipt = {
        exported: !!payload.exported,
        verified: !!readback.verified,
        current: current,
        enumeration_count: Number(payload.enumeration_count || 0),
        export_id: payload.export_id || null,
        export_dir: payload.export_dir || null,
        project_version: payload.project_version || null,
        content_hash: payload.content_hash || null,
        evidence_snapshot_id: payload.evidence ? payload.evidence.snapshot_id : null,
        at: readback.measured_at || nowIso(),
        checks: (readback.checks || []).map(function (item) {
          return { name: item.name, pass: !!item.ok, actual: item.actual, expected: item.expected };
        })
      };
      S.exportFiles = (payload.files || []).map(function (file) { return { name: file.name, url: file.download_url }; });
      logOperation("export", "readback.verified=true 且 enumeration>0",
        "verified=" + S.exportReceipt.verified + " · " + S.exportReceipt.enumeration_count + " 项 · current=" + current,
        S.exportReceipt.verified ? "ok" : "err");
      await sync({ scene: false });
      return { ok: !!res.ok && S.exportReceipt.verified, receipt: S.exportReceipt };
    } catch (err) { return fail("export", err); }
  }

  async function startSolve(options) {
    try {
      const res = await api("/api/solve/start", { method: "POST", body: Object.assign({ base_version: S.project_version }, options || {}) });
      await sync({ scene: false });
      return { ok: true, job: res.result };
    } catch (err) { return fail("solve_start", err); }
  }

  async function cancelJob() {
    if (!S.activeJobId) return { ok: false, code: "NO_ACTIVE_JOB", message: "没有运行中的任务" };
    try {
      await api("/api/solve/cancel/" + encodeURIComponent(S.activeJobId), { method: "POST", body: {} });
      await sync({ scene: false });
      return { ok: true };
    } catch (err) { return fail("solve_cancel", err); }
  }

  function previewIsLive() { return !!(S.preview && S.preview.token); }
  function canUndo() { return !!S.can_undo; }
  function canRedo() { return !!S.can_redo; }

  /* ---------------- W163 便携包（W154 格式） ----------------
     全部动作都落到 /api/portable/*：导出由服务端从已保存 bundle 构建并自校验，
     打开走 open_package（只读核查，不替换编辑会话），重定位只接受全文件哈希相同
     的文件。页面不生成包、不改包内文件、不猜测状态。 */
  async function portableStatus() {
    try {
      const doc = await api("/api/portable/status");
      S.portableStatus = doc;
      S.portableDetail = doc.current || null;
      if (doc.current_summary) S.portable = doc.current_summary;
      emit();
      return { ok: true, status: doc };
    } catch (err) { return fail("portable_status", err); }
  }

  async function portableExport() {
    try {
      const res = await api("/api/portable/export", {
        method: "POST", body: { distribution_mode: "REFERENCE_ONLY" }
      });
      const result = res.result || {};
      S.portableError = res.ok ? null : {
        code: (res.error && res.error.code) || "PORTABLE_EXPORT_FAILED",
        message: (res.error && res.error.message) || "导出包未通过复核",
        at: nowIso()
      };
      S.portableReceipt = {
        exported: !!result.exported,
        package: result.package || null,
        package_id: result.package_id || null,
        distribution_mode: result.distribution_mode || null,
        source_domain: result.source_domain || null,
        verification: result.verification || null,
        assets: result.assets || null,
        evidence: result.evidence || null,
        source_integrity: result.source_integrity || null,
        copied_asset_bytes: !!result.copied_asset_bytes,
        open_status: result.open ? result.open.status : null,
        warnings: result.open ? (result.open.warnings || []) : [],
        at: nowIso()
      };
      if (result.open) S.portableDetail = result.open;
      await portableStatus();
      await sync({ scene: false });
      return { ok: !!res.ok, receipt: S.portableReceipt, code: S.portableError && S.portableError.code };
    } catch (err) { return fail("portable_export", err); }
  }

  async function portableOpen(packagePath) {
    if (!packagePath) return { ok: false, code: "PORTABLE_PATH_REQUIRED", message: "未选择便携包" };
    try {
      const res = await api("/api/portable/open", {
        method: "POST", body: { package: packagePath }
      });
      const doc = res.result || {};
      S.portableDetail = doc;
      S.portableError = null;
      await portableStatus();
      await sync({ scene: false });
      return { ok: !!res.ok && !!doc.ok, status: doc.status, doc: doc,
               code: (res.error && res.error.code) || null };
    } catch (err) { return fail("portable_open", err); }
  }

  async function portableRelocate(packagePath, assetId, newPath) {
    if (!packagePath) return { ok: false, code: "PORTABLE_PATH_REQUIRED", message: "未打开便携包" };
    try {
      const res = await api("/api/portable/relocate", {
        method: "POST", body: { package: packagePath, asset_id: assetId, new_path: newPath }
      });
      const result = res.result || {};
      if (result.open) S.portableDetail = result.open;
      S.portableError = res.ok ? null : {
        code: (res.error && res.error.code) || "RELOCATE_REFUSED",
        message: (res.error && res.error.message) || "重定位被拒绝",
        at: nowIso()
      };
      await portableStatus();
      await sync({ scene: false });
      return { ok: !!res.ok, hash_relocate: (result.relocated || {}).hash_relocate,
               status: result.open ? result.open.status : null,
               code: S.portableError && S.portableError.code };
    } catch (err) { return fail("portable_relocate", err); }
  }

  async function portableClose() {
    try {
      await api("/api/portable/close", { method: "POST", body: {} });
      S.portableDetail = null;
      S.portableReceipt = null;
      await portableStatus();
      await sync({ scene: false });
      return { ok: true };
    } catch (err) { return fail("portable_close", err); }
  }

  /* ---------------- M5 贯通可辨识性与主动复核（/api/identifiability/*） ---------------- */
  async function identLoadCases() {
    S.ident.loading_cases = true;
    S.ident.cases_error = null;
    emit();
    try {
      const res = await api("/api/identifiability/cases");
      S.ident.cases = res.cases || [];
      S.ident.total_cases = Number(res.total_cases || S.ident.cases.length);
      S.ident.site_counts = res.site_counts || {};
      S.ident.status_counts = res.status_counts || {};
      S.ident.m2_run_id = res.m2_run_id || null;
      S.ident.m3_verdict_sha256 = res.m3_verdict_sha256 || null;
      S.ident.not_claimed = res.not_claimed || [];
      S.ident.cases_loaded = true;
      S.ident.loading_cases = false;
      emit();
      return { ok: true, data: res };
    } catch (err) {
      S.ident.loading_cases = false;
      S.ident.cases_error = { code: (err && err.code) || "IDENT_CASES_FAILED", message: (err && err.message) || "" };
      return fail("ident_load_cases", err);
    }
  }

  async function identOpenCase(clusterId, opts) {
    if (!clusterId) return { ok: false, code: "CLUSTER_ID_REQUIRED", message: "缺少簇 ID" };
    const options = opts || {};
    S.ident.selected_cluster_id = clusterId;
    S.ident.loading_case = true;
    S.ident.case_error = null;
    S.ident.collapse_state = null;
    S.ident.check_history = [];
    S.ident.selected_fracture_id = null;
    emit();
    try {
      const q = "/api/identifiability/case?id=" + encodeURIComponent(clusterId) + (options.live === false ? "&live=0" : "");
      const res = await api(q);
      S.ident.current_case = res;
      S.ident.loading_case = false;
      if (res.status === "CERTIFIED_CONNECTED") S.ident.witness_view = "connected";
      else if (res.status === "CERTIFIED_DISCONNECTED") S.ident.witness_view = "disconnected";
      else S.ident.witness_view = "nominal";
      const ranked = (res.active_check && res.active_check.ranked_fractures) || [];
      if (ranked.length) S.ident.selected_fracture_id = ranked[0].fracture_id;
      // 同步回左侧清单的实时状态与 ρ
      for (let i = 0; i < S.ident.cases.length; i += 1) {
        if (S.ident.cases[i].cluster_id === clusterId) {
          S.ident.cases[i] = Object.assign({}, S.ident.cases[i], {
            status: res.status,
            cert_source: res.cert_source,
            rho: res.rho ? res.rho.infimum : null,
            rho_float: res.rho ? res.rho.infimum_float : null,
            m3_judge_verdict: res.m3_judge_verdict
          });
        }
      }
      emit();
      return { ok: true, data: res };
    } catch (err) {
      S.ident.loading_case = false;
      S.ident.case_error = { code: (err && err.code) || "IDENT_CASE_FAILED", message: (err && err.message) || "" };
      return fail("ident_open_case:" + clusterId, err);
    }
  }

  function identSetWitnessView(mode) {
    if (mode !== "nominal" && mode !== "connected" && mode !== "disconnected") return;
    S.ident.witness_view = mode;
    emit();
  }

  function identSelectFracture(fid) {
    S.ident.selected_fracture_id = fid || null;
    emit();
  }

  async function identApplyCheck(clusterId, fractureId, branchId, step2Opts) {
    S.ident.applying_check = true;
    emit();
    try {
      const s2 = step2Opts || {};
      const body = { cluster_id: clusterId, fracture_id: fractureId, branch_id: branchId };
      if (s2.step2BranchId) {
        body.step2_branch_id = s2.step2BranchId;
        if (s2.step2FractureId) body.step2_fracture_id = s2.step2FractureId;
      }
      const res = await api("/api/identifiability/apply_check", {
        method: "POST",
        body: body
      });
      S.ident.applying_check = false;
      S.ident.collapse_state = res;
      S.ident.check_history = S.ident.check_history.concat([{
        step_count: res.step_count || 1,
        fracture_id: res.fracture_id || fractureId,
        branch_id: res.branch_id || branchId,
        branch_label: res.branch_label || branchId,
        previous_status: res.previous_status,
        resulting_status: res.resulting_status,
        provenance_mode: res.provenance_mode,
        check_receipt_sha256: res.check_receipt_sha256,
        at_utc: res.at_utc
      }]);
      if (res.resulting_status === "CERTIFIED_CONNECTED") S.ident.witness_view = "connected";
      else if (res.resulting_status === "CERTIFIED_DISCONNECTED") S.ident.witness_view = "disconnected";
      S.ident.selected_fracture_id = res.fracture_id || fractureId;
      emit();
      return { ok: true, data: res };
    } catch (err) {
      S.ident.applying_check = false;
      return fail("ident_apply_check", err);
    }
  }

  function identResetCollapse() {
    S.ident.collapse_state = null;
    S.ident.check_history = [];
    const cur = S.ident.current_case;
    if (cur) {
      if (cur.status === "CERTIFIED_CONNECTED") S.ident.witness_view = "connected";
      else if (cur.status === "CERTIFIED_DISCONNECTED") S.ident.witness_view = "disconnected";
      else S.ident.witness_view = "nominal";
      const ranked = (cur.active_check && cur.active_check.ranked_fractures) || [];
      if (ranked.length) S.ident.selected_fracture_id = ranked[0].fracture_id;
    }
    emit();
  }

  /* ---------------- 启动 ---------------- */
  let readyPromise = null;
  async function init() {
    if (readyPromise) return readyPromise;
    readyPromise = (async function () {
      try {
        const catalog = await api("/api/catalog");
        S.catalog = catalog;
        /* W169 D09：catalog 到达即 emit，配合 sig() 纳入 catalog 签名，登记资料立刻重渲染，
           不再被 /api/state 时延门控（负载下数据集按钮曾长时间缺失，受控延迟已确定性复现）。 */
        emit();
      } catch (err) {
        logOperation("catalog", "登记资料清单", "读取失败 · " + err.code, "err");
      }
      /* W169 D03：/api/state 是恢复已保存版本（S.library/S.bundle_id）的唯一来源；
         单次失败会把界面永久卡在 v0。改为有限重试（3 次，递增退避），瞬时失败/超时不再致命；
         仍失败则如实记日志，重开按钮另有按需重新拉取兜底（reopenSaved）。 */
      let stateOk = false;
      for (let attempt = 1; attempt <= 3 && !stateOk; attempt += 1) {
        try {
          await sync();
          stateOk = true;
        } catch (err) {
          logOperation("state", "服务状态", "读取失败 · 第" + attempt + "次 · " + err.code, "err");
          emit();
          if (attempt < 3) await new Promise((r) => setTimeout(r, 400 * attempt));
        }
      }
      try {
        await portableStatus();
      } catch (err) {
        logOperation("portable_status", "便携包清单", "读取失败 · " + (err && err.code), "err");
        emit();
      }
      S.ready = true;
      document.body.dataset.w162Ready = "1";
      window.__W162_READY__ = Promise.resolve(true);
      emit();
      return true;
    })();
    return readyPromise;
  }

  return {
    S: S, COPY: COPY, SEED: SEED,
    init: init, subscribe: subscribe, emit: emit,
    previewIsLive: previewIsLive, canUndo: canUndo, canRedo: canRedo,
    getObject: getObject, getEvidence: getEvidence, selectObject: selectObject,
    defaultPayload: defaultPayload,
    openCatalog: openCatalog, openSaved: openSaved, reopenProject: reopenProject, reopenSaved: reopenSaved,
    openProject: function (source) { return openCatalog(source === "internal" ? "internal_pf02" : "synthetic_contract"); },
    migrate: migrate, closeProject: closeProject, importProject: importProject,
    preview: preview, createPreview: preview, commit: commit, commitPreview: commit,
    cancelPreview: cancelPreview, undoRedo: undoRedo, saveProject: saveProject,
    exportProject: exportProject, startSolve: startSolve, cancelJob: cancelJob,
    portableStatus: portableStatus, portableExport: portableExport,
    portableOpen: portableOpen, portableRelocate: portableRelocate,
    portableClose: portableClose,
    identLoadCases: identLoadCases, identOpenCase: identOpenCase,
    identSetWitnessView: identSetWitnessView, identSelectFracture: identSelectFracture,
    identApplyCheck: identApplyCheck, identResetCollapse: identResetCollapse,
    sync: sync, snapshot: snapshot
  };
})();
