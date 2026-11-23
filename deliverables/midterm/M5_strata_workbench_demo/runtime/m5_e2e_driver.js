
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const { createRequire } = require("node:module");

const pwIndex = process.argv[2];
const base = process.argv[3];
const shotDir = process.argv[4];
const chromePath = process.argv[5];

const { chromium } = require(pwIndex);

async function main() {
  fs.mkdirSync(shotDir, { recursive: true });
  const report = {
    base,
    started_at_utc: new Date().toISOString(),
    steps: [],
    externalRequests: [],
    consoleErrors: [],
    pageErrors: [],
    screenshots: [],
    addFragmentCheck: {},
    dualWitnessCheck: {},
    activeCollapseCheck: {},
    negativeUiCheck: {},
    ok: false,
  };

  let browser;
  try {
    browser = await chromium.launch({
      executablePath: chromePath,
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--window-size=1440,900",
        "--lang=zh-CN",
        "--force-device-scale-factor=1",
      ],
    });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      locale: "zh-CN",
      timezoneId: "Asia/Shanghai",
    });
    const page = await context.newPage();
    page.setDefaultTimeout(120000);

    let expectHttpReject = false;
    page.on("request", (req) => {
      const u = new URL(req.url());
      if (u.protocol !== "data:" && u.protocol !== "blob:" && u.origin !== new URL(base).origin) {
        report.externalRequests.push({ method: req.method(), url: req.url() });
      }
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        if (!expectHttpReject) report.consoleErrors.push(msg.text());
      }
    });
    page.on("pageerror", (err) => {
      report.pageErrors.push(String((err && err.stack) || err));
    });

    async function shot(name) {
      const p = path.join(shotDir, name);
      await page.screenshot({ path: p, fullPage: false });
      report.screenshots.push({ name, path: p, bytes: fs.statSync(p).size });
    }

    async function canvasFingerprint() {
      return await page.evaluate(() => {
        const c = document.getElementById("plate");
        return c ? c.toDataURL("image/png").slice(0, 4000) : "";
      });
    }

    // 1) Load Workbench mode & verify ADD_FRAGMENT option + transaction
    await page.goto(base + "/", { waitUntil: "domcontentloaded" });
    await page.locator('[data-testid="open-synthetic"]').waitFor({ state: "visible" });
    await page.locator('[data-testid="open-synthetic"]').click();
    await page.locator('[data-testid="object-row-surf_a"]').waitFor({ state: "visible", timeout: 30000 });
    await page.locator('[data-testid="object-row-frag_a"]').click();
    await page.waitForTimeout(300);
    await page.locator('[data-testid="object-row-surf_a"]').click();
    await page.locator('#ed-action').waitFor({ state: "visible" });

    const edOptions = await page.evaluate(() => {
      const sel = document.getElementById("ed-action");
      return sel ? Array.from(sel.options).map((o) => o.value) : [];
    });
    if (!edOptions.includes("ADD_FRAGMENT")) {
      throw new Error("ADD_FRAGMENT missing from #ed-action options: " + JSON.stringify(edOptions));
    }
    await page.selectOption("#ed-action", "ADD_FRAGMENT");
    await page.waitForTimeout(300);
    const payloadText = await page.locator("#ed-payload").inputValue();
    const parsedPayload = JSON.parse(payloadText);
    if (!parsedPayload.fragment || !parsedPayload.surface_id) {
      throw new Error("ADD_FRAGMENT skeleton invalid: " + payloadText);
    }
    await page.locator('[data-testid="preview"]').click();
    await page.locator('[data-testid="commit"]').waitFor({ state: "visible", timeout: 30000 });
    await page.locator('[data-testid="commit"]').click();
    await page.waitForFunction(() => {
      const el = document.querySelector('[data-testid="project-version"]');
      return el && Number(el.textContent.trim()) >= 2;
    }, null, { timeout: 30000 });
    await shot("00_workbench_add_fragment.png");
    report.addFragmentCheck = {
      hasOption: true,
      optionCount: edOptions.length,
      skeletonKeys: Object.keys(parsedPayload),
      committedVersion: await page.locator('[data-testid="project-version"]').textContent(),
    };

    // 2) Negative control on ADD_FRAGMENT (insufficient indices < 3 rejected HTTP 400)
    expectHttpReject = true;
    const negAdd = await page.evaluate(async () => {
      const r = await fetch("/api/edit/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "ADD_FRAGMENT",
          target_ids: ["frag_a"],
          payload: { indices: [0, 1], observation_id: "obs_synth" },
          base_version: Number(document.querySelector('[data-testid="project-version"]').textContent.trim()),
        }),
      });
      return { status: r.status, json: await r.json() };
    });
    expectHttpReject = false;
    if (negAdd.status < 400 || negAdd.json.ok !== false) {
      throw new Error("Expected ADD_FRAGMENT <3 indices to fail, got: " + JSON.stringify(negAdd));
    }
    report.negativeUiCheck.badAddFragment = { httpStatus: negAdd.status, error: negAdd.json.error };

    // 3) Switch to Identifiability Mode and verify all 22 real clusters load
    await page.locator('[data-testid="mode-tab-identifiability"]').click();
    await page.locator('[data-testid="id-cluster-row-jardim_c02"]').waitFor({ state: "visible", timeout: 30000 });
    await page.locator('[data-testid="id-status-badge"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 45000 });
    await page.waitForTimeout(450);

    const clusterCount = await page.locator('[data-testid^="id-cluster-row-"]').count();
    if (clusterCount !== 22) {
      throw new Error("Expected 22 real clusters, rendered: " + clusterCount);
    }
    const boundaryText = await page.locator('[data-testid="id-academic-boundary-bar"]').innerText();
    if (!boundaryText.includes("REAL_DERIVED_CONDITIONED_FINITE_UNION") || !boundaryText.includes("null")) {
      throw new Error("Academic boundary bar incomplete: " + boundaryText);
    }
    const fpNominal = await canvasFingerprint();
    await shot("01_overview_jardim_c02_ambiguous.png");

    // 4) Toggle Connected Witness (m+, t+) and Disconnected Witness (m-, t-)
    await page.locator('[data-testid="witness-btn-connected"]').click();
    await page.waitForTimeout(350);
    const connCap = await page.locator('#vp-caption').innerText();
    const fpConn = await canvasFingerprint();
    if (!connCap.includes("连通链") || fpConn === fpNominal) {
      throw new Error("Connected witness did not visibly update canvas/caption: " + connCap);
    }
    await shot("02_witness_connected_emerald.png");

    await page.locator('[data-testid="witness-btn-disconnected"]').click();
    await page.waitForTimeout(350);
    const discCap = await page.locator('#vp-caption').innerText();
    const fpDisc = await canvasFingerprint();
    if (!discCap.includes("断割见证") || fpDisc === fpConn) {
      throw new Error("Disconnected witness did not visibly update canvas/caption: " + discCap);
    }
    await shot("03_witness_disconnected_crimson.png");
    report.dualWitnessCheck = {
      clusterId: "jardim_c02",
      nominalChangedToConnected: fpConn !== fpNominal,
      connectedChangedToDisconnected: fpDisc !== fpConn,
      connectedCaption: connCap.replace(/\s+/g, " "),
      disconnectedCaption: discCap.replace(/\s+/g, " "),
    };

    // 5) Active Next-Best-Check & Real-time 2-step Collapse on jardim_c02 (AMBIGUOUS -> Step 1 -> Step 2 CERTIFIED_DISCONNECTED)
    const beforeBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    const provInitial = await page.locator('[data-testid="id-provenance-mode"]').innerText();
    await page.locator('[data-testid="apply-check-r1-b0"]').click();
    await page.locator('[data-testid="id-next-step-panel"]').waitFor({ state: "visible", timeout: 25000 });
    await page.waitForTimeout(3200);
    const step1Badge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    if (step1Badge !== "AMBIGUOUS_WITH_VERIFIED_WITNESSES") {
      throw new Error("Expected jardim_c02 rank#1 step1 to remain AMBIGUOUS pending step2, got: " + step1Badge);
    }
    await shot("04a_step1_narrowed_jardim_c02.png");

    await page.locator('[data-testid="apply-step2-b1"]').click();
    await page.locator('[data-testid="id-status-badge"][data-status="CERTIFIED_DISCONNECTED"]').waitFor({ state: "visible", timeout: 25000 });
    await page.locator('[data-testid="id-collapse-result"]').waitFor({ state: "visible" });
    await page.waitForTimeout(3200);
    const afterDiscBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    const provDisc = await page.locator('[data-testid="id-provenance-mode"]').innerText();
    await shot("04_active_collapse_certified_disconnected.png");

    // Reset back to unreviewed AMBIGUOUS state
    await page.locator('[data-testid="id-reset-collapse"]').click();
    await page.locator('[data-testid="id-status-badge"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 10000 });
    const resetBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");

    // 6) Select holed Jardim cluster jardim_c04 & collapse AMBIGUOUS -> CERTIFIED_CONNECTED (1-step)
    await page.locator('[data-testid="id-cluster-row-jardim_c04"]').click();
    await page.locator('[data-testid="id-status-badge"][data-cluster="jardim_c04"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 45000 });
    await page.waitForTimeout(450);
    const c04Title = await page.locator('#vp-caption').textContent();
    if (!String(c04Title).includes("jardim_c04")) {
      throw new Error("Expected jardim_c04 caption, got: " + c04Title);
    }
    await shot("05_jardim_c04_holed_before_collapse.png");

    await page.locator('[data-testid="apply-check-r1-b0"]').click();
    await page.locator('[data-testid="id-status-badge"][data-cluster="jardim_c04"][data-status="CERTIFIED_CONNECTED"]').waitFor({ state: "visible", timeout: 25000 });
    await page.waitForTimeout(3200);
    const afterConnBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    await shot("06_active_collapse_certified_connected.png");

    // 7) Switch site to Perozzo 2024 and inspect perozzo_c02
    await page.locator('[data-testid="ident-site-perozzo_finale_ligure"]').click();
    await page.locator('[data-testid="id-cluster-row-perozzo_c02"]').waitFor({ state: "visible" });
    await page.locator('[data-testid="id-cluster-row-perozzo_c02"]').click();
    await page.locator('[data-testid="id-status-badge"][data-cluster="perozzo_c02"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 30000 });
    await page.waitForTimeout(450);
    await shot("07_perozzo_c02_outcrop_overview.png");

    report.activeCollapseCheck = {
      jardim_c02_before: beforeBadge,
      jardim_c02_step1: step1Badge,
      jardim_c02_after_step2: afterDiscBadge,
      jardim_c02_after_reset: resetBadge,
      jardim_c02_provenance_initial: provInitial,
      jardim_c02_provenance_collapse: provDisc,
      jardim_c04_after: afterConnBadge,
      perozzo_c02_loaded: true,
      clusterCountRendered: clusterCount,
    };

    if (report.externalRequests.length > 0) {
      throw new Error("External network requests detected: " + JSON.stringify(report.externalRequests));
    }
    if (report.consoleErrors.length > 0 || report.pageErrors.length > 0) {
      throw new Error("Browser console/page errors: " + JSON.stringify({ console: report.consoleErrors, page: report.pageErrors }));
    }

    report.ok = true;
    report.finished_at_utc = new Date().toISOString();
    console.log(JSON.stringify(report, null, 2));
  } finally {
    if (browser) await browser.close();
  }
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
