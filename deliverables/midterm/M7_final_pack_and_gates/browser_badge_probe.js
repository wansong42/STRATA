/* M7 · STRATA 绿色 M3 判卷徽章探针（headless Chrome, 外链全阻断） */
const fs = require("fs");
const path = require("path");
const PW = "C:/Users/晚菘/Documents/Default Project/results/w179_midterm_consolidation_20261117/execution/demo/node_modules/playwright-core";
const { chromium } = require(PW);
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";

(async () => {
  const base = process.argv[2];
  const shotDir = process.argv[3];
  const clusters = process.argv.slice(4);
  const out = { ok: false, externalRequests: [], abortedExternal: [], consoleErrors: [], pageErrors: [], cases: [] };
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: CHROME, headless: true,
      args: ["--no-sandbox", "--disable-dev-shm-usage", "--window-size=1440,900",
             "--lang=zh-CN", "--force-device-scale-factor=1"],
    });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 }, locale: "zh-CN", timezoneId: "Asia/Shanghai",
    });
    await context.route("**/*", (route) => {
      const u = new URL(route.request().url());
      const local = u.hostname === "127.0.0.1" || u.hostname === "localhost";
      if (local) return route.continue();
      out.abortedExternal.push(route.request().url());
      return route.abort();
    });
    const page = await context.newPage();
    page.setDefaultTimeout(90000);
    page.on("request", (req) => {
      const u = new URL(req.url());
      const local = u.hostname === "127.0.0.1" || u.hostname === "localhost";
      if (u.protocol !== "data:" && u.protocol !== "blob:" && !local) out.externalRequests.push(req.url());
    });
    page.on("console", (m) => { if (m.type() === "error") out.consoleErrors.push(m.text()); });
    page.on("pageerror", (e) => out.pageErrors.push(String((e && e.stack) || e)));

    fs.mkdirSync(shotDir, { recursive: true });
    for (const cid of clusters) {
      await page.goto(`${base}/?mode=identifiability&cluster=${cid}`, { waitUntil: "domcontentloaded" });
      const badge = page.locator('[data-testid="id-m3-badge"]');
      await badge.waitFor({ state: "visible", timeout: 90000 });
      await page.waitForFunction(() => {
        const el = document.querySelector('[data-testid="id-m3-badge"]');
        return el && el.textContent && !el.textContent.includes("PENDING");
      }, null, { timeout: 90000 });
      const info = await page.evaluate(() => {
        const b = document.querySelector('[data-testid="id-m3-badge"]');
        const s = document.querySelector('[data-testid="id-status-badge"]');
        const cs = getComputedStyle(b);
        return {
          badge_text: b.textContent.trim(),
          badge_class: b.className,
          badge_bg: cs.backgroundColor,
          badge_color: cs.color,
          status_text: s ? s.textContent.trim() : null,
          status_attr: s ? s.getAttribute("data-status") : null,
        };
      });
      const shot = path.join(shotDir, `m3_badge_${cid}.png`);
      await page.screenshot({ path: shot, fullPage: false });
      out.cases.push({ cluster_id: cid, ...info, screenshot: shot, bytes: fs.statSync(shot).size });
    }
    out.ok = true;
  } catch (e) {
    out.error = String((e && e.stack) || e);
  } finally {
    if (browser) { try { await browser.close(); } catch (_) {} }
  }
  console.log(JSON.stringify(out, null, 2));
  process.exit(out.ok ? 0 : 3);
})();
