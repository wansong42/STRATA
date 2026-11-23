# STRATA 三维贯通可辨识性公网展示台 · 云服务器上线指南 (`DEPLOY.md`)

本部署包 (`deploy/strata_showcase/`) 专为目标公网服务器 **`49.233.186.35`**（SSH 22 端口开放；**80/443 端口已由现役 Caddy 占用并做 308 HTTPS 跳转**）深度适配：

- **严格零端口冲突**：绝不占用 80/443 端口，绝不启动额外 Nginx，绝不覆盖原有 `Caddyfile` 其它站点。
- **100% `/strata/` 子路径相对路径兼容**：全部静态资源、22 簇 JSON 与 `/api/copilot/chat` 均使用相对路径 `./...`。
- **真 LLM 流式 AI 讲解 + 离线零错兜底**：配置服务器环境变量 `STRATA_LLM_API_KEY` 即开启 OpenAI 兼容真 LLM 流式 SSE 对话与 3D 自动联动（含防刷限流：每 IP 每分钟 ≤ 12 次、每小时 ≤ 60 次）；未配置 Key 或上游超时时自动返回 `fallback: true` 无缝切至离线增强知识库，绝不报 500。
- **零遮挡桌面四页签 + 深度移动端适配**：桌面端将「AI 讲解」整合为右侧第 4 页签（彻底告别悬浮窗遮挡 3D 分屏断口）；移动端支持三态可拖拽底部抽屉（沉浸 3D / 均衡 / 展开）、双指缩放+双指平移、单行可滑操控坞与安全区留白。

---

## 一、推荐上线方案：Caddy 静态直出 + `/strata/api/*` 反代本机 `127.0.0.1:8856`（支持真 LLM SSE 流式）

### 第 1 步：上传完整部署包至 `49.233.186.35`

```powershell
# 已为您打包好的归档（位于 D:\share\showcase.tar.gz）直接上传：
scp D:\share\showcase.tar.gz root@49.233.186.35:/tmp/
```

登录服务器解压并同步静态目录：
```bash
ssh root@49.233.186.35
sudo mkdir -p /opt /srv/strata_showcase
sudo tar -xzf /tmp/showcase.tar.gz -C /opt/
sudo cp -a /opt/strata_showcase/dist/. /srv/strata_showcase/
sudo chown -R caddy:caddy /srv/strata_showcase 2>/dev/null || sudo chmod -R a+rX /srv/strata_showcase
```

> **仅增量更新前端与移动端补丁时**，只需上传补丁包 `showcase_update_patch.tar.gz` 并覆盖解压：
> ```bash
> sudo tar -xzf /tmp/showcase_update_patch.tar.gz -C /opt/strata_showcase/
> sudo cp -a /opt/strata_showcase/dist/. /srv/strata_showcase/
> sudo systemctl restart strata-showcase 2>/dev/null || true
> ```

### 第 2 步：安全配置大模型环境变量（严禁写入前端 `dist/`）

```bash
sudo tee /etc/strata_showcase.env > /dev/null << 'EOF'
# 兼容任意 OpenAI Chat Completions 接口（DeepSeek / OpenAI / 硅基流动 / 通义兼容模式等）
STRATA_LLM_BASE_URL=https://api.deepseek.com/v1
STRATA_LLM_API_KEY=在此填入你的真实API_Key（留空则自动走离线增强知识库）
STRATA_LLM_MODEL=deepseek-chat
STRATA_LLM_TIMEOUT=25
STRATA_COPILOT_RPM=12
STRATA_COPILOT_RPH=60
EOF
sudo chmod 600 /etc/strata_showcase.env
```

### 第 3 步：启动只读 Python 服务（systemd 守护，仅监听本机 `127.0.0.1:8856`）

```bash
python3 /opt/strata_showcase/server.py --self-check

sudo tee /etc/systemd/system/strata-showcase.service > /dev/null << 'EOF'
[Unit]
Description=STRATA 3D Public Read-Only Showcase + Copilot SSE Proxy (127.0.0.1:8856)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/strata_showcase
EnvironmentFile=-/etc/strata_showcase.env
ExecStart=/usr/bin/python3 /opt/strata_showcase/server.py --host 127.0.0.1 --port 8856
Restart=always
RestartSec=3
MemoryMax=160M

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now strata-showcase
curl -s http://127.0.0.1:8856/api/identifiability/health
```

### 第 4 步：向现役 `/etc/caddy/Caddyfile` 站点块追加 `/strata/` 与 SSE 反代片段

> **关键提醒**：SSE 流式输出必须配置 **`flush_interval -1`**，并且不要对 `/strata/api/*` 启用外层 `encode gzip/zstd` 压缩缓冲，否则打字机流式块会被代理攒到结束后才一次性吐出！

先备份现役配置：
```bash
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak.$(date +%Y%m%d_%H%M%S)
```

在现有 HTTPS 站点块 `{ ... }` 内部追加以下片段（完整模板见 `Caddyfile.snippet`）：
```caddy
    # --- STRATA 三维展示台：API & SSE 流式反代（务必关闭缓冲 flush_interval -1） ---
    redir /strata /strata/ 308
    handle_path /strata/api/* {
        rewrite * /api{uri}
        reverse_proxy 127.0.0.1:8856 {
            flush_interval -1
            header_up X-Real-IP {remote_host}
            transport http {
                read_timeout 60s
            }
        }
    }

    # --- STRATA 静态资源直出（3D 模型、Three.js、22 簇只读 JSON） ---
    handle_path /strata/* {
        root * /srv/strata_showcase
        encode zstd gzip
        header X-Content-Type-Options "nosniff"
        header X-Frame-Options "SAMEORIGIN"
        header Referrer-Policy "strict-origin-when-cross-origin"
        header X-Strata-Mode "READONLY_STATIC_SHOWCASE"
        @immutable path /vendor/*
        header @immutable Cache-Control "public, max-age=2592000, immutable"
        @data path /data/*
        header @data Cache-Control "public, max-age=3600"
        file_server
    }
```

校验并零中断热重载 Caddy：
```bash
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

快速验证上线与流式接口：
```bash
curl -I https://<现役域名或49.233.186.35>/strata/
curl -s https://<现役域名或49.233.186.35>/strata/api/identifiability/health
curl -N -X POST https://<现役域名或49.233.186.35>/strata/api/copilot/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"一句话解释为什么用分数算2.97mm断口","context":{"cluster_id":"jardim_c02"}}'
```

---

## 二、极简替代方案：纯静态零进程托管（离线增强知识库模式）

若服务器暂不运行 Python 进程，仅需保留上述第 4 步中的 `handle_path /strata/*` 静态块（去掉 `/strata/api/*` 块）。前端在无法连接 `/api/copilot/chat` 时会自动静默降级至内置离线增强知识库，3D 分屏、微距聚焦、1/2 步复核坍缩、AI 讲解与 SHA256 核验全部正常可用。

---

## 三、公网访问与答辩 PPT 二维码建议

- **公网直达 URL**：`https://<服务器现役域名>/strata/`（或已配 IP 证书时 `https://49.233.186.35/strata/`）
- **答辩扫码体验**：评委手机扫码进入后，单指旋转、双指缩放/平移三维裂隙网络；点底部「**AI 讲解**」可边看上方 3D 实时分屏联动边提问，点抽屉手柄「**沉浸 3D**」一键全屏观赏模型。
