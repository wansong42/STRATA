# =========================================================================
# STRATA 3D Public Showcase · 轻量只读容器镜像 (适配 1C1G 云服务器)
# 运行内存占用 < 40 MB · 零写盘接口 · 启动即核验 22/22 簇 SHA256
# =========================================================================
FROM python:3.11-slim

LABEL maintainer="CUMTB STRATA Research Team" \
      description="STRATA 3D Rock Fracture Network Identifiability Read-Only Showcase"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    STRATA_SHOWCASE_QUIET=0

WORKDIR /app/deploy/strata_showcase

# 创建非特权只读运行用户
RUN groupadd -r strata && useradd -r -g strata strata

# 复制只读服务、零幻觉知识模块与预烘焙静态校验包
COPY server.py /app/deploy/strata_showcase/server.py
COPY copilot_knowledge.py /app/deploy/strata_showcase/copilot_knowledge.py
COPY export_static_bundle.py /app/deploy/strata_showcase/export_static_bundle.py
COPY dist /app/deploy/strata_showcase/dist

# 构建期执行 22 簇 SHA256 完整性自检，任何 1 字节不一致即中断镜像构建
RUN python server.py --self-check --dist /app/deploy/strata_showcase/dist && \
    chown -R strata:strata /app

USER strata

EXPOSE 8856

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,json,sys; r=json.loads(urllib.request.urlopen('http://127.0.0.1:8856/api/identifiability/health',timeout=3).read()); sys.exit(0 if r.get('ok') and r.get('sha_verified_count')==22 else 1)"

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8856", "--dist", "/app/deploy/strata_showcase/dist"]
