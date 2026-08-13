# FunnelRAG - 企业级智能 Agent RAG 系统

<p align="center">
  <strong>双阶段漏斗检索 · ReAct 推理 · 私有化部署</strong>
</p>

---

## 项目概述

FunnelRAG 是一个面向企业级私有知识库的高性能 RAG（检索增强生成）系统，采用 **Python + C++ 混合架构**。系统通过双阶段漏斗检索策略实现高效精准的知识召回：

- **粗筛阶段**：Milvus 向量检索，20ms 内捞出 Top-100 候选文档
- **精排阶段**：C++ ONNX Cross-Encoder 重排，50ms 内完成打分与重排
- **生成阶段**：ReAct Agent 自主决策，调用 LLM 生成高质量响应

## 核心特性

| 特性 | 说明 |
|------|------|
| 混合架构 | Python 智能编排 + C++ 精度加速，兼顾开发效率与运行性能 |
| 双阶段漏斗检索 | 粗筛 + 精排策略，平衡召回率与精确率 |
| ReAct 推理循环 | 基于 LangChain Agent，LLM 自主决策工具调用 |
| gRPC 跨语言通信 | Python 层通过 gRPC 异步调用 C++ 精排服务 |
| FastAPI 异步网关 | REST 接口 + SSE 流式响应 |
| 数据不出域 | 文档与模型完全私有化部署 |
| 全链路观测 | Prometheus 监控 P99 延迟，LangSmith 追踪思维链 |

## 系统架构

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Client     │────▶│            FastAPI Gateway :8000             │
└─────────────┘     │  /api/v1/query    /api/v1/query/stream       │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │           ReAct Agent Engine                  │
                    │  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
                    │  │ Milvus  │ │ Rerank   │ │ DirectAnswer │  │
                    │  │ Search  │ │ Tool     │ │ Tool         │  │
                    │  └────┬────┘ └────┬─────┘ └──────────────┘  │
                    └───────┼───────────┼──────────────────────────┘
                            │           │
              ┌─────────────▼──┐    ┌───▼──────────────┐
              │  Milvus 粗筛   │    │  C++ 精排服务     │
              │  Top-100 候选  │    │  gRPC :50051     │
              │  向量 ANN 检索  │    │  ONNX Cross-Encoder│
              └────────────────┘    └───────────────────┘
                            │
              ┌─────────────▼──┐
              │  PostgreSQL    │
              │  文档原文存储   │
              └────────────────┘
```

## 工作流程

1. **请求接入** — 用户查询发送至 FastAPI 网关
2. **Agent 决策** — ReAct Agent 判断是否需要检索知识库
3. **粗筛** — 查询向量化后在 Milvus 中执行 ANN 搜索，召回 Top-100 候选
4. **精排** — 候选文档通过 gRPC 传至 C++ 精排服务，Cross-Encoder 打分重排
5. **生成** — 精排结果注入 Prompt，LLM 生成最终回答
6. **观测** — 全链路指标上报 Prometheus，思维链记录至 LangSmith

## 技术栈

### Python 编排层
- **LangChain** — Agent 框架，ReAct 推理循环
- **FastAPI** — 异步 Web 网关
- **PyMilvus** — 向量数据库客户端
- **asyncpg** — PostgreSQL 异步驱动
- **grpcio** — gRPC 跨语言通信

### C++ 精排层
- **gRPC / protobuf** — 跨语言 RPC 通信
- **ONNX Runtime** — Cross-Encoder 模型推理加速
- **CMake** — 构建管理

### 基础设施
- **Milvus** — 向量数据库（粗筛）
- **PostgreSQL** — 文档原文存储
- **Redis** — 缓存加速
- **Prometheus** — 指标监控
- **LangSmith** — LLM 链路追踪
- **Docker** — 容器化部署

## 项目结构

```
FunnelAgent/
├── python/                    # Python 编排层
│   ├── agent/
│   │   ├── engine.py          # ReAct Agent 引擎
│   │   └── prompt_templates.py
│   ├── api/
│   │   ├── main.py            # FastAPI 入口
│   │   └── routes.py          # 路由定义
│   ├── config/
│   │   └── settings.py        # 全局配置
│   ├── schemas/
│   │   └── models.py          # 请求/响应模型
│   ├── tools/
│   │   ├── milvus_search.py   # Milvus 粗筛工具
│   │   ├── rerank_tool.py     # C++ 精排调用工具
│   │   ├── doc_store.py       # PostgreSQL 文档存储
│   │   └── direct_answer.py   # 直接回答工具
│   └── utils/
│       ├── logging_config.py
│       └── metrics.py
├── cpp/                       # C++ 精排服务
│   ├── include/
│   │   └── onnx_reranker.h
│   ├── src/
│   │   ├── onnx_reranker.cpp
│   │   └── grpc_server.cpp
│   └── CMakeLists.txt
├── proto/
│   └── reranker.proto         # gRPC 服务定义
├── scripts/
│   ├── init_collections.py    # Milvus 集合初始化
│   ├── init_db.py             # PostgreSQL 建表
│   └── ingest.py              # 文档入库
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.python
│   │   └── Dockerfile.cpp
│   └── prometheus/
│       └── prometheus.yml
├── config.yaml                # 应用配置
├── .env.example               # 环境变量模板
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url> && cd FunnelAgent

# 创建并激活 conda 环境
conda create -n langchain python=3.13 -y
conda activate langchain

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入实际的 API Key 和服务地址
```

### 3. 启动外部服务

确保 Milvus、PostgreSQL、Redis 等 Docker 容器已运行：

```bash
docker compose up -d
```

### 4. 初始化数据库

```bash
# 创建 Milvus 集合
python scripts/init_collections.py

# 创建 PostgreSQL 文档表
python scripts/init_db.py
```

### 5. 启动服务

```bash
# 启动 Python 网关
python python/api/main.py
```

### 6. 测试接口

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 查询
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "你好，介绍一下你自己"}'

# 流式查询
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是向量数据库？"}'
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/query` | 查询接口 |
| POST | `/api/v1/query/stream` | SSE 流式查询 |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/docs` | Swagger 文档 |

## 性能目标

| 指标 | 目标 |
|------|------|
| 粗筛延迟 (Top-100) | < 20ms |
| 精排延迟 (100 候选) | < 50ms |
| 端到端响应时间 | < 2s |
| Top-100 召回率 | > 95% |
| Top-5 精确率 | > 85% |

## 开发环境

- Python 3.10+
- C++17 编译器 (GCC 9+)
- CMake 3.16+
- Docker 20.10+

## License

Private - Internal Use Only
