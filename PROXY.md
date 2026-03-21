# API Proxy — 多账号 OpenAI 中转池兼容修复

轻量反向代理，修复通过多账号中转池（共享 API Key 服务，请求在多个 OpenAI 组织间轮转）使用 OpenAI 兼容 API 时的兼容性问题。

## 问题背景

多账号 API 中转池会将请求分散到不同的 OpenAI 组织。模型响应中包含的 `reasoning.encrypted_content` 由当前 org 的密钥加密。如果下一次请求被路由到另一个 org，该加密内容无法解密，报错：

```
Encrypted content could not be verified. Reason: organization_id did not match
```

## 使用方法

```bash
# LLM_BASE_URL 已在环境中配置（~/.bashrc）
# 启动代理（默认端口 13889，可通过 PROXY_PORT 环境变量修改）
python3 proxy.py
```

所有 Agent（Codex CLI、OpenClaw、Nanobot 等）共用此代理。

功能：
- **请求端**：从 `include` 列表和消息历史中剥离 `reasoning.encrypted_content`
- **响应端**：从流式和非流式响应中剥离 `encrypted_content`

## Agent 配置

### Codex CLI

```bash
export OPENAI_BASE_URL=http://127.0.0.1:13889
```

### OpenClaw

在 `~/.openclaw/openclaw.json` 和 `~/.openclaw/agents/main/agent/models.json` 中设置：

```json
"baseUrl": "http://127.0.0.1:13889"
```

**重要**：OpenClaw 的 provider 名必须设为 `"openai"`（不能用自定义名），否则 `coding` profile 的工具调度会失败。（参见 [openclaw#45269](https://github.com/openclaw/openclaw/issues/45269)）

### Nanobot

Nanobot 的 `custom` provider 直接调用 openai SDK，不受 encrypted_content 影响，可直连 API 不需要 proxy。如需使用 proxy：

```json
"custom": {
  "apiKey": "sk-xxx",
  "apiBase": "http://127.0.0.1:13889/v1"
}
```

## 已知问题

- **旧的 Codex 对话无法继续。** 在代理启用之前创建的对话，消息历史中已经包含了 `encrypted_content`，只有新对话能正常工作。

- **必须先启动代理再启动 Agent。** 如果代理未运行，API 调用会直接报连接拒绝。
