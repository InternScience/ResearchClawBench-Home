#!/usr/bin/env python3
"""API Proxy — fixes encrypted_content org-id mismatch on multi-account OpenAI pools.

Listens on 127.0.0.1:13889, forwards to upstream OpenAI-compatible API.
Strips reasoning.encrypted_content from both requests and responses to prevent
org-id mismatch errors when the pool rotates across different organizations.

Usage:
    export UPSTREAM_URL=http://your-api-pool:port
    python3 proxy.py
"""

import http.server, urllib.request, urllib.error, json, os, sys, time, gzip

UPSTREAM = os.environ.get("LLM_BASE_URL", "").rstrip("/")
if UPSTREAM.endswith("/v1"):
    UPSTREAM = UPSTREAM[:-3]  # Strip /v1 suffix — proxy adds /v1 per-request
if not UPSTREAM:
    sys.stderr.write("Error: LLM_BASE_URL environment variable is required.\n")
    sys.stderr.write("Example: export LLM_BASE_URL=http://your-api:port/v1\n")
    sys.exit(1)

LISTEN_PORT = int(os.environ.get("PROXY_PORT", "13889"))
STRIP_HEADERS = {"transfer-encoding", "content-encoding", "content-length"}


def _strip_encrypted_content(obj):
    """Remove encrypted_content from responses and messages to prevent org-id mismatch."""
    for c in obj.get("choices", []):
        for key in ("message", "delta"):
            msg = c.get(key)
            if not msg:
                continue
            if isinstance(msg.get("reasoning"), dict):
                msg["reasoning"].pop("encrypted_content", None)
            if isinstance(msg.get("content"), list):
                msg["content"] = [
                    item for item in msg["content"]
                    if not (isinstance(item, dict) and item.get("type") == "reasoning" and "encrypted_content" in item)
                ]
    for m in obj.get("messages", []):
        if isinstance(m.get("reasoning"), dict):
            m["reasoning"].pop("encrypted_content", None)
        if isinstance(m.get("content"), list):
            for item in m["content"]:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    item.pop("encrypted_content", None)
    return obj


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def _forward(self, body=None):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in {"host", "content-length", "transfer-encoding", "accept-encoding"}}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        path = self.path
        if not path.startswith("/v1"):
            path = "/v1" + path
        req = urllib.request.Request(UPSTREAM + path, data=body, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            return e

    def _decompress(self, raw, resp):
        enc = resp.headers.get("Content-Encoding", "")
        if enc == "gzip" or raw[:2] == b'\x1f\x8b':
            try:
                return gzip.decompress(raw)
            except Exception:
                pass
        return raw

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Strip encrypted_content from request
        try:
            req_json = json.loads(body)
            if isinstance(req_json.get("include"), list):
                req_json["include"] = [x for x in req_json["include"] if x != "reasoning.encrypted_content"]
                if not req_json["include"]:
                    del req_json["include"]
            _strip_encrypted_content(req_json)
            body = json.dumps(req_json).encode()
        except Exception:
            pass

        t0 = time.time()
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] POST {self.path} body={len(body)}b\n")
        sys.stderr.flush()

        resp = self._forward(body)
        ct = resp.headers.get("Content-Type", "")
        is_stream = "text/event-stream" in ct

        if is_stream:
            self.send_response(200)
            for k, v in resp.headers.items():
                if k.lower() not in STRIP_HEADERS:
                    self.send_header(k, v)
            self.end_headers()
            chunks_sent = 0
            event_lines = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    if event_lines:
                        for i, el in enumerate(event_lines):
                            if el.startswith("data: ") and el != "data: [DONE]":
                                try:
                                    chunk = json.loads(el[6:])
                                    if isinstance(chunk.get("choices"), list) and len(chunk["choices"]) == 0:
                                        event_lines = []
                                        break
                                    chunk.pop("obfuscation", None)
                                    chunk.pop("prompt_filter_results", None)
                                    _strip_encrypted_content(chunk)
                                    for c in chunk.get("choices", []):
                                        c.pop("content_filter_results", None)
                                        c.pop("content_filter_result", None)
                                    event_lines[i] = "data: " + json.dumps(chunk)
                                except Exception:
                                    pass
                                break
                        if event_lines:
                            try:
                                self.wfile.write(("\n".join(event_lines) + "\n\n").encode("utf-8"))
                                self.wfile.flush()
                                chunks_sent += 1
                            except BrokenPipeError:
                                break
                        event_lines = []
                else:
                    event_lines.append(line)
            if event_lines:
                try:
                    self.wfile.write(("\n".join(event_lines) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
            sys.stderr.write(f"  stream done: {chunks_sent} events in {time.time()-t0:.2f}s\n")
            sys.stderr.flush()
        else:
            raw = resp.read()
            data = self._decompress(raw, resp)
            try:
                resp_json = json.loads(data)
                _strip_encrypted_content(resp_json)
                data = json.dumps(resp_json).encode("utf-8")
            except Exception:
                pass
            status = resp.status if hasattr(resp, "status") else 200
            self.send_response(status)
            for k, v in resp.headers.items():
                if k.lower() not in STRIP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            sys.stderr.write(f"  non-stream: {len(data)}b in {time.time()-t0:.2f}s\n")
            sys.stderr.flush()

    def do_GET(self):
        resp = self._forward()
        raw = resp.read()
        data = self._decompress(raw, resp)
        status = resp.status if hasattr(resp, "status") else 200
        self.send_response(status)
        for k, v in resp.headers.items():
            if k.lower() not in STRIP_HEADERS:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args): pass

if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
    sys.stderr.write(f"Proxy on http://127.0.0.1:{LISTEN_PORT} -> {UPSTREAM}\n")
    sys.stderr.flush()
    server.serve_forever()
