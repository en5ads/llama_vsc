# ollama_llama_proxy.py
#
# PURPOSE: VS Code Copilot (as of May 2026) only supports Ollama as a local model
# provider. This proxy pretends to BE Ollama on port 11434 so VS Code Copilot can
# connect to it, while secretly forwarding every request to your real llama.cpp
# server running on localhost:8080.
#
# Flow:
#   VS Code Copilot  -->  this proxy (port 11434, speaks Ollama API)
#                              |
#                              v
#                     llama.cpp server (port 8080, speaks OpenAI-compatible API)
#
# VS Code Copilot calls TWO different paths depending on its version / mode:
#   - Older / chat mode:  POST /v1/chat/completions  (OpenAI Chat Completions style)
#   - Newer / agent mode: POST /v1/responses         (OpenAI Responses API style)
# Both arrive at THIS proxy on port 11434 and are forwarded to llama.cpp.
# llama.cpp only speaks Chat Completions, so /v1/responses is translated on the fly.
#
# Configure VS Code: Settings -> "GitHub Copilot: Local Provider" -> Ollama
# Make sure llama.cpp is already running with --port 8080 before starting this proxy.

import json
import uuid
import time
import http.server
import urllib.request
import urllib.error
from datetime import datetime
import re

# --- Configuration ---
LLAMA_HOST = "localhost"          # llama.cpp server host
LLAMA_PORT = 8080                 # llama.cpp server port
PROXY_PORT = 11434                # must match Ollama's default so VS Code finds it
MODEL_NAME = "Qwen3.5-9B:latest"  # Ollama REQUIRES the 'name:tag' format

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>.*?</tool_call>',
    re.DOTALL
)
TOOL_CALL_OPEN = re.compile(r'<tool_call>.*$', re.DOTALL)

def get_llama_context_size():
    """Read actual context size from llama.cpp /props at startup."""
    try:
        req = urllib.request.Request(f"http://{LLAMA_HOST}:{LLAMA_PORT}/props")
        with urllib.request.urlopen(req, timeout=5) as resp:
            props = json.loads(resp.read().decode())
            ctx = props.get("default_generation_settings", {}).get("n_ctx", 32768)
            print(f"[proxy] llama.cpp context size: {ctx}")
            return ctx
    except Exception as e:
        print(f"[proxy] WARNING: couldn't read /props, defaulting context to 32768: {e}")
        return 32768

# At module level, read once at startup
LLAMA_CONTEXT_SIZE = get_llama_context_size()

def sanitize_reasoning_content(text: str) -> str:
    if not text:
        return ""
    cleaned = TOOL_CALL_PATTERN.sub('', text)   # strip complete blocks
    cleaned = TOOL_CALL_OPEN.sub('', cleaned)    # strip unclosed opening tags
    cleaned = re.sub(r'</tool_call>', '', cleaned) # strip orphaned closing tags
    return cleaned  # NO strip() — preserve spaces between tokens!

def forward_to_llama(path, body):
    """Send a non-streaming POST to llama.cpp and return (status_code, response_text)."""
    data = json.dumps(body).encode()
    print(f"[proxy] -> llama.cpp POST {path}  (non-stream)")
    print(f"[proxy]    body: {json.dumps(body)[:300]}")
    req = urllib.request.Request(
        f"http://{LLAMA_HOST}:{LLAMA_PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            response_text = resp.read().decode()
            print(f"[proxy] <- llama.cpp {resp.status}  response: {response_text[:300]}")
            return resp.status, response_text
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"[proxy] !!! llama.cpp returned {e.code}: {error_body[:500]}")
        return e.code, error_body

def trim_messages_to_context(messages, max_chars=None):
    """
    Drop oldest non-system messages if total content exceeds the context limit.
    Always keeps the system message and the last user message.
    max_chars defaults to ~3 chars per token as a conservative estimate.
    """
    if max_chars is None:
        max_chars = LLAMA_CONTEXT_SIZE * 3

    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= max_chars:
        return messages

    print(f"[proxy] WARNING: trimming messages ({total} chars > {max_chars} limit)")
    system = [m for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]

    while len(others) > 1:
        total = sum(len(str(m.get("content", ""))) for m in system + others)
        if total <= max_chars:
            break
        dropped = others.pop(0)
        print(f"[proxy] trimmed: role={dropped['role']} chars={len(str(dropped.get('content', '')))}")

    return system + others

def log_incoming_messages(messages, path):
    """Print incoming messages from VS Code in a readable format."""
    print(f"\n=================== VS CODE -> LLM [{path}] ===================")
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            # content parts — join text parts for display
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        chars = len(content)
        # truncate long messages (e.g. file contents) to keep log readable
        preview = content[:300].replace("\n", " ")
        if chars > 300:
            preview += f"... [{chars} chars]"
        print(f"  [{i}] {role:10s}: {preview}")
    print("================================================================\n")

def forward_stream_to_llama(path, body, wfile):
    if path == "/v1/chat/completions":
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "assistant":
            messages.pop()
        messages = trim_messages_to_context(messages)
        messages.append({"role": "assistant", "content": "</think>"})
        body["messages"] = messages
        body["add_generation_prompt"] = False

    log_incoming_messages(body.get("messages", []), path)  # <-- add here
    data = json.dumps(body).encode()
    print(f"\n=================== STREAM START [{path}] ===================")

    req = urllib.request.Request(
        f"http://{LLAMA_HOST}:{LLAMA_PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    chat_id = f"chatcmpl-{uuid.uuid4()}"
    model_name = body.get("model", "Qwen3.5-9B:latest")
    chunk_count = 0

    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            while True:
                line = resp.readline()
                if not line:
                    print("\n[STREAM] EOF from llama.cpp.")
                    break

                decoded_line = line.decode('utf-8').strip()
                if not decoded_line:
                    continue

                chunk_count += 1

                if decoded_line.startswith("data:"):
                    if "[DONE]" in decoded_line:
                        break

                    try:
                        json_str = decoded_line.replace("data:", "").strip()
                        parsed_json = json.loads(json_str)

                        if "choices" in parsed_json and len(parsed_json["choices"]) > 0:
                            choice = parsed_json["choices"][0]
                            delta = choice.get("delta", {})

                            # --- QWEN REASONING REMAP ---
                            if "reasoning_content" in delta:
                                thinking = delta.pop("reasoning_content")
                                if thinking and thinking.strip():
                                    print(thinking, end="", flush=True)  # show thinking in console
                                if not delta and choice.get("finish_reason") is None:
                                    continue

                            # --- TELEMETRY FILTER ---
                            if not delta and choice.get("finish_reason") is None:
                                continue  # silent drop

                            # --- STOP BLOCK ---
                            if choice.get("finish_reason") == "stop":
                                print(f"[STREAM] stop after {chunk_count} chunks")
                                clean_stop = {
                                    "id": parsed_json.get("id", chat_id),
                                    "object": "chat.completion.chunk",
                                    "created": parsed_json.get("created", int(time.time())),
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                                }
                                wfile.write(f"data: {json.dumps(clean_stop)}\n\n".encode('utf-8'))
                                wfile.flush()
                                continue

                            # --- NORMAL CONTENT CHUNK — just print the token ---
                            token = delta.get("content", "")
                            if token:
                                print(token, end="", flush=True)
                            wfile.write(f"data: {json.dumps(parsed_json)}\n\n".encode('utf-8'))

                        else:
                            # Usage/telemetry chunk
                            if "usage" in parsed_json:
                                usage = parsed_json["usage"]
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                print(f"\n[STREAM] tokens: prompt={prompt_tokens} completion={completion_tokens} total={prompt_tokens+completion_tokens} ctx={LLAMA_CONTEXT_SIZE} ({100*prompt_tokens//LLAMA_CONTEXT_SIZE}% used)")
                                ollama_final = {
                                    "id": parsed_json.get("id", chat_id),
                                    "object": "chat.completion.chunk",
                                    "created": parsed_json.get("created", int(time.time())),
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    "usage": {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens)
                                    },
                                    "num_ctx": LLAMA_CONTEXT_SIZE,
                                    "prompt_eval_count": prompt_tokens,
                                    "eval_count": completion_tokens,
                                }
                                wfile.write(f"data: {json.dumps(ollama_final)}\n\n".encode('utf-8'))
                                wfile.flush()
                            # all other empty-choices chunks dropped silently

                    except json.JSONDecodeError:
                        # Unusual — log in full
                        print(f"\n[STREAM] WARNING: non-JSON chunk #{chunk_count}: {decoded_line}")
                        raw_text = decoded_line.replace("data:", "").strip()
                        openai_chunk = {
                            "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()),
                            "model": model_name, "choices": [{"index": 0, "delta": {"content": raw_text}, "finish_reason": None}]
                        }
                        wfile.write(f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8'))

                wfile.flush()

        wfile.write(b"data: [DONE]\n\n")
        wfile.flush()
        print(f"=================== STREAM END ===================\n")

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"\n[STREAM] HTTP {e.code} from llama.cpp: {error_body[:500]}")
    except Exception as e:
        print(f"\n[STREAM] CRITICAL ERROR: {e}")


def responses_to_chat_completions(parsed):
    """
    Translate an OpenAI Responses API request body into a Chat Completions body.

    The Responses API (used by newer VS Code Copilot) differs from Chat Completions:
      - "input" (string or list) instead of "messages"
      - "instructions" instead of a system message
      - no "stream" at the top level (streaming is implied differently)

    We normalise everything into the Chat Completions format that llama.cpp understands.
    """
    messages = []

    instructions = parsed.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_field = parsed.get("input", [])
    if isinstance(input_field, str):
        messages.append({"role": "user", "content": input_field})
    elif isinstance(input_field, list):
        for item in input_field:
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for p in content:
                    ptype = p.get("type", "")
                    if ptype == "text":
                        text_parts.append(p.get("text", ""))
                    elif ptype == "input_file":
                        # File content — extract filename + text if present
                        fname = p.get("filename", "file")
                        ftext = p.get("file_data", "") or p.get("text", "")
                        if ftext:
                            text_parts.append(f"[File: {fname}]\n{ftext}")
                        else:
                            text_parts.append(f"[File attached: {fname}]")
                    elif ptype in ("image_url", "image_file"):
                        # llama.cpp can handle image_url natively — pass through
                        text_parts.append(p.get("text", "[image]"))
                    else:
                        # Unknown part type — log and skip rather than crash
                        print(f"[proxy] WARNING: unknown content part type '{ptype}' — skipping")
                content = "\n".join(text_parts)
            messages.append({"role": role, "content": content})

    chat_body = {
        "model": parsed.get("model", MODEL_NAME),
        "messages": messages,
        "stream": parsed.get("stream", False),
    }

    for key in ("temperature", "top_p", "max_tokens", "stop"):
        if key in parsed:
            chat_body[key] = parsed[key]

    print(f"[proxy] responses_to_chat: {len(messages)} messages, last role={messages[-1]['role'] if messages else 'none'}")
    return chat_body


class OllamaProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[proxy] {self.command} {self.path} -> {args[0] if args else ''}")

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    # ------------------------------------------------------------------
    # GET handlers
    # ------------------------------------------------------------------
    def do_GET(self):
        print(f"[proxy] GET {self.path}")

        if self.path in ("/", "/api/version"):
            print("[proxy]   -> version probe, returning 0.6.4")
            self.send_json(200, {"version": "0.6.4"})

        elif self.path == "/api/tags":
            print(f"[proxy]   -> /api/tags probe, advertising model '{MODEL_NAME}'")
            self.send_json(200, {
                "models": [
                    {
                        "name": MODEL_NAME,
                        "model": MODEL_NAME,
                        "modified_at": datetime.utcnow().isoformat() + "Z",
                        "size": 22000000000,
                        "digest": "aaaaaaaaaaaaaaaa",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "qwen3",
                            "families": ["qwen3"],
                            "parameter_size": "27B",
                            "quantization_level": "Q6_K"
                        }
                    }
                ]
            })

        elif self.path == "/v1/models":
            print(f"[proxy]   -> /v1/models probe, advertising model '{MODEL_NAME}'")
            self.send_json(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(datetime.utcnow().timestamp()),
                    "owned_by": "local"
                }]
            })

        else:
            print(f"[proxy]   -> UNHANDLED GET '{self.path}'")
            self.send_json(404, {"error": f"Unhandled GET: {self.path}"})

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------
    def do_POST(self):
        try:
            parsed = self.read_body()
            is_stream = parsed.get("stream", False)
            print(f"[proxy] POST {self.path}  stream={is_stream}")

            # --- /api/show --- VS Code calls this after /api/tags to get model details
            if self.path == "/api/show":
                model = parsed.get("model", MODEL_NAME)
                print(f"[proxy]   -> /api/show for '{model}', returning model details")
                self.send_json(200, {
                    "model": MODEL_NAME,
                    "modelfile": f"FROM {MODEL_NAME}",
                    "parameters": f"temperature 1.0\ntop_p 0.95\nnum_ctx {LLAMA_CONTEXT_SIZE}",
                    "template": "{{ .Prompt }}",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "qwen3",
                        "families": ["qwen3"],
                        "parameter_size": "27B",
                        "quantization_level": "Q6_K"
                    },
                    "model_info": {
                        "general.architecture": "qwen3",
                        "general.parameter_count": 27000000000,
                        "general.quantization_version": 2,
                        "qwen3.context_length": LLAMA_CONTEXT_SIZE,
                        "qwen3.attention.head_count": 32,
                    },
                    "capabilities": ["completion", "tools"]
                })

            # --- /api/chat -> /v1/chat/completions (Ollama native chat path) ---
            elif self.path == "/api/chat":
                llama_body = {
                    "model": MODEL_NAME,
                    "messages": parsed.get("messages", []),
                    "stream": is_stream,
                    "temperature": parsed.get("options", {}).get("temperature"),
                    "top_p": parsed.get("options", {}).get("top_p"),
                }
                if is_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    forward_stream_to_llama("/v1/chat/completions", llama_body, self.wfile)
                else:
                    status, body = forward_to_llama("/v1/chat/completions", llama_body)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            # --- /api/generate -> /v1/completions ---
            elif self.path == "/api/generate":
                llama_body = {
                    "model": MODEL_NAME,
                    "prompt": parsed.get("prompt", ""),
                    "stream": is_stream,
                }
                if is_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    forward_stream_to_llama("/v1/completions", llama_body, self.wfile)
                else:
                    status, body = forward_to_llama("/v1/completions", llama_body)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            # --- /v1/chat/completions ---
            # VS Code Copilot calls this DIRECTLY on port 11434 (bypassing /api/chat).
            # We just forward it straight through to llama.cpp on port 8080.
            elif self.path == "/v1/chat/completions":
                parsed["model"] = parsed.get("model", MODEL_NAME)
                print("[proxy]   -> /v1/chat/completions passthrough to llama.cpp")
                if is_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    forward_stream_to_llama("/v1/chat/completions", parsed, self.wfile)
                else:
                    status, body = forward_to_llama("/v1/chat/completions", parsed)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            # --- /v1/responses ---
            # Newer VS Code Copilot agent mode uses the OpenAI Responses API.
            # llama.cpp doesn't speak this; we translate it to Chat Completions first.
            elif self.path == "/v1/responses":
                print("[proxy]   -> /v1/responses (Responses API) -> translating to Chat Completions")
                chat_body = responses_to_chat_completions(parsed)
                is_stream = chat_body.get("stream", False)
                if is_stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    forward_stream_to_llama("/v1/chat/completions", chat_body, self.wfile)
                else:
                    status, body = forward_to_llama("/v1/chat/completions", chat_body)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            else:
                print(f"[proxy]   -> UNHANDLED POST '{self.path}' body={json.dumps(parsed)[:300]}")
                self.send_json(404, {"error": f"Unhandled POST: {self.path}"})

        except Exception as e:
            print(f"[proxy] ERROR on {self.path}: {e}")
            self.send_json(500, {"error": str(e)})


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PROXY_PORT), OllamaProxyHandler)
    print("=" * 60)
    print("Ollama <-> llama.cpp proxy")
    print("=" * 60)
    print(f"  Listening (as Ollama):  http://0.0.0.0:{PROXY_PORT}")
    print(f"  Forwarding to llama.cpp: http://{LLAMA_HOST}:{LLAMA_PORT}")
    print(f"  Advertised model name:   {MODEL_NAME}")
    print()
    print("Handled routes:")
    print("  GET  /api/version, /api/tags, /v1/models")
    print("  POST /api/show, /api/chat, /api/generate")
    print("  POST /v1/chat/completions  <- VS Code Copilot (chat mode)")
    print("  POST /v1/responses         <- VS Code Copilot (agent mode)")
    print()
    print("In VS Code: Settings -> 'GitHub Copilot: Local Provider' -> Ollama")
    print(f"            Model: {MODEL_NAME}")
    print("=" * 60)
    server.serve_forever()