#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runs test_llmserver against two fake OpenAI-compatible servers.

usage: test_llmserver.py <test_llmserver> <llmserver plugin> [reference plugin]

The fake server answers GET /v1/models and POST /v1/chat/completions (server-sent events). The reply to a prompt echoes
what the server received (model, options, key, prompt) so the C test can check the request without a real model. This
verifies the backend against the API as documented, not against Ollama, LM Studio or SGLang themselves.
"""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["fake-model", "gone", "tagged:latest"]


def make_handler(required_key):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if required_key and self.headers.get("Authorization") != "Bearer " + required_key:
                self._json(401, {"error": {"message": "bad key"}})
                return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
            elif self.path == "/v1/not-a-list":
                self._json(200, {"hello": "world"})
            else:
                self._json(404, {"error": "no such path"})

        def _event(self, payload):
            data = payload if isinstance(payload, str) else json.dumps(payload)
            self.wfile.write(b"data: " + data.encode() + b"\n\n")
            self.wfile.flush()

        def do_POST(self):
            if not self._authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "no such path"})
                return
            messages = body["messages"]
            prompt = messages[-1]["content"]
            if body["model"] == "gone":
                self._json(404, {"error": {"message": "model 'gone' not found"}})
                return
            if prompt == "__error500__":
                self._json(500, {"error": {"message": "boom"}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if prompt == "__midstream__":
                self._event({"choices": [{"delta": {"content": "partial"}}]})
                self._event({"error": {"message": "out of memory"}})
                return
            if prompt == "__count__":
                text = "".join("%d " % i for i in range(50))
            else:
                system = messages[0]["content"] if messages[0]["role"] == "system" else ""
                text = "model=%s;sys=%s;max=%s;temp=%s;top_p=%s;seed=%s;key=%s;prompt=%s" % (
                    body["model"], system, body.get("max_tokens"), body.get("temperature"), body.get("top_p"),
                    body.get("seed"), (self.headers.get("Authorization") or "").replace("Bearer ", ""), prompt)
            self._event({"choices": [{"delta": {"role": "assistant"}}]})
            for i in range(0, len(text), 3):
                self._event({"choices": [{"delta": {"content": text[i:i + 3]}}]})
            self._event({"choices": [], "usage": {"total_tokens": 1}})
            self.wfile.write(b": comment\r\n\r\n")
            self._event("[DONE]")
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "AFTER-DONE"}}]}).encode() + b"\n\n")

    return Handler


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A client that stops reading (the callback asked to stop) closes the connection mid-stream.
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def serve(required_key):
    server = QuietServer(("127.0.0.1", 0), make_handler(required_key))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    exe, plugin = sys.argv[1], sys.argv[2]
    reference = sys.argv[3] if len(sys.argv) > 3 else ""
    plain, keyed = serve(None), serve("secret")
    result = subprocess.run([
        exe, plugin, reference or "-",
        "http://127.0.0.1:%d" % plain.server_address[1],
        "http://127.0.0.1:%d" % keyed.server_address[1],
    ])
    plain.shutdown()
    keyed.shutdown()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
