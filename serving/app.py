"""Local HTTP adapter for pinned feln-lora client and read-only spatial execution."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb
from src.edge_client import compile_raw, complete, healthy, validate_text
from src.feln_data import Schema
from src.spatial_query import SpatialQuery

ROOT = Path(os.environ.get("FELN_ROOT", "/data/feln-qwen-20260917"))
BUNDLE = ROOT / "bundle"
MODEL_URL = os.environ.get("FELN_MODEL_URL", "http://127.0.0.1:18092")
CONFIG = json.loads((BUNDLE / "inference_config.json").read_text())
SCHEMA = Schema(BUNDLE / "Layers.json")
DATABASE = (
    SpatialQuery(ROOT / "NorthSea.ddb", SCHEMA)
    if (ROOT / "NorthSea.ddb").is_file()
    else None
)
SLOT = threading.BoundedSemaphore(1)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(190)

    def send_json(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "Not found"})
        ready = healthy(MODEL_URL)
        self.send_json(
            200 if ready else 503,
            {
                "ready": ready,
                "model_url": MODEL_URL,
                "spatial_execution": DATABASE is not None,
            },
        )

    def do_POST(self):
        if self.path not in {"/feln", "/query"}:
            return self.send_json(404, {"error": "Not found"})
        if not SLOT.acquire(blocking=False):
            return self.send_json(
                429, {"error": "One inference request is already running"}
            )
        started = time.monotonic()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Body must contain 1–8192 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) - {"text", "limit"}:
                raise ValueError("Expected text and optional limit")
            text = payload.get("text")
            validate_text(text)
            if self.path == "/query" and DATABASE is None:
                return self.send_json(
                    503,
                    {
                        "error": "Local database is not installed; use /feln for generation"
                    },
                )
            limit = payload.get("limit", 50)
            if type(limit) is not int or not 1 <= limit <= 5000:
                raise ValueError("limit must be an integer from 1 to 5000")
            response = complete(CONFIG, text, MODEL_URL)
            meta = compile_raw(SCHEMA, response["content"])
            result = {
                "feln": meta,
                "raw": response["content"],
                "model_timings": response.get("timings"),
            }
            if self.path == "/query":
                assert DATABASE is not None
                result["execution"] = DATABASE.execute(meta, limit)
            result["seconds"] = time.monotonic() - started
            self.send_json(200, result)
        except (ValueError, TypeError) as exc:
            self.send_json(422, {"error": str(exc)})
        except (OSError, duckdb.Error) as exc:
            self.send_json(503, {"error": str(exc)})
        finally:
            SLOT.release()


if __name__ == "__main__":
    # Load the preinstalled spatial extension before declaring the application ready.
    if DATABASE is not None:
        DATABASE.connect().close()
    ThreadingHTTPServer(("127.0.0.1", 18091), Handler).serve_forever()
