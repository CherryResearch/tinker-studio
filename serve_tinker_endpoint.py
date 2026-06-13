from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook.renderers import TrainOnWhat

from run_tinker_experiment import find_latest_payload
from tinker_notebook_env import ensure_tinker_api_key
from tinker_training_utils import (
    ConversationExample,
    build_datums,
    sample_generations,
    select_renderer_name,
)


def extract_sampler_checkpoint(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    sampler_model_path = payload.get("sampler_model_path")
    if isinstance(sampler_model_path, str) and sampler_model_path.strip():
        return sampler_model_path
    summary = payload.get("summary")
    if isinstance(summary, dict):
        sampler_model_path = summary.get("sampler_model_path")
        if isinstance(sampler_model_path, str) and sampler_model_path.strip():
            return sampler_model_path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local OpenAI-compatible chat bridge for a Tinker sampler checkpoint.")
    parser.add_argument("--workspace", default=str(Path.cwd()), help="Workspace root.")
    parser.add_argument("--run-name", default="essay_recent_r16", help="Run name used to infer the sampler checkpoint.")
    parser.add_argument("--sampler-checkpoint", help="Explicit sampler checkpoint path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    parser.add_argument("--model-id", default="", help="Model id advertised through /v1/models.")
    return parser.parse_args()


class TinkerEndpoint:
    def __init__(self, *, workspace: Path, run_name: str, sampler_checkpoint: str | None, model_id: str) -> None:
        ensure_tinker_api_key(required=True)
        self.workspace = workspace
        self.run_name = run_name
        self.sampler_checkpoint = sampler_checkpoint or self._resolve_sampler_checkpoint()
        self.model_id = model_id or run_name
        self.service_client = tinker.ServiceClient()
        rest_client = self.service_client.create_rest_client()
        checkpoint_info = rest_client.get_weights_info_by_tinker_path(self.sampler_checkpoint).result()
        self.base_model = str(getattr(checkpoint_info, "base_model", None) or "unknown")
        self.sampling_client = self.service_client.create_sampling_client(self.sampler_checkpoint)
        self.tokenizer = self.sampling_client.get_tokenizer()
        self.renderer_name = select_renderer_name(self.base_model)
        self.renderer = None

    def _resolve_sampler_checkpoint(self) -> str:
        payload = find_latest_payload(self.workspace / "run_outputs", run_name=self.run_name)
        checkpoint = extract_sampler_checkpoint(payload)
        if not checkpoint:
            raise RuntimeError(
                f"No sampler checkpoint found for {self.run_name}. "
                "Pass --sampler-checkpoint or run post-train sampling first."
            )
        return checkpoint

    def chat(self, *, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        user_text = last_user_text(messages)
        example = ConversationExample(
            example_id="chat-request",
            opening_text=user_text,
            target_text="",
            messages=[
                {"role": "user", "content": build_chat_prompt(messages)},
                {"role": "assistant", "content": ""},
            ],
            metadata={"source": "endpoint_chat"},
        )
        if self.renderer is None:
            self.renderer, _ = build_datums(
                [example],
                self.tokenizer,
                self.base_model,
                renderer_name=self.renderer_name,
                max_length=512,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
        rows = sample_generations(
            self.sampling_client,
            self.renderer,
            [example],
            max_tokens=max(1, int(max_tokens)),
            temperature=float(temperature),
        )
        return str(rows[0].get("generated_text") or "").strip()


def build_chat_prompt(messages: list[dict[str, str]]) -> str:
    user_text = last_user_text(messages)
    system_text = "\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system" and str(message.get("content") or "").strip()
    )
    instructions = [
        "Write in the trained target voice.",
        "Preserve the user's opening or request when it is clearly a continuation prompt.",
        "Return only the final response text.",
    ]
    if system_text:
        instructions.append(f"System guidance: {system_text}")
    return " ".join(instructions) + "\n\nUser message:\n" + user_text


def last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = str(message.get("content") or "").strip()
            if text:
                return text
    return ""


def make_handler(endpoint: TinkerEndpoint):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TinkerStudioEndpoint/0.1"

        def do_GET(self) -> None:
            if self.path in {"/health", "/v1/health"}:
                self.write_json({"status": "ok", "model": endpoint.model_id, "sampler_checkpoint": endpoint.sampler_checkpoint})
                return
            if self.path == "/v1/models":
                self.write_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": endpoint.model_id,
                                "object": "model",
                                "created": 0,
                                "owned_by": "local-tinker",
                            }
                        ],
                    }
                )
                return
            self.write_error(404, "not found")

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.write_error(404, "not found")
                return
            try:
                payload = self.read_json()
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                output = endpoint.chat(
                    messages=messages,
                    temperature=float(payload.get("temperature", 0.7)),
                    max_tokens=int(payload.get("max_tokens", 128)),
                )
                created = int(time.time())
                self.write_json(
                    {
                        "id": f"chatcmpl-tinker-{created}",
                        "object": "chat.completion",
                        "created": created,
                        "model": payload.get("model") or endpoint.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": output},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            except Exception as exc:
                self.write_error(500, str(exc))

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            data = self.rfile.read(length).decode("utf-8")
            payload = json.loads(data) if data else {}
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_error(self, status: int, message: str) -> None:
            self.write_json({"error": {"message": message}}, status=status)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[endpoint] {self.address_string()} - {format % args}")

    return Handler


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    endpoint = TinkerEndpoint(
        workspace=workspace,
        run_name=args.run_name,
        sampler_checkpoint=args.sampler_checkpoint,
        model_id=args.model_id,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(endpoint))
    print(f"Serving {endpoint.model_id} at http://{args.host}:{args.port}/v1")
    print(f"Sampler checkpoint: {endpoint.sampler_checkpoint}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
