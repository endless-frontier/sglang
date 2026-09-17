#!/usr/bin/env python3
"""Qwen3.8-Flash-Next 1M PD 压测客户端：测 TTFT / 端到端 / 输出 TPS + KV pool 信息。

用法（在任意能访问 router 的节点上跑，需要能读到模型目录的 tokenizer）：
  python3 bench_pd.py --input-tokens 300000 --output-tokens 10000
  python3 bench_pd.py --input-tokens 800000 --output-tokens 10000 --url http://10.0.1.127:40000

输出：单行 JSON + 可读摘要。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

BASE_TEXT = (
    "Large language models have changed how engineers approach retrieval, reasoning and code "
    "generation. A production serving stack must juggle prefill throughput, decode latency, KV "
    "cache capacity and fault tolerance. This paragraph is repeated to synthesise a long prompt "
    "whose length is controlled by the benchmark harness. "
    "大语言模型推理服务需要同时兼顾预填充吞吐、解码延迟、KV 缓存容量与容错能力，"
    "这一段文字会被反复拼接，用来构造指定长度的长上下文输入。 "
)


def build_prompt_text(tokenizer, n_tokens: int) -> tuple[str, int]:
    ids = tokenizer.encode(BASE_TEXT, add_special_tokens=False)
    if not ids:
        raise SystemExit("tokenizer 返回空 token 序列")
    if n_tokens <= len(ids):
        text = tokenizer.decode(ids[:n_tokens])
        return text, len(ids[:n_tokens])
    repeat = n_tokens // len(ids) + 1
    ids = ids * repeat
    ids = ids[:n_tokens]
    return tokenizer.decode(ids), len(ids)


def stream_request(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:40000",
                    help="router 或 worker 地址（默认本机 router 40000）")
    ap.add_argument("--model", default="qwen38-flash-next-1m")
    ap.add_argument("--input-tokens", type=int, required=True)
    ap.add_argument("--output-tokens", type=int, default=10000)
    ap.add_argument("--tokenizer", default="/mnt/data/public_data/public_model/Qwen3.8/Qwen3.8-Flash-Next-1M")
    ap.add_argument("--reasoning-effort", default="low", choices=["xhigh", "medium", "low", "off"])
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--ignore-eos", type=int, default=1, choices=[0, 1],
                    help="1=强制生成到 max_tokens（压测用），0=允许模型自然停止")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt_text, client_tokens = build_prompt_text(tok, args.input_tokens)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": args.output_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.reasoning_effort == "off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        payload["reasoning_effort"] = args.reasoning_effort

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    label = args.label or f"in={args.input_tokens} out={args.output_tokens}"
    print(f"[bench] {label}: 客户端构造 {client_tokens} token，POST {endpoint}", file=sys.stderr)

    t0 = time.perf_counter()
    resp = stream_request(endpoint, payload, args.timeout)

    t_first = None
    n_chunks = 0
    finish_reason = None
    completion_tokens = None
    prompt_tokens = None
    first_chunk_t = None
    last_t = t0
    first_delta_t = None

    buf = b""
    for raw in resp:
        last_t = time.perf_counter()
        buf += raw
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if first_chunk_t is None:
                first_chunk_t = last_t
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    n_chunks += 1
                    if t_first is None:
                        t_first = last_t
                        first_delta_t = last_t

    t_end = last_t
    ttft = (t_first or t_end) - t0
    gen_window = max(t_end - (t_first or t_end), 1e-6)
    out_tokens = completion_tokens if completion_tokens else n_chunks
    tps = (out_tokens - 1) / gen_window if out_tokens > 1 else float("nan")
    wall = out_tokens / max(t_end - t0, 1e-6)

    result = {
        "label": label,
        "input_tokens_client": client_tokens,
        "prompt_tokens_server": prompt_tokens,
        "completion_tokens": out_tokens,
        "ttft_s": round(ttft, 3),
        "e2e_s": round(t_end - t0, 3),
        "decode_tps": round(tps, 1),
        "wall_tps": round(wall, 1),
        "chunks": n_chunks,
        "finish_reason": finish_reason,
        "first_chunk_s": round((first_chunk_t - t0), 3) if first_chunk_t else None,
    }
    print(json.dumps(result, ensure_ascii=False))
    print(
        f"[bench] {label}: prompt={prompt_tokens}(server)/{client_tokens}(client) "
        f"completion={out_tokens} TTFT={result['ttft_s']}s "
        f"e2e={result['e2e_s']}s decode={result['decode_tps']} tok/s wall={result['wall_tps']} tok/s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
