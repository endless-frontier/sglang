"""4P4D OpenAI router using worker token demand, KV budgets and dispatch credits.

Worker protocol is the pinned PD router's simultaneous P/D HTTP dispatch with
shared bootstrap rooms. Never stores bodies, prompts, completions or headers.
"""

import asyncio
import json
import logging
import os
import secrets
import time
from collections import Counter, deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from pd_control import (
    connection_error_metadata,
    connection_trace,
    normalize_messages,
    observe_admission_credit,
)
from pd_state import read_json, save

RUN = Path(__file__).resolve().parent
STATE = RUN / "state"


class Router:
    def __init__(self, run=RUN):
        self.run = Path(run)
        self.state = self.run / "state"
        self.settings = read_json(self.run / "deployment-settings.json")
        self.model = self.settings["model_name"]
        self.nodes = []
        self.credit = {}
        self.queue = 0
        self.condition = asyncio.Condition()
        self.tokenizer = None
        self.tok_lock = asyncio.Lock()
        self.tok_sem = asyncio.Semaphore(8)
        self.phase_cursors = {}
        self.counts = Counter()
        self.pairs = Counter()
        self.recent = deque(maxlen=256)
        self.audit = logging.getLogger("capacity_router." + self.model)
        self.audit.setLevel(logging.INFO)
        self.audit.propagate = False
        h = RotatingFileHandler(
            self.state / "routing-requests.jsonl",
            maxBytes=64 * 1024 * 1024,
            backupCount=3,
        )
        h.setFormatter(logging.Formatter("%(message)s"))
        self.audit.addHandler(h)

    def event(self, event, **meta):
        self.audit.info(
            json.dumps(dict(event=event, ts=time.time(), **meta), separators=(",", ":"))
        )

    async def estimate(self, body):
        begin = time.monotonic()
        timings = {}
        try:
            async with self.tok_lock:
                if self.tokenizer is None:
                    from transformers import AutoTokenizer

                    self.tokenizer = await asyncio.to_thread(
                        AutoTokenizer.from_pretrained,
                        str(self.run / "model"),
                        trust_remote_code=True,
                    )
        except Exception as e:
            self.event("token_estimator_unavailable", error_type=type(e).__name__)
            raise ValueError("tokenizer_unavailable")
        tokenizer_ready = time.monotonic()
        timings["tokenizer_ready_wait_s"] = tokenizer_ready - begin

        def count():
            timings["tokenizer_thread_start_mono"] = time.monotonic()
            if "messages" in body:
                kw = dict(body.get("chat_template_kwargs") or {})
                kw.pop("tokenize", None)
                kw.pop("add_generation_prompt", None)
                messages, normalized = normalize_messages(body["messages"])
                ids = self.tokenizer.apply_chat_template(
                    messages,
                    tools=body.get("tools"),
                    tokenize=True,
                    add_generation_prompt=True,
                    **kw,
                )
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                length = (
                    len(ids[0])
                    if ids and isinstance(ids[0], (list, tuple))
                    else len(ids)
                )
                return length * int(body.get("n", 1) or 1)
            prompt = body.get("prompt", "")
            prompts = (
                prompt
                if isinstance(prompt, list)
                and prompt
                and isinstance(prompt[0], (str, list))
                else [prompt]
            )
            return sum(
                len(self.tokenizer.encode(p, add_special_tokens=False))
                if isinstance(p, str)
                else len(p)
                for p in prompts
            ) * int(body.get("n", 1) or 1)

        try:
            sem_begin = time.monotonic()
            async with self.tok_sem:
                thread_submitted = time.monotonic()
                timings["tokenizer_semaphore_wait_s"] = thread_submitted - sem_begin
                result = max(1, await asyncio.to_thread(count))
                done = time.monotonic()
                start = timings.pop("tokenizer_thread_start_mono", thread_submitted)
                timings["tokenizer_thread_wait_s"] = max(0, start - thread_submitted)
                timings["tokenizer_compute_s"] = max(0, done - start)
                timings["tokenizer_total_s"] = done - begin
                return result, "hf_chat_template", timings
        except Exception as e:
            # Conservative fallback is observable, never silently called exact.
            n = min(
                1048576,
                max(64, len(json.dumps(body, ensure_ascii=False).encode()) // 2 + 1024),
            )
            self.event("token_estimator_fallback", error_type=type(e).__name__)
            start = timings.pop("tokenizer_thread_start_mono", None)
            if start is not None:
                timings["tokenizer_compute_s"] = max(0, time.monotonic() - start)
            timings["tokenizer_total_s"] = time.monotonic() - begin
            return n, "utf8_estimate", timings

    def pending_credit(self, rank):
        return sum(
            x["tokens"]
            for x in self.credit.values()
            if x["rank"] == rank and not x.get("observed")
        )

    def select(self, tokens):
        good = [
            n
            for n in self.nodes
            if n.get("healthy") and time.monotonic() - n.get("loaded_mono", 0) < 10
        ]
        ps = [n for n in good if n["rank"] < 4]
        ds = [n for n in good if n["rank"] >= 4]
        if not ps or not ds:
            return None
        # Include allocated KV, unallocated prealloc demand and unobserved dispatch.
        ds = [
            n
            for n in ds
            if n["load"]["num_total_tokens"]
            + self.pending_credit(n["rank"])
            + tokens
            + 64
            <= n["load"]["max_total_num_tokens"]
        ]
        if not ds:
            return None

        def pick(ns, score):
            values = [score(n) for n in ns]
            minimum = min(values)
            return secrets.choice(
                [n for n, v in zip(ns, values) if v <= minimum + 1e-6]
            )

        d = pick(
            ds,
            lambda n: (
                (n["load"]["num_total_tokens"] + self.pending_credit(n["rank"]))
                / n["load"]["max_total_num_tokens"]
            ),
        )
        p = pick(
            ps,
            lambda n: (
                n["load"]
                .get("pd_admission", {})
                .get("fair_prefill", {})
                .get("queued_work_units", n["load"]["num_waiting_uncached_tokens"])
                + self.pending_credit(n["rank"])
                + n["load"]["disaggregation"]["prefill_bootstrap_queue_reqs"] * 65536
            ),
        )
        return p, d

    async def reserve(self, tokens, rooms):
        begin = time.monotonic()
        async with self.condition:
            self.queue += 1
            try:
                while True:
                    selected = self.select(tokens)
                    if selected:
                        p, d = selected
                        for n in (p, d):
                            for room in rooms:
                                self.credit[(room, n["rank"])] = {
                                    "rank": n["rank"],
                                    "tokens": tokens / len(rooms),
                                    "observed": False,
                                    "dispatch_ts": time.time(),
                                }
                        return p, d, time.monotonic() - begin
                    await self.condition.wait()
            finally:
                self.queue -= 1

    def read_phases(self, n):
        p = self.state / f"request-phases-{n['rank']}.jsonl"
        try:
            s = p.stat()
            old = self.phase_cursors.get(n["rank"], (s.st_ino, 0))
            offset = old[1] if old[0] == s.st_ino else 0
            with p.open("rb") as f:
                f.seek(offset)
                data = f.read(1024 * 1024)
            cut = data.rfind(b"\n")
            if cut < 0:
                return
            self.phase_cursors[n["rank"]] = (s.st_ino, offset + cut + 1)
            for line in data[: cut + 1].splitlines():
                row = json.loads(line)
                c = self.credit.get((row["room"], n["rank"]))
                if c and row["phase"] == "admitted":
                    c["admitted_seen"] = time.monotonic()
                    c["admitted_ts"] = row["ts"]
        except (OSError, ValueError, KeyError):
            pass

    async def abort_sides(self, nodes, rooms):
        async def one(n, room):
            try:
                async with self.load_session.post(
                    n["url"] + "/abort_request",
                    json={"rid": f"pd-{room}", "abort_all": False},
                    timeout=ClientTimeout(total=4),
                ) as r:
                    await r.read()
                    self.event(
                        "abort_ack", room=room, rank=n["rank"], http_status=r.status
                    )
            except Exception as e:
                self.event(
                    "abort_ack_unavailable",
                    room=room,
                    rank=n["rank"],
                    error_type=type(e).__name__,
                )

        await asyncio.gather(*(one(n, room) for n in nodes for room in rooms))

    async def sample(self, n):
        # New workers acknowledge admission in their own HTTP load snapshot.
        start = time.monotonic()
        try:
            wd = read_json(self.state / f"watchdog-{n['rank']}.json")
            if (
                time.time() - wd["updated_at"] > 45
                or wd["worker"]["state"] != "HEALTHY"
            ):
                raise ValueError("watchdog_not_ready")
            sched = wd["scheduler_processes"]
            if len(sched) != 8 or any(x["state"] in ("Z", "X") for x in sched):
                raise ValueError("scheduler_not_alive")
            if not (self.state / f"ready-{n['rank']}.json").exists():
                raise ValueError("not_ready")
            async with self.load_session.get(
                n["url"] + "/v1/loads", timeout=ClientTimeout(total=4)
            ) as r:
                if r.status != 200:
                    raise ValueError("load_unavailable")
                load = (await r.json())["loads"][0]
            ack = load.get("pd_admission")
            if self.settings.get("require_worker_load_admission_ack") and (
                not isinstance(ack, dict)
                or ack.get("rank") != n["rank"]
                or ack.get("protocol") != "same-worker-admission-v1"
            ):
                raise ValueError("worker_admission_not_ready")
            n.update(load=load, loaded_mono=time.monotonic(), healthy=True)
            direct_ack = observe_admission_credit(self.credit, n["rank"], load)
            n["admission_source"] = (
                "worker_load_snapshot" if direct_ack else "cpfs_compatibility"
            )
            if not direct_ack:
                self.read_phases(n)
                for c in self.credit.values():
                    if (
                        c["rank"] == n["rank"]
                        and load.get("timestamp", 0)
                        > c.get("admitted_ts", float("inf")) + 0.001
                    ):
                        c["observed"] = True
        except Exception as e:
            n.update(healthy=False, error_type=type(e).__name__)

    async def monitor(self):
        last = 0
        while True:
            if len(self.nodes) < 8:
                found = []
                for rank in range(8):
                    p = self.state / f"node-{rank}.json"
                    if p.exists():
                        try:
                            n = read_json(p)
                        except (OSError, ValueError):
                            continue
                        n.update(url=f"http://{n['ip']}:{n['port']}", healthy=False)
                        found.append(n)
                self.nodes = found
            await asyncio.gather(*(self.sample(n) for n in self.nodes))
            async with self.condition:
                self.condition.notify_all()
            if time.monotonic() - last > 15:
                last = time.monotonic()
                save(self.state / "router-loads.json", self.status())
                self.event(
                    "load_snapshot",
                    workers=self.status()["workers"],
                    waiting_for_capacity=self.queue,
                )
            await asyncio.sleep(1)

    def status(self):
        return {
            "pid": os.getpid(),
            "updated_at": time.time(),
            "policy": "kv-capacity-and-prefill-work-v2",
            "model": self.model,
            "queued_for_capacity": self.queue,
            "inflight_side_requests": len(self.credit),
            "counts": dict(self.counts),
            "pair_counts": dict(self.pairs),
            "recent": list(self.recent),
            "workers": [
                {
                    "rank": n["rank"],
                    "url": n["url"],
                    "is_healthy": n.get("healthy", False),
                    "pending_credit_tokens": self.pending_credit(n["rank"]),
                    "admission_source": n.get("admission_source"),
                    "load": n.get("load"),
                }
                for n in self.nodes
            ],
        }

    async def handler(self, request):
        if request.method == "GET":
            if request.path == "/health":
                healthy = any(
                    n.get("healthy") for n in self.nodes if n["rank"] < 4
                ) and any(n.get("healthy") for n in self.nodes if n["rank"] >= 4)
                return web.json_response(
                    {"healthy": healthy}, status=200 if healthy else 503
                )
            if request.path in ("/workers", "/router_status"):
                return web.json_response(self.status())
            if request.path == "/v1/models":
                return web.json_response(
                    {
                        "object": "list",
                        "data": [
                            {"id": self.model, "object": "model", "owned_by": "sglang"}
                        ],
                    }
                )
            return web.json_response(
                {"error": {"message": "Unknown endpoint"}}, status=404
            )
        if request.method != "POST" or request.path not in (
            "/v1/chat/completions",
            "/v1/completions",
        ):
            return web.json_response(
                {"error": {"message": "Use OpenAI chat/completions or completions"}},
                status=404,
            )
        received_ts = time.time()
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("object_required")
            if body.get("model", self.model) != self.model:
                raise ValueError("wrong_model")
            tokens, estimator, estimator_timings = await self.estimate(body)
            batch = int(body.get("n", 1) or 1) * (
                len(body["prompt"])
                if isinstance(body.get("prompt"), list)
                and body["prompt"]
                and isinstance(body["prompt"][0], str)
                else 1
            )
            if batch < 1:
                raise ValueError("invalid_batch")
            capacities = [
                x["load"]["max_total_num_tokens"]
                for x in self.nodes
                if x["rank"] >= 4 and x.get("load")
            ]
            if capacities and tokens + 64 > max(capacities):
                raise ValueError("request_exceeds_single_worker_capacity")
        except (ValueError, KeyError, TypeError):
            return web.json_response(
                {"error": {"message": "Invalid model or request format"}}, status=400
            )
        rooms = [secrets.randbits(53) for _ in range(batch)]
        p, d, queued = await self.reserve(tokens, rooms)
        room = rooms[0]
        record = {
            "room": room,
            "rooms": rooms,
            "P": p["rank"],
            "D": d["rank"] - 4,
            "input_tokens_estimate": tokens,
            "estimator": estimator,
            "estimator_timings": estimator_timings,
            "queued_s": queued,
            "stream": bool(body.get("stream")),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens")),
            "received_ts": received_ts,
            "dispatch_ts": time.time(),
        }
        self.event("dispatch", **record)
        self.counts["dispatched"] += 1
        self.pairs[f"P{p['rank']}-D{d['rank'] - 4}"] += 1
        prepared = dict(
            body,
            bootstrap_host=p["ip"],
            bootstrap_port=p["bootstrap_port"],
            bootstrap_room=rooms if batch > 1 else room,
            model=self.model,
            rid=f"pd-{room}",
        )
        pref = dict(prepared, stream=False)
        tasks = []
        responses = []
        downstream = None
        record["side_connections"] = {}
        try:

            async def send(n, payload):
                side = "P" if n["rank"] < 4 else "D"
                ctx = {
                    "side": side,
                    "rank": n["rank"],
                    "stage": "before_send",
                    "connection_reused": False,
                }
                record["side_connections"][side] = ctx
                try:
                    r = await self.session.post(
                        n["url"] + request.path, json=payload, trace_request_ctx=ctx
                    )
                    ctx["stage"] = "response_headers_received"
                    responses.append(r)
                    return r
                except Exception as error:
                    record.update(connection_error_metadata(error, side, ctx["stage"]))
                    raise

            async def prefill():
                r = await send(p, pref)
                try:
                    record["side_connections"]["P"]["stage"] = "response_body"
                    raw = await r.read()
                    record["side_connections"]["P"]["stage"] = "response_complete"
                except Exception as error:
                    record.update(
                        connection_error_metadata(error, "P", "response_body")
                    )
                    raise
                record["P_completed_ts"] = time.time()
                if r.status < 400:
                    pdata = json.loads(raw)
                    record["P_usage"] = pdata.get("usage")
                    self.event(
                        "P_response_complete", room=room, usage=record["P_usage"]
                    )
                return r, raw

            tasks = [
                asyncio.create_task(prefill()),
                asyncio.create_task(send(d, prepared)),
            ]
            todo = set(tasks)
            while todo:
                done, todo = await asyncio.wait(
                    todo, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    value = t.result()
                    r = value[0] if t is tasks[0] else value
                    if r.status >= 400:
                        raw = value[1] if t is tasks[0] else await r.read()
                        record["http_status"] = r.status
                        self.counts["worker_rejected"] += 1
                        return web.Response(
                            body=raw, status=r.status, content_type="application/json"
                        )
            (pr, _), dr = [t.result() for t in tasks]
            if prepared.get("stream"):
                downstream = web.StreamResponse(
                    status=dr.status,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    },
                )
                await downstream.prepare(request)
                lines = bytearray()
                async for chunk in dr.content.iter_any():
                    await downstream.write(chunk)
                    lines.extend(chunk)
                    while b"\n" in lines:
                        i = lines.index(b"\n")
                        line = bytes(lines[:i]).strip()
                        del lines[: i + 1]
                        if line == b"data: [DONE]":
                            record["sse_done"] = True
                        elif line.startswith(b"data: "):
                            try:
                                x = json.loads(line[6:])
                                if x.get("usage"):
                                    record["D_usage"] = x["usage"]
                                if "first_token_ts" not in record and any(
                                    c.get("delta", {}).get("content")
                                    or c.get("delta", {}).get("reasoning_content")
                                    or c.get("delta", {}).get("tool_calls")
                                    for c in x.get("choices", [])
                                ):
                                    record["first_token_ts"] = time.time()
                            except (ValueError, TypeError):
                                pass
                    if len(lines) > 4 * 1024 * 1024:
                        lines.clear()
                await downstream.write_eof()
                record["http_status"] = dr.status
                return downstream
            record["side_connections"]["D"]["stage"] = "response_body"
            raw = await dr.read()
            record["side_connections"]["D"]["stage"] = "response_complete"
            result = json.loads(raw)
            record["D_usage"] = result.get("usage")
            record["http_status"] = dr.status
            return web.Response(
                body=raw, status=dr.status, content_type="application/json"
            )
        except asyncio.CancelledError:
            record["cancelled"] = True
            self.counts["cancelled"] += 1
            raise
        except Exception as e:
            record["error_type"] = type(e).__name__
            self.counts["errors"] += 1
            if "failed_side" not in record:
                record.update(connection_error_metadata(e, "D", "response_body"))
            if downstream is not None:
                if request.transport:
                    request.transport.close()
                return downstream
            return web.json_response(
                {
                    "error": {
                        "message": "Paired worker request failed",
                        "type": "deployment_error",
                    }
                },
                status=502,
            )
        finally:
            if (
                record.get("cancelled")
                or record.get("error_type")
                or record.get("http_status", 200) >= 400
            ):
                await self.abort_sides((p, d), rooms)
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for r in responses:
                r.close()
            for n in (p, d):
                for rm in rooms:
                    self.credit.pop((rm, n["rank"]), None)
            record["finished_ts"] = time.time()
            record["duration_s"] = record["finished_ts"] - record["received_ts"]
            self.recent.append(record)
            self.counts["finished"] += 1
            self.event("finished", **record)
            async with self.condition:
                self.condition.notify_all()

    async def startup(self, app):
        self.session = ClientSession(
            connector=TCPConnector(limit=0, force_close=True),
            timeout=ClientTimeout(total=None, sock_connect=10, sock_read=7200),
            trust_env=False,
            trace_configs=[connection_trace()],
        )
        self.load_session = ClientSession(
            connector=TCPConnector(limit=0, keepalive_timeout=2),
            timeout=ClientTimeout(total=4),
            trust_env=False,
        )
        self.task = asyncio.create_task(self.monitor())

    async def cleanup(self, app):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.session.close()
        await self.load_session.close()


def main():
    router = Router()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{path:.*}", router.handler)
    app.on_startup.append(router.startup)
    app.on_cleanup.append(router.cleanup)
    web.run_app(
        app,
        host="0.0.0.0",
        port=router.settings["router_port"],
        access_log=None,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
