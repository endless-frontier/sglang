"""Request normalization and metadata-only admission accounting."""

import json
import math

from aiohttp import TraceConfig


def normalize_messages(messages):
    """Match the pinned OpenAI worker's strict tool-argument normalization.

    Copy only the container objects changed here; request content is not logged
    and the original payload sent to P/D is never modified.
    """
    result = []
    normalized = 0
    for original in messages:
        message = dict(original)
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(calls, list):
            calls = [dict(c) if isinstance(c, dict) else c for c in calls]
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                function = dict(function)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool_arguments_must_be_json_object")
                    function["arguments"] = arguments
                    normalized += 1
                call["function"] = function
            message["tool_calls"] = calls
        result.append(message)
    return result, normalized


def observe_admission_credit(credits, rank, load):
    """Only retire credit covered by a later snapshot from that same worker.

    A missing/truncated admission ring leaves reservations in place; it never
    causes speculative retirement or rejection of a request.
    """
    ack = load.get("pd_admission")
    if not isinstance(ack, dict) or ack.get("rank") != rank:
        return False
    snapshot_ts = load.get("timestamp", 0)
    if not isinstance(snapshot_ts, (int, float)) or not math.isfinite(snapshot_ts):
        return False
    seen = {}
    for item in ack.get("recent", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        room, ts = item
        if isinstance(room, int) and isinstance(ts, (int, float)) and math.isfinite(ts):
            seen[room] = ts
    for (room, worker_rank), credit in credits.items():
        if worker_rank != rank or credit.get("rank") != rank or room not in seen:
            continue
        admitted_ts = seen[room]
        # Both timestamps are produced by this worker. Never compare the
        # worker's wall clock with the router's wall clock for acknowledgment.
        if snapshot_ts > admitted_ts + 0.001:
            credit.update(
                observed=True,
                admitted_ts=admitted_ts,
                observed_snapshot_ts=snapshot_ts,
                admission_source="worker_load_snapshot",
            )
    return True


def connection_error_metadata(error, side, stage):
    """Never log exception text, URL/query, headers or payloads."""
    cause = getattr(error, "os_error", None)
    if cause is None:
        cause = getattr(error, "__cause__", None)
    errno = getattr(error, "errno", None)
    if errno is None:
        errno = getattr(cause, "errno", None)
    return {
        "failed_side": side,
        "failed_stage": stage,
        "connection_error_type": type(error).__name__,
        "connection_errno": errno if isinstance(errno, int) else None,
        "connection_cause_type": type(cause).__name__ if cause is not None else None,
    }


def connection_trace():
    trace = TraceConfig()

    async def update(ctx, **values):
        if isinstance(ctx.trace_request_ctx, dict):
            ctx.trace_request_ctx.update(values)

    async def start(session, ctx, params):
        await update(ctx, stage="request_start", connection_reused=False)

    async def create_start(session, ctx, params):
        await update(ctx, stage="tcp_connect")

    async def create_end(session, ctx, params):
        await update(ctx, stage="connected")

    async def reused(session, ctx, params):
        await update(ctx, connection_reused=True, stage="reuse_connection")

    async def headers(session, ctx, params):
        await update(ctx, stage="request_headers_sent")

    async def sent(session, ctx, params):
        await update(ctx, stage="request_body_sent")

    trace.on_request_start.append(start)
    trace.on_connection_create_start.append(create_start)
    trace.on_connection_create_end.append(create_end)
    trace.on_connection_reuseconn.append(reused)
    trace.on_request_headers_sent.append(headers)
    trace.on_request_chunk_sent.append(sent)
    return trace
