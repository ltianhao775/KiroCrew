"""Tests for the per-job debug_log flag on CronJob.

When debug_log=True, an AGENT (message) cron's tick is spawned with
KIRO_LOG_LEVEL=debug so its kiro-cli subprocess writes a full, real-time
transcript (reasoning + tool calls) to that session's chat log
(<scratch>/kiro-log/kiro-chat.log, which agent_scratch.scratch_env already
pins), for an operator to `tail -f`. The log PATH is not new state — this flag
only raises the child's verbosity. Default False (legacy jobs read as False, no
behaviour change); verbose (~40 MiB/min, bounded by cap_kiro_cli_logs). No-op
for script/command crons, which run no kiro-cli tick.
"""

from __future__ import annotations

import json

import pytest

from kiro_crew.cron import CronJob, CronService


@pytest.fixture(autouse=True)
def _isolate_cron_store(monkeypatch, tmp_path):
    monkeypatch.setattr("kiro_crew.cron._DEFAULT_DIR", tmp_path)
    yield


class TestCronJobDebugLogField:
    """Dataclass field + serialization round-trip."""

    def test_default_is_false(self):
        job = CronJob(id="j1", name="x", message="y")
        assert job.debug_log is False

    def test_field_roundtrips_through_save_load(self, tmp_path):
        svc = CronService()
        job = svc.add_job(name="digest", message="summarize", every_secs=86400)
        job.debug_log = True
        svc._save()

        svc2 = CronService()
        loaded = [j for j in svc2.list_jobs() if j.id == job.id][0]
        assert loaded.debug_log is True

    def test_false_roundtrips_through_save_load(self, tmp_path):
        svc = CronService()
        job = svc.add_job(name="quiet", message="run", every_secs=3600)
        job.debug_log = False
        svc._save()

        svc2 = CronService()
        loaded = [j for j in svc2.list_jobs() if j.id == job.id][0]
        assert loaded.debug_log is False

    def test_legacy_job_without_field_defaults_to_false(self, tmp_path):
        """A crons.json predating the flag must read as debug_log=False."""
        path = tmp_path / "crons.json"
        legacy = {
            "version": 2,
            "jobs": [
                {
                    "id": "legacy1",
                    "name": "old",
                    "message": "hello",
                    "schedule": {"kind": "every", "every_secs": 300},
                    "created_ts": 1_700_000_000.0,
                }
            ],
        }
        path.write_text(json.dumps(legacy))

        svc = CronService()
        loaded = [j for j in svc.list_jobs() if j.id == "legacy1"][0]
        assert loaded.debug_log is False

    def test_serialized_dict_includes_field(self, tmp_path):
        svc = CronService()
        job = svc.add_job(name="d", message="m", every_secs=600)
        job.debug_log = True
        svc._save()

        raw = json.loads((tmp_path / "crons.json").read_text(encoding="utf-8"))
        entry = [j for j in raw["jobs"] if j["id"] == job.id][0]
        assert entry["debug_log"] is True

    def test_add_job_accepts_flag(self, tmp_path):
        svc = CronService()
        job = svc.add_job(name="d", message="m", every_secs=600, debug_log=True)
        assert job.debug_log is True


class TestCronServiceUpdateDebugLog:
    """update_job applies debug_log."""

    def test_update_sets_true(self, tmp_path):
        svc = CronService()
        job = svc.add_job(name="d", message="m", every_secs=600)
        assert job.debug_log is False

        updated = svc.update_job(job.id, debug_log=True)
        assert updated is not None
        assert updated.debug_log is True

        svc2 = CronService()
        loaded = [j for j in svc2.list_jobs() if j.id == job.id][0]
        assert loaded.debug_log is True

    def test_update_back_to_false(self, tmp_path):
        svc = CronService()
        job = svc.add_job(name="d", message="m", every_secs=600)
        svc.update_job(job.id, debug_log=True)
        updated = svc.update_job(job.id, debug_log=False)
        assert updated is not None
        assert updated.debug_log is False


class TestLiveEventRecord:
    """live_event_record projects + redacts an LLMEvent into a trace record."""

    def _ev(self, **kw):
        from kiro_crew.acp.types import AcpEvent

        return AcpEvent(**kw)

    def test_tool_call_redacts_input_and_title(self):
        from kiro_crew.slack.gateway import live_event_record

        ev = self._ev(
            kind="tool_call",
            tool_call_id="c1",
            title="run shell",
            tool_input="echo AKIAIOSFODNN7EXAMPLE",
            tool_input_redacted=False,
        )
        rec = live_event_record(ev)
        assert rec is not None
        assert rec["t"] == "tool_call"
        assert rec["id"] == "c1"
        # credential-shaped token must not survive verbatim
        assert "AKIAIOSFODNN7EXAMPLE" not in rec["input"]

    def test_tool_result_never_persists_raw_output(self):
        from kiro_crew.slack.gateway import live_event_record

        ev = self._ev(
            kind="tool_result",
            tool_call_id="c1",
            tool_status="completed",
            tool_final=True,
            tool_output="super secret raw output body",
            tool_output_digest="abc123",
            tool_output_bytes=29,
        )
        rec = live_event_record(ev)
        assert rec is not None
        assert rec["t"] == "tool_result"
        assert rec["output_digest"] == "abc123"
        assert rec["output_bytes"] == 29
        # raw body must never appear anywhere in the record
        assert "super secret raw output body" not in json.dumps(rec)

    def test_complete_record(self):
        from kiro_crew.slack.gateway import live_event_record

        rec = live_event_record(self._ev(kind="complete", stop_reason="end_turn"))
        assert rec == {"t": "complete", "stop_reason": "end_turn", "refusal": False}

    def test_untraced_kinds_return_none(self):
        from kiro_crew.slack.gateway import live_event_record

        # text_chunk is aggregated by the tee, not emitted per-event here
        assert live_event_record(self._ev(kind="text_chunk", text="hi")) is None
        assert live_event_record(self._ev(kind="thinking_chunk", text="…")) is None


class TestCronHistoryLiveSink:
    """append_live_event / begin_live_trace are best-effort and off-loop."""

    def test_append_and_truncate_roundtrip(self, tmp_path):
        import asyncio

        from kiro_crew.cron_history import CronHistoryStore

        store = CronHistoryStore(base_dir=tmp_path)

        async def _run():
            await store.begin_live_trace("job1")
            await store.append_live_event("job1", {"t": "text", "text": "hello"})
            await store.append_live_event("job1", {"t": "complete", "stop_reason": "end_turn"})

        asyncio.run(_run())
        p = tmp_path / "cron-history" / "job1.live.jsonl"
        lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert [r["t"] for r in lines] == ["text", "complete"]

        # begin_live_trace truncates: a second run starts fresh
        async def _run2():
            await store.begin_live_trace("job1")
            await store.append_live_event("job1", {"t": "text", "text": "world"})

        asyncio.run(_run2())
        lines2 = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert [r["t"] for r in lines2] == ["text"]
        assert lines2[0]["text"] == "world"

    def test_disabled_store_is_silent(self, tmp_path):
        import asyncio

        from kiro_crew.cron_history import CronHistoryStore

        store = CronHistoryStore(base_dir=tmp_path)
        store._enabled = False  # simulate a degraded/unusable store

        async def _run():
            # Must not raise and must write nothing.
            await store.begin_live_trace("job1")
            await store.append_live_event("job1", {"t": "text", "text": "x"})

        asyncio.run(_run())
        assert not (tmp_path / "cron-history" / "job1.live.jsonl").exists()


class TestStreamAndCollectOnEvent:
    """stream_and_collect fires on_event for every event, best-effort."""

    def test_on_event_fires_for_every_event_and_swallows_errors(self):
        import asyncio

        from kiro_crew.acp.types import AcpEvent
        from kiro_crew.llm_helpers import stream_and_collect

        events = [
            AcpEvent(kind="text_chunk", text="hi "),
            AcpEvent(kind="tool_call", tool_call_id="c1", title="t"),
            AcpEvent(kind="tool_result", tool_call_id="c1", tool_status="completed"),
            AcpEvent(kind="complete", stop_reason="end_turn"),
        ]

        class _Provider:
            def stream(self, message):
                async def _gen():
                    for e in events:
                        yield e

                return _gen()

        seen = []

        def _obs(ev):
            seen.append(ev.kind)
            raise RuntimeError("observer boom")  # must be swallowed

        text = asyncio.run(stream_and_collect(_Provider(), "go", on_event=_obs))
        assert text == "hi "
        assert seen == ["text_chunk", "tool_call", "tool_result", "complete"]
