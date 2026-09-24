"""Tests for the cron reaper that force-kills zombie cron jobs."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.cron import (
    _JOB_TIMEOUT_SECS,
    CronJob,
    CronSchedule,
    CronService,
    _RunClaim,
)
from kiro_crew.cron_history import CronHistoryStore


def _mock_sessions() -> MagicMock:
    sessions = MagicMock()
    sessions.reset = AsyncMock()
    sessions._sessions = {}
    return sessions


def _live_task() -> MagicMock:
    """A tracked task that has not finished (the reaper's deadline path)."""
    return MagicMock(done=MagicMock(return_value=False))


def _make_job(job_id: str = "job1", name: str = "test job") -> CronJob:
    return CronJob(
        id=job_id,
        name=name,
        message="do something",
        schedule=CronSchedule(kind="every", every_secs=300),
        created_ts=time.time(),
    )


class TestCronReaper:
    """Tests for the periodic reaper that force-kills zombie cron jobs."""

    @pytest.mark.asyncio
    async def test_reaper_kills_expired_job(self, tmp_path: object) -> None:
        """Reaper marks expired job as error and emits SEL event."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        sessions = _mock_sessions()
        svc._sessions = sessions

        job = _make_job("expired1")
        svc._jobs = [job]
        claim = svc._claims["expired1"] = _RunClaim(
            trigger="scheduled",
            claimed_at=time.time() - _JOB_TIMEOUT_SECS - 120,
            task=_live_task(),
        )

        with patch("kiro_crew.sel.sel") as mock_sel, patch.object(svc, "_save"):
            await svc._force_reap("expired1", _JOB_TIMEOUT_SECS + 120, claim=claim)

        assert job.last_status == "error"
        assert "Reaped" in (job.last_error or "")
        assert svc._reaped_jobs.has("expired1", claim)
        assert "expired1" not in svc._claims  # released
        # ``ends_conversation``: the reaper has given up on the run, so its conversation
        # is over and its sub-agent runs end with it. Asserting the whole call keeps a
        # later edit from dropping that and leaving a reaped job's children running.
        sessions.reset.assert_awaited_once_with("cron:expired1", ends_conversation=True)
        mock_sel().log_tool_invocation.assert_called_once_with(
            session_key="cron:expired1",
            source="cron",
            tool_name="reaper_force_kill",
            outcome="reaped",
            metadata={
                "job_id": "expired1",
                "session_key": "cron:expired1",
                "elapsed": _JOB_TIMEOUT_SECS + 120,
            },
        )

    @pytest.mark.asyncio
    async def test_reaper_skips_jobs_within_deadline(self) -> None:
        """Reaper does not touch jobs still within the timeout."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        sessions = _mock_sessions()
        svc._sessions = sessions

        svc._claims["ok1"] = _RunClaim(trigger="scheduled", claimed_at=time.time() - 60)  # 60s old

        with patch("kiro_crew.sel.sel"), patch(
            "asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        # Should not have been reaped
        assert not svc._reaped_jobs._marks  # nothing reaped
        sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reaper_skips_done_tasks(self) -> None:
        """Reaper skips jobs whose asyncio task already completed (race guard)."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        done_task = MagicMock()
        done_task.done.return_value = True
        svc._claims["done1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60, task=done_task
        )

        with patch("kiro_crew.sel.sel"), patch(
            "asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert not svc._reaped_jobs._marks  # nothing reaped
        assert "done1" not in svc._claims  # cleaned up

    @pytest.mark.asyncio
    async def test_reaper_handles_reset_timeout(self, tmp_path: object) -> None:
        """Reaper falls back to SIGKILL when reset() hangs."""
        sessions = _mock_sessions()

        async def hanging_reset(key: str) -> None:
            await asyncio.sleep(999)

        sessions.reset = hanging_reset

        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = sessions

        job = _make_job("hang1")
        svc._jobs = [job]
        claim = svc._claims["hang1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60, task=_live_task()
        )

        with patch("kiro_crew.sel.sel"), patch(
            "kiro_crew.cron._REAPER_RESET_TIMEOUT", 0.05
        ), patch.object(
            svc, "_sigkill_session", new_callable=AsyncMock
        ) as mock_kill, patch.object(svc, "_save"):
            await svc._force_reap("hang1", _JOB_TIMEOUT_SECS + 60, claim=claim)

        assert job.last_status == "error"
        mock_kill.assert_awaited_once_with("cron:hang1")

    @pytest.mark.asyncio
    async def test_reaper_sigkill_on_reset_exception(self, tmp_path: object) -> None:
        """Reaper falls back to SIGKILL when reset() raises a non-timeout exception."""
        sessions = _mock_sessions()
        sessions.reset = AsyncMock(side_effect=RuntimeError("broken"))

        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = sessions

        job = _make_job("exc1")
        svc._jobs = [job]
        claim = svc._claims["exc1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 10, task=_live_task()
        )

        with patch("kiro_crew.sel.sel"), patch.object(
            svc, "_sigkill_session", new_callable=AsyncMock
        ) as mock_kill, patch.object(svc, "_save"):
            await svc._force_reap("exc1", _JOB_TIMEOUT_SECS + 10, claim=claim)

        assert job.last_status == "error"
        mock_kill.assert_awaited_once_with("cron:exc1")

    @pytest.mark.asyncio
    async def test_reaper_cancels_asyncio_task(self, tmp_path: object) -> None:
        """Reaper cancels the running asyncio task for the job."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("cancel1")
        svc._jobs = [job]
        mock_task = MagicMock()
        mock_task.done.return_value = False
        claim = svc._claims["cancel1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 10, task=mock_task
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"):
            await svc._force_reap("cancel1", _JOB_TIMEOUT_SECS + 10, claim=claim)

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_reaper_persists_state(self, tmp_path: object) -> None:
        """Reaper calls _save() after updating job state."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("persist1")
        svc._jobs = [job]
        claim = svc._claims["persist1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 10, task=_live_task()
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save") as mock_save:
            await svc._force_reap("persist1", _JOB_TIMEOUT_SECS + 10, claim=claim)

        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_reaped_flag_prevents_merge(self, tmp_path: object) -> None:
        """When reaper kills a job, _run_job_isolated skips _merge_job_result."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("reaped1")
        svc._jobs = [job]
        claim = svc._claim_run("reaped1", "scheduled")
        svc._reaped_jobs.mark("reaped1", claim)

        with patch.object(svc, "_execute_with_timeout", new_callable=AsyncMock), patch.object(
            svc, "_merge_job_result"
        ) as mock_merge:
            await svc._run_job_isolated(job, claim)

        mock_merge.assert_not_called()
        assert not svc._reaped_jobs.has("reaped1", claim)  # cleaned up

    @pytest.mark.asyncio
    async def test_reaped_flag_prevents_merge_on_cancel(self) -> None:
        """Reaped job skips merge even when CancelledError propagates."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("reaped2")
        svc._jobs = [job]
        claim = svc._claim_run("reaped2", "scheduled")
        svc._reaped_jobs.mark("reaped2", claim)

        with patch.object(
            svc, "_execute_with_timeout", side_effect=asyncio.CancelledError
        ), patch.object(svc, "_merge_job_result") as mock_merge:
            with pytest.raises(asyncio.CancelledError):
                await svc._run_job_isolated(job, claim)

        mock_merge.assert_not_called()
        assert not svc._reaped_jobs.has("reaped2", claim)

    @pytest.mark.asyncio
    async def test_non_reaped_job_merges_normally(self) -> None:
        """Normal (non-reaped) job still merges results."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("normal1")

        with patch.object(svc, "_execute_with_timeout", new_callable=AsyncMock), patch.object(
            svc, "_merge_job_result"
        ) as mock_merge:
            await svc._run_job_isolated(job, svc._claim_run("normal1", "scheduled"))

        mock_merge.assert_called_once()
        (record,) = mock_merge.call_args.args
        # The record is the job as the run left it, plus this run's own
        # generation, which only a merge ever writes to the job.
        assert record.run_generation == 1
        assert job.run_generation == 0
        record.run_generation = 0
        assert record == job

    @pytest.mark.asyncio
    async def test_start_reaper_creates_task(self) -> None:
        """start_reaper creates a background asyncio task."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        sessions = _mock_sessions()

        svc.start_reaper(sessions)
        assert svc._reaper_task is not None
        assert svc._sessions is sessions

        # Cleanup
        svc._reaper_task.cancel()
        try:
            await svc._reaper_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_stop_cancels_reaper(self) -> None:
        """stop() cancels the reaper task."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc.start_reaper(_mock_sessions())
        assert svc._reaper_task is not None

        await svc.stop()
        assert svc._reaper_task is None

    @pytest.mark.asyncio
    async def test_force_reap_without_sessions(self, tmp_path: object) -> None:
        """_force_reap handles missing sessions gracefully."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = None

        job = _make_job("nosess1")
        svc._jobs = [job]
        claim = svc._claims["nosess1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 10
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"):
            await svc._force_reap("nosess1", _JOB_TIMEOUT_SECS + 10, claim=claim)

        assert job.last_status == "error"
        assert svc._reaped_jobs.has("nosess1", claim)

    @pytest.mark.asyncio
    async def test_job_start_time_tracked(self) -> None:
        """_run_job_isolated records and cleans up start time."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        job = _make_job("track1")

        start_captured: list[bool] = []

        async def capture_start(j: CronJob, claim: object = None) -> None:
            start_captured.append(svc.running_since("track1") is not None)

        with patch.object(svc, "_execute_with_timeout", side_effect=capture_start), patch.object(
            svc, "_merge_job_result"
        ):
            await svc._run_job_isolated(job, svc._claim_run("track1", "scheduled"))

        assert start_captured == [True]  # was tracked during execution
        assert "track1" not in svc._claims  # cleaned up after

    @pytest.mark.asyncio
    async def test_reaper_loop_invokes_force_reap_for_expired_job(self) -> None:
        """Reaper loop calls _force_reap for an expired, non-done job."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        svc._claims["exp1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60, task=_live_task()
        )

        with patch.object(
            svc, "_force_reap", new_callable=AsyncMock
        ) as mock_reap, patch(
            "asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        mock_reap.assert_awaited_once()
        assert mock_reap.call_args[0][0] == "exp1"

    @pytest.mark.asyncio
    async def test_force_reap_cleans_up_executing_and_running_tasks(self, tmp_path: object) -> None:
        """_force_reap releases the job's claim -- tracked task and all -- directly."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("cleanup1")
        svc._jobs = [job]
        mock_task = MagicMock(done=MagicMock(return_value=False))
        claim = svc._claims["cleanup1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 10, task=mock_task
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"):
            await svc._force_reap("cleanup1", _JOB_TIMEOUT_SECS + 10, claim=claim)

        assert "cleanup1" not in svc._claims
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_reaper_respects_custom_timeout_secs(self) -> None:
        """Reaper does not kill a job still within its custom timeout_secs."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("custom1")
        job.timeout_secs = 5400  # 90 min
        svc._jobs = [job]
        # Running for 2000s — past default 1800 but within custom 5400
        svc._claims["custom1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - 2000, task=_live_task()
        )

        with patch("asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert not svc._reaped_jobs._marks  # nothing reaped

    @pytest.mark.asyncio
    async def test_reaper_kills_job_exceeding_custom_timeout(self, tmp_path: object) -> None:
        """Reaper kills a job that exceeds its custom timeout_secs."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("custom2")
        job.timeout_secs = 5400
        svc._jobs = [job]
        claim = svc._claims["custom2"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - 5500, task=_live_task()
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"), patch(
            "asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert svc._reaped_jobs.has("custom2", claim)
        assert job.last_status == "error"
        assert "exceeded 5400s deadline" in (job.last_error or "")

    @pytest.mark.asyncio
    async def test_reaper_enforces_floor_for_low_timeout(self) -> None:
        """Reaper uses _JOB_TIMEOUT_SECS as floor even if job.timeout_secs is lower."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("floor1")
        job.timeout_secs = 600  # below 1800 floor
        svc._jobs = [job]
        # Running for 1000s — past job.timeout_secs but within floor
        svc._claims["floor1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - 1000, task=_live_task()
        )

        with patch("asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert not svc._reaped_jobs._marks  # nothing reaped

    @pytest.mark.asyncio
    async def test_reaper_caps_at_86400(self, tmp_path: object) -> None:
        """Reaper caps deadline at 86400 even if job.timeout_secs exceeds it."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("cap1")
        job.timeout_secs = 100000  # exceeds 86400 cap
        svc._jobs = [job]
        claim = svc._claims["cap1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - 86500, task=_live_task()
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"), patch(
            "asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert svc._reaped_jobs.has("cap1", claim)
        assert "exceeded 86400s deadline" in (job.last_error or "")

    @pytest.mark.asyncio
    async def test_reaper_falls_back_for_deleted_job(self, tmp_path: object) -> None:
        """Reaper uses the default deadline when the job is absent from self._jobs."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        # No job in self._jobs, but the claim still stands (race: job removed while running)
        svc._jobs = []
        claim = svc._claims["ghost1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60, task=_live_task()
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"), patch(
            "asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert svc._reaped_jobs.has("ghost1", claim)


def _one_sweep() -> Any:
    """Patch ``asyncio.sleep`` so ``_reaper_loop`` runs one sweep then unwinds."""
    return patch("asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError]))


class TestReaperMonotonicDeadline:
    """The backstop must measure elapsed runtime on the same clock as ``wait_for``.

    ``_execute_with_timeout`` arms ``asyncio.wait_for``, whose deadline runs on
    the event loop's monotonic clock. If the reaper decides on the wall clock the
    two deadlines disagree whenever the wall clock jumps (host suspend, NTP step)
    and the backstop pre-empts a run the primary path considers healthy.
    """

    @pytest.mark.asyncio
    async def test_reaper_ignores_wallclock_jump_from_host_sleep(self) -> None:
        """A wall-clock start older than the deadline is not reaped while the
        monotonic runtime is still short (the host slept mid-run)."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("slept1")
        svc._jobs = [job]
        # Wall clock says the job has been running past its deadline only
        # because the host was suspended; it has actually executed for 60s.
        svc._claims["slept1"] = _RunClaim(
            trigger="scheduled",
            claimed_at=time.time() - _JOB_TIMEOUT_SECS - 600,
            started_monotonic=time.monotonic() - 60,
            task=_live_task(),
        )

        with patch.object(svc, "_force_reap", new_callable=AsyncMock) as mock_reap, _one_sweep():
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        mock_reap.assert_not_awaited()
        assert not svc._reaped_jobs._marks  # nothing reaped

    @pytest.mark.asyncio
    async def test_reaper_still_kills_genuine_overrun_on_monotonic_clock(self) -> None:
        """A job whose monotonic runtime exceeds the deadline is still reaped."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        job = _make_job("over1")
        svc._jobs = [job]
        svc._claims["over1"] = _RunClaim(
            trigger="scheduled",
            claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60,
            started_monotonic=time.monotonic() - _JOB_TIMEOUT_SECS - 60,
            task=_live_task(),
        )

        with patch.object(svc, "_force_reap", new_callable=AsyncMock) as mock_reap, _one_sweep():
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        mock_reap.assert_awaited_once()
        assert mock_reap.call_args[0][0] == "over1"

    @pytest.mark.asyncio
    async def test_run_job_isolated_stamps_and_clears_monotonic_start(self) -> None:
        """_run_job_isolated stamps the monotonic start on the claim it releases."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        job = _make_job("mono1")
        claim = svc._claim_run("mono1", "scheduled")

        seen: list[bool] = []

        async def capture(j: CronJob, claim_arg: object = None) -> None:
            seen.append(claim.started_monotonic is not None)

        with patch.object(svc, "_merge_job_result"):
            with patch.object(svc, "_execute_with_timeout", side_effect=capture):
                await svc._run_job_isolated(job, claim)

        assert seen == [True]  # was tracked during execution
        assert "mono1" not in svc._claims  # cleaned up after

    @pytest.mark.asyncio
    async def test_force_reap_clears_monotonic_start(self, tmp_path: object) -> None:
        """_force_reap pops the claim, monotonic stamp and all, so a reap never repeats."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("reap1")
        svc._jobs = [job]
        claim = svc._claims["reap1"] = _RunClaim(
            trigger="scheduled",
            claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60,
            started_monotonic=time.monotonic() - _JOB_TIMEOUT_SECS - 60,
            task=_live_task(),
        )

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"):
            await svc._force_reap("reap1", _JOB_TIMEOUT_SECS + 60, claim=claim)

        assert "reap1" not in svc._claims

    @pytest.mark.asyncio
    async def test_force_reap_releases_the_jitter_stamp_its_fenced_finalizer_skips(
        self, tmp_path: object
    ) -> None:
        """A reaped run's jitter stamp is released by the reap itself.

        ``_run_job_isolated`` stamps the jitter on its claim and its ``finally``
        releases the claim only while the run still holds it (identity).
        ``_force_reap`` takes that claim before the finalizer runs, so the fence
        is False for a reaped run and the finalizer leaves it alone; the reap
        has to release the claim itself, or a one-shot that is reaped and then
        removed keeps its entry for the process lifetime.
        """
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("reapjit1")
        svc._jobs = [job]
        claim = svc._claims["reapjit1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60
        )

        async def reap_mid_run(job_arg: CronJob, claim_arg: Any = None) -> None:
            # The sweep fires while this run is in flight: it takes the claim
            # and cancels the task, whose CancelledError lands here.
            assert claim.jitter is not None  # stamped by the run itself
            await svc._force_reap("reapjit1", _JOB_TIMEOUT_SECS + 60, claim=claim)
            raise asyncio.CancelledError

        with (
            patch("kiro_crew.sel.sel"),
            patch.object(svc, "_save"),
            patch.object(svc, "_execute_with_timeout", side_effect=reap_mid_run),
            patch.object(svc, "_merge_job_result") as mock_merge,
        ):
            with pytest.raises(asyncio.CancelledError):
                await svc._run_job_isolated(job, claim)

        mock_merge.assert_not_called()
        assert "reapjit1" not in svc._claims, (
            "the reaped run's claim was orphaned: _force_reap took the claim, the "
            "finalizer's ownership fence then skipped its release, and the claim "
            f"still stands with jitter {claim.jitter!r}"
        )

    @pytest.mark.asyncio
    async def test_reaper_done_task_cleanup_clears_monotonic_start(self) -> None:
        """The done-task cleanup branch releases the whole claim, monotonic stamp included."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._sessions = _mock_sessions()

        svc._claims["done2"] = _RunClaim(
            trigger="scheduled",
            claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60,
            started_monotonic=time.monotonic() - _JOB_TIMEOUT_SECS - 60,
            task=MagicMock(done=MagicMock(return_value=True)),
        )

        with patch("kiro_crew.sel.sel"), _one_sweep():
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert "done2" not in svc._claims

    @pytest.mark.asyncio
    async def test_reaper_falls_back_to_wallclock_without_monotonic_stamp(
        self, tmp_path: object
    ) -> None:
        """A claim with no monotonic stamp (a run that has not stamped one)
        still reaps, on the wall-clock elapsed."""
        svc = CronService(base_dir=None, on_job=AsyncMock())
        svc._history = CronHistoryStore(base_dir=tmp_path)
        svc._sessions = _mock_sessions()

        job = _make_job("legacy1")
        svc._jobs = [job]
        claim = svc._claims["legacy1"] = _RunClaim(
            trigger="scheduled", claimed_at=time.time() - _JOB_TIMEOUT_SECS - 60, task=_live_task()
        )
        assert claim.started_monotonic is None

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"), _one_sweep():
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()

        assert svc._reaped_jobs.has("legacy1", claim)


class TestReaperReleasesFinishedTask:
    """A finished task the sweep meets never ran its ``finally``: release it.

    ``_run_job_isolated``'s ``finally`` releases the run's claim before its task
    ends, so a task that is ``done()`` while its claim is still stored exited
    without reaching that ``finally``, and the claim -- the job's occupancy --
    is still standing. The due-scan (``_on_timer``), ``_next_wake_secs`` and
    ``run_job`` all skip a claimed job, so
    until something releases it the job silently misses every scheduled fire.
    The manual-run route releases it only when a user clicks Run; the sweep is
    the consumer that runs on its own, so it must do the same release -- and
    on the sweep that meets the finished task, not at the run's deadline,
    which is at least ``_JOB_TIMEOUT_SECS`` and up to a day away.
    """

    @pytest.mark.asyncio
    async def test_scheduled_fire_resumes_after_the_sweep_meets_a_finished_task(
        self, tmp_path: Path
    ) -> None:
        ran: list[str] = []

        async def callback(job: CronJob) -> None:
            ran.append(job.id)

        svc = CronService(base_dir=tmp_path, on_job=callback)
        svc._sessions = _mock_sessions()
        await svc.start()
        try:
            svc.add_job("watch", "go", every_secs=60)
            job = svc._jobs[0]
            job.last_run_ts = time.time() - 120  # due now

            async def _died_before_cleanup() -> None:
                raise RuntimeError("run ended without reaching its finally")

            # What a run leaves behind when its task ends ahead of the
            # try/finally: its claim still stored, holding the finished task
            # and the stamps -- and well inside the deadline, so the sweep's
            # timeout path is not what releases it.
            stale = asyncio.get_running_loop().create_task(_died_before_cleanup())
            await asyncio.gather(stale, return_exceptions=True)
            assert stale.done()
            svc._claims[job.id] = _RunClaim(
                trigger="scheduled",
                claimed_at=time.time() - 60,
                started_monotonic=time.monotonic() - 60,
                jitter=0.0,
                task=stale,
            )

            # Control: while the leftovers stand, the due-scan skips the job.
            await svc._on_timer()
            assert svc._claims[job.id].task is stale
            assert ran == []

            with patch("kiro_crew.sel.sel"), _one_sweep():
                with pytest.raises(asyncio.CancelledError):
                    await svc._reaper_loop()

            assert job.id not in svc._claims, (
                "the sweep met a finished task and left the job claimed, "
                "so the due-scan keeps skipping every scheduled fire of it"
            )
            # Released, not reaped: the run was over, there was nothing to kill.
            assert not svc._reaped_jobs._marks
            svc._sessions.reset.assert_not_awaited()

            # The next tick fires the job again.
            await svc._on_timer()
            fresh = svc._claims[job.id].task
            assert fresh is not None and fresh is not stale
            await fresh
            assert ran == [job.id]
        finally:
            await svc.stop()
