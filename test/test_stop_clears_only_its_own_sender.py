"""``/stop`` clears the CALLER's queued messages, not the whole shared queue.

Under ``messaging.dm_scope = "unified"`` ``build_dm_session_key`` reduces a direct
chat's bucket to ``unified:{agent}``, dropping the channel and the user, so every
allow-listed person's DM on every transport resolves to one session key and therefore
one queue. Three layers have to agree for one person's Stop to leave everybody else's
messages alone, so each is pinned here:

* the QUEUE -- ``clear_queue`` with an ownership predicate keeps what it does not match,
  including entries another transport recorded and entries nobody claimed;
* the RECEIPT -- a bubble still carrying other people's lines is rewritten as queued
  rather than finalized to a cancellation they never asked for;
* the CALL SITES -- every ``/stop`` handler passes an owner, so a channel wired up
  later cannot inherit the whole-queue clear by leaving it out.

The enumeration is deliberate rather than a sample: the predicate has to answer for a
second sender, a second transport, an untagged producer, the same person in a different
place, and an owner it cannot name at all, and each of those is a different way to
discard somebody's message.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from typing import Any

import pytest

from kiro_crew.messaging.commands import stop_running_turn
from kiro_crew.messaging.queue_drain import (
    QUEUED_CHANNEL_KEY,
    QUEUED_OWNER_KEY,
    entries_queued_by,
    entry_owner,
    owner_token,
    tag_entry,
)
from kiro_crew.messaging.queue_receipt import (
    QueueReceipt,
    ReceiptLine,
    ReceiptQueue,
    receipt_text,
)

#: Two allow-listed people on one transport, and a third on another. Under a unified
#: scope all three land on ONE session key, which is the whole premise.
ALICE = owner_token("telegram", ("u-alice", "chat-alice", "", "private"))
BOB = owner_token("telegram", ("u-bob", "chat-bob", "", "private"))
CARLA = owner_token("discord", ("u-carla", "dm-carla", ""))


# ── the queue layer ─────────────────────────────────────────────────────────


class _Deps:
    """The one dependency the allocation boundary uses on a queue entry."""

    def __init__(self) -> None:
        self.unlinked: list[str] = []

    def unlink_queued_temp_paths(self, kwargs: dict) -> None:
        self.unlinked.append(str(kwargs.get("mark", "")))


def _manager() -> tuple[Any, Any, _Deps]:
    """A real ``SessionManager`` queue with one session on the shared key.

    The allocation boundary is exercised through the manager rather than reached into,
    because the manager's signature is what every channel calls.
    """
    from kiro_crew.session import SessionManager

    mgr = SessionManager.__new__(SessionManager)
    boundary = mgr._allocation_boundary()
    deps = _Deps()
    boundary._deps = deps  # type: ignore[attr-defined]
    return mgr, boundary, deps


def _queued(owner: str, mark: str, channel: str = "telegram") -> dict:
    """One entry's keyword arguments, tagged the way a producer tags them."""
    return tag_entry({"mark": mark}, channel, owner)  # type: ignore[arg-type]


class _Session:
    """The queue-carrying parts of a session, which is all ``clear_queue`` reads."""

    def __init__(self, entries: list[tuple[str, str, dict]]) -> None:
        from collections import deque

        self.queue = deque(entries)
        self.cancelled: set[str] = set()


def _clear(
    entries: list[tuple[str, str, dict]], owned_by: Any
) -> tuple[list[str], _Deps, _Session]:
    """Run a clear over *entries* and report the marks that survived."""
    mgr, boundary, deps = _manager()
    session = _Session(entries)
    boundary._sessions = {"unified:agent": session}  # type: ignore[attr-defined]
    mgr._fold_key = lambda key: key  # type: ignore[assignment,method-assign]
    boundary.clear_queue("unified:agent", owned_by)
    return [kwargs["mark"] for _, _, kwargs in session.queue], deps, session


def _burst() -> list[tuple[str, str, dict]]:
    """Alice, Bob and Carla each holding one message on the one shared queue."""
    return [
        ("1", "alice asked", _queued(ALICE, "a")),
        ("2", "bob asked", _queued(BOB, "b")),
        ("3", "carla asked", _queued(CARLA, "c", channel="discord")),
    ]


class TestTheQueueKeepsWhatIsNotTheCallers:
    def test_a_second_senders_message_on_the_same_transport_survives(self) -> None:
        """The reported bug. Bob is still waiting for an answer.

        Alice's Stop means "stop MY turn". Bob never typed anything, so discarding his
        queued message is a loss he is not told about: the receipt he was shown flips to
        cancelled and his text is gone.
        """
        survived, _deps, _session = _clear(_burst(), entries_queued_by(ALICE))
        assert survived == ["b", "c"]

    def test_another_transports_message_survives(self) -> None:
        """Carla is on Discord; Alice's Telegram Stop cannot speak for her.

        Her entry carries no field Alice's channel can even read, which is why the owner
        token is neutral: the comparison has to work on an entry from a transport the
        caller knows nothing about.
        """
        survived, _deps, _session = _clear(_burst(), entries_queued_by(ALICE))
        assert "c" in survived

    def test_an_entry_that_named_no_owner_survives(self) -> None:
        """Untagged is nobody's, so it is nobody's to discard.

        Defaulting the other way would let any caller drop an entry it cannot prove is
        its own, which is the loss this change exists to stop.
        """
        entries = [("1", "who queued me", {"mark": "u", QUEUED_CHANNEL_KEY: "telegram"})]
        survived, _deps, _session = _clear(entries, entries_queued_by(ALICE))
        assert survived == ["u"]

    def test_the_same_person_in_a_different_place_survives(self) -> None:
        """The token is sender AND place, because a reply goes to a place.

        Alice in a forum Topic is a different conversation from Alice in her DM: the two
        get separate turns and separate answers, so a Stop in one may not empty the other.
        """
        elsewhere = owner_token("telegram", ("u-alice", "chat-alice", "99", "supergroup"))
        entries = [
            ("1", "in her dm", _queued(ALICE, "dm")),
            ("2", "in the topic", _queued(elsewhere, "topic")),
        ]
        survived, _deps, _session = _clear(entries, entries_queued_by(ALICE))
        assert survived == ["topic"]

    def test_an_unnamed_caller_clears_nothing(self) -> None:
        """An empty owner selects nothing, never everything.

        A caller that cannot say who it is has no claim on anyone's message, and the
        failure that matters here is the destructive one.
        """
        survived, _deps, _session = _clear(_burst(), entries_queued_by(""))
        assert survived == ["a", "b", "c"]

    def test_only_the_dropped_entries_temp_files_are_unlinked(self) -> None:
        """A surviving entry's attachment must still be there when its turn runs."""
        _survived, deps, _session = _clear(_burst(), entries_queued_by(ALICE))
        assert deps.unlinked == ["a"]

    def test_no_predicate_still_clears_the_whole_queue(self) -> None:
        """The whole-session callers are unchanged: ``/new``, teardown, a generation bump.

        Those genuinely retire the queue they are emptying, so narrowing them would leave
        entries on a key nothing will ever drain.
        """
        survived, deps, session = _clear(_burst(), None)
        assert survived == []
        assert deps.unlinked == ["a", "b", "c"]
        assert session.cancelled == set()


class TestTheCancelledMarkersSurviveAPartialClear:
    """``cancelled`` holds bare timestamps with nothing saying whose they are.

    ``cancel_queued`` puts a timestamp there when the message it names is already being
    drained, so ``dequeue`` skips it later. Clearing the set on a partial clear would
    un-cancel a cancel somebody else asked for, and that person's message would then be
    answered after they withdrew it.
    """

    def test_a_partial_clear_leaves_them_alone(self) -> None:
        mgr, boundary, _deps = _manager()
        session = _Session(_burst())
        session.cancelled.add("bobs-withdrawn-ts")
        boundary._sessions = {"unified:agent": session}  # type: ignore[attr-defined]
        mgr._fold_key = lambda key: key  # type: ignore[assignment,method-assign]
        boundary.clear_queue("unified:agent", entries_queued_by(ALICE))
        assert session.cancelled == {"bobs-withdrawn-ts"}

    def test_a_whole_session_clear_still_drops_them(self) -> None:
        mgr, boundary, _deps = _manager()
        session = _Session(_burst())
        session.cancelled.add("bobs-withdrawn-ts")
        boundary._sessions = {"unified:agent": session}  # type: ignore[attr-defined]
        mgr._fold_key = lambda key: key  # type: ignore[assignment,method-assign]
        boundary.clear_queue("unified:agent", None)
        assert session.cancelled == set()


class TestTheTokenAndItsReader:
    def test_a_producer_records_the_owner_beside_the_channel(self) -> None:
        kwargs = _queued(ALICE, "a")
        assert entry_owner(kwargs) == ALICE
        assert kwargs[QUEUED_OWNER_KEY] == ALICE

    def test_two_transports_that_spell_one_id_the_same_are_two_principals(self) -> None:
        """The channel leads the token, so a shared id cannot merge two people."""
        assert owner_token("telegram", ("u1",)) != owner_token("discord", ("u1",))

    def test_a_missing_owner_reads_as_unclaimed_rather_than_raising(self) -> None:
        assert entry_owner({}) == ""


# ── the receipt layer ───────────────────────────────────────────────────────


class _Surface:
    """One conversation's receipt bubble, standing in for one chat.

    ``msg_id`` is the id this conversation hands out for its next bubble. The
    cross-chat tests below give two conversations the SAME id deliberately: per-chat
    message ids are small and dense, so two chats holding the same number is the
    ordinary case, and it is what makes a mis-addressed edit land on an unrelated
    message rather than fail loudly.
    """

    def __init__(self, label: str = "fake", msg_id: Any = 7) -> None:
        self.label = label
        self._msg_id = msg_id
        self.sent: list[str] = []
        self.edits: list[tuple[Any, str]] = []

    async def send_receipt(self, body: str) -> Any | None:
        self.sent.append(body)
        return self._msg_id

    async def edit_receipt(self, msg_id: Any, body: str) -> None:
        self.edits.append((msg_id, body))


async def _bubble(
    queue: ReceiptQueue,
    surface: _Surface,
    lines: list[tuple[str, str]],
    session_key: str = "s",
) -> None:
    """Grow one receipt over *lines*, the way a mid-turn burst grows it."""
    async with queue.lock:
        for owner, text in lines:
            await queue.create_or_grow_locked(session_key, surface, text, owner)


class TestAPartialStopWritesToNobodysSurface:
    """The disclosure rule: a caller may not render another principal's line anywhere.

    Under ``messaging.dm_scope = "unified"`` one session key spans several chats, so
    one bubble's lines can belong to several people, and the registry is keyed on the
    session key alone -- so a transition takes its address from whichever caller
    invoked it. While somebody else's line is on the bubble there is therefore no
    surface a partial stop may write to: the caller's chat is not where the bubble is,
    and the bubble's chat is not the caller's to be told about. It writes nothing.
    """

    def test_a_stop_writes_nothing_at_all_while_another_line_remains(self) -> None:
        """Bob opened the bubble and is still queued; Alice stops. No chat is written.

        Both chats hand out id 7, so an edit through Alice's surface would rewrite
        whatever message 7 is in her chat, and an edit through it carrying Bob's text
        would put his message in her conversation. Neither happens.
        """

        async def go() -> tuple[_Surface, _Surface]:
            queue = ReceiptQueue()
            bobs_chat, alices_chat = _Surface("bob"), _Surface("alice")
            # Each sender arrives through their OWN chat, which is what production
            # hands the queue: one surface standing in for both cannot tell a
            # correctly addressed edit from a mis-addressed one.
            async with queue.lock:
                await queue.create_or_grow_locked("s", bobs_chat, "bob asked", BOB)
                await queue.create_or_grow_locked("s", alices_chat, "alice asked", ALICE)
                before_bob, before_alice = len(bobs_chat.edits), len(alices_chat.edits)
                await queue.finish_cancelled_locked("s", alices_chat, ALICE)
                assert len(bobs_chat.edits) == before_bob, "Bob is not told he was stopped"
                assert len(alices_chat.edits) == before_alice
            return bobs_chat, alices_chat

        bobs_chat, alices_chat = asyncio.run(go())
        assert alices_chat.sent == [], "the second sender grows the burst, never a bubble"

    def test_the_grow_edit_goes_through_the_growing_senders_own_surface(self) -> None:
        """A second sender's line is appended through the surface that sender holds.

        Which conversation a shared bubble's id addresses is the receipt registry's own
        question, not this one. What this pins is that the append must NOT go through the
        OPENER's surface, which would write this sender's text into the opener's chat.
        """

        async def go() -> tuple[_Surface, _Surface]:
            queue = ReceiptQueue()
            bobs_chat, alices_chat = _Surface("bob"), _Surface("alice")
            async with queue.lock:
                await queue.create_or_grow_locked("s", bobs_chat, "bob asked", BOB)
                await queue.create_or_grow_locked("s", alices_chat, "alice asked", ALICE)
            return bobs_chat, alices_chat

        bobs_chat, alices_chat = asyncio.run(go())
        assert bobs_chat.edits == [], "the opener's chat never receives another's text"
        assert alices_chat.edits == [(7, receipt_text(["bob asked", "alice asked"]))]


class TestWhatTheBubbleSaysAfterAPartialStop:
    def test_nothing_is_written_while_someone_elses_lines_remain(self) -> None:
        """Bob's message is still queued, so nothing about this burst is recorded.

        Writing "Cancelled" would tell Bob his message went, when it is still on the
        queue and still owed an answer; rewriting the bubble to what is left would put
        Bob's text through Alice's surface. Alice learns her own stop worked from the
        stop reply, and the end-of-turn flip is what next updates the bubble.
        """

        async def go() -> _Surface:
            queue, chat = ReceiptQueue(), _Surface()
            await _bubble(queue, chat, [(BOB, "bob asked"), (ALICE, "alice asked")])
            before = len(chat.edits)
            async with queue.lock:
                await queue.finish_cancelled_locked("s", chat, ALICE)
            assert len(chat.edits) == before
            return chat

        chat = asyncio.run(go())
        assert not any("Cancelled" in body for _, body in chat.edits)

    def test_withdraw_takes_the_callers_lines_and_leaves_the_rest(self) -> None:
        """The record-level half, pinned on the object that holds it.

        What the boundary does with the remainder is its own decision; what must never
        happen is a withdraw that reaches past the caller's own lines.
        """
        receipt = QueueReceipt(
            msg_id=7,
            opened_by=BOB,
            lines=[
                ReceiptLine(owner=BOB, text="bob asked"),
                ReceiptLine(owner=ALICE, text="alice secret"),
                ReceiptLine(owner=CARLA, text="carla asked"),
            ],
        )
        assert receipt.withdraw(ALICE) == ["alice secret"]
        assert receipt.texts == ["bob asked", "carla asked"]

    def test_the_entry_is_dropped_so_a_later_drain_cannot_flip_a_foreign_id(self) -> None:
        """Alice opens the bubble, Bob queues, Alice stops. The entry must NOT survive.

        A drain flips using the chat of the entry it is answering, so an entry left
        behind here would hand Bob's drain a ``msg_id`` minted in ALICE's chat, and
        ``edit_message`` addresses a message by that per-chat id pair -- it would
        overwrite whatever unrelated message holds that number in Bob's chat.
        """

        async def go() -> tuple[ReceiptQueue, _Surface, _Surface]:
            queue = ReceiptQueue()
            alices_chat, bobs_chat = _Surface("alice"), _Surface("bob")
            async with queue.lock:
                await queue.create_or_grow_locked("s", alices_chat, "alice asked", ALICE)
                await queue.create_or_grow_locked("s", bobs_chat, "bob asked", BOB)
                await queue.finish_cancelled_locked("s", alices_chat, ALICE)
                # Bob's drain, arriving with Bob's own chat.
                await queue.flip_answering_locked("s", bobs_chat, ["bob asked"])
            return queue, alices_chat, bobs_chat

        queue, alices_chat, bobs_chat = asyncio.run(go())
        assert not queue.has_receipt("s")
        assert not any("Now answering" in body for _, body in bobs_chat.edits)
        assert alices_chat.edits[-1] == (7, receipt_text(["alice asked"], cancelled=True))

    def test_a_non_opener_stopping_writes_nothing_and_still_drops_the_entry(self) -> None:
        """Bob stops a bubble Alice opened. Bob's surface does not address it."""

        async def go() -> tuple[ReceiptQueue, _Surface, _Surface]:
            queue = ReceiptQueue()
            alices_chat, bobs_chat = _Surface("alice"), _Surface("bob")
            async with queue.lock:
                await queue.create_or_grow_locked("s", alices_chat, "alice asked", ALICE)
                await queue.create_or_grow_locked("s", bobs_chat, "bob asked", BOB)
                before = len(alices_chat.edits), len(bobs_chat.edits)
                await queue.finish_cancelled_locked("s", bobs_chat, BOB)
                assert (len(alices_chat.edits), len(bobs_chat.edits)) == before
            return queue, alices_chat, bobs_chat

        queue, _alices_chat, bobs_chat = asyncio.run(go())
        assert not queue.has_receipt("s")
        assert not any("Cancelled" in body for _, body in bobs_chat.edits)

    def test_the_last_line_going_finalizes_and_drops_the_receipt(self) -> None:
        """Nobody is left waiting, and the caller is necessarily the opener here."""

        async def go() -> tuple[ReceiptQueue, _Surface]:
            queue, chat = ReceiptQueue(), _Surface()
            await _bubble(queue, chat, [(ALICE, "alice asked")])
            async with queue.lock:
                await queue.finish_cancelled_locked("s", chat, ALICE)
            return queue, chat

        queue, chat = asyncio.run(go())
        assert chat.edits[-1] == (7, receipt_text(["alice asked"], cancelled=True))
        assert not queue.has_receipt("s")

    def test_a_caller_with_nothing_on_the_bubble_does_not_touch_it(self) -> None:
        """Carla stopping changes nothing Alice and Bob can see."""

        async def go() -> tuple[ReceiptQueue, _Surface]:
            queue, chat = ReceiptQueue(), _Surface()
            await _bubble(queue, chat, [(ALICE, "alice asked"), (BOB, "bob asked")])
            before = len(chat.edits)
            async with queue.lock:
                await queue.finish_cancelled_locked("s", chat, CARLA)
            assert len(chat.edits) == before
            return queue, chat

        queue, _chat = asyncio.run(go())
        assert queue.has_receipt("s")

    def test_no_owner_still_finalizes_the_whole_receipt(self) -> None:
        """The whole-session callers are unchanged.

        Rendering every line as cancelled is right there because no principal is left
        waiting: the queue that was cleared was all of it.
        """

        async def go() -> tuple[ReceiptQueue, _Surface]:
            queue, chat = ReceiptQueue(), _Surface()
            await _bubble(queue, chat, [(ALICE, "alice asked"), (BOB, "bob asked")])
            async with queue.lock:
                await queue.finish_cancelled_locked("s", chat)
            return queue, chat

        queue, chat = asyncio.run(go())
        assert chat.edits[-1] == (
            7,
            receipt_text(["alice asked", "bob asked"], cancelled=True),
        )
        assert not queue.has_receipt("s")

    def test_an_untagged_line_is_never_withdrawn_by_a_named_caller(self) -> None:
        """Mirrors the queue side: unclaimed is nobody's to withdraw, so it survives.

        Alice's own line goes from the record and the untagged one stays. Nothing is
        written either, because the untagged producer opened this bubble, so the chat
        Alice holds is not the one the id was minted in.
        """

        async def go() -> tuple[ReceiptQueue, _Surface]:
            queue, chat = ReceiptQueue(), _Surface()
            await _bubble(queue, chat, [("", "whose is this"), (ALICE, "alice asked")])
            before = len(chat.edits)
            async with queue.lock:
                await queue.finish_cancelled_locked("s", chat, ALICE)
            assert len(chat.edits) == before
            return queue, chat

        queue, chat = asyncio.run(go())
        assert not any("Cancelled" in body for _, body in chat.edits)
        assert not queue.has_receipt("s")

    def test_an_unnamed_caller_withdraws_nothing_at_all(self) -> None:
        """The guard the boundary above never reaches, pinned on the object itself.

        ``finish_cancelled_locked`` only calls ``withdraw`` for a named caller, so an
        empty token reaches it only from a future caller. Without the guard the empty
        string would match every untagged line -- the one population the queue side
        deliberately keeps -- so the contract belongs to the object, not to its caller.
        """
        receipt = QueueReceipt(
            msg_id=7,
            lines=[
                ReceiptLine(owner="", text="whose is this"),
                ReceiptLine(owner=ALICE, text="alice"),
            ],
        )
        assert receipt.withdraw("") == []
        assert receipt.texts == ["whose is this", "alice"]


# ── the two together, through the shared /stop ──────────────────────────────


class _Sessions:
    """The session surface ``stop_running_turn`` touches, holding a real queue."""

    def __init__(self, entries: list[tuple[str, str, dict]]) -> None:
        self.entries = list(entries)

    def is_busy(self, key: str) -> bool:
        return False

    def get_provider(self, key: str) -> Any:
        return None

    def clear_queue(self, key: str, owned_by: Any = None) -> None:
        if owned_by is None:
            self.entries = []
            return
        self.entries = [item for item in self.entries if not owned_by(item[2])]


class TestTheSharedStopPath:
    def test_one_persons_stop_leaves_the_others_message_queued(self) -> None:
        """End to end: Alice stops, Bob keeps his message and is told nothing."""

        async def go() -> tuple[_Sessions, _Surface, ReceiptQueue, int]:
            queue, surface = ReceiptQueue(), _Surface()
            sessions = _Sessions(_burst())
            await _bubble(
                queue,
                surface,
                [(ALICE, "alice asked"), (BOB, "bob asked")],
                "unified:agent",
            )
            before = len(surface.edits)
            await stop_running_turn(
                sessions, "unified:agent", queue=queue, surface=surface, owner=ALICE
            )
            return sessions, surface, queue, before

        sessions, surface, queue, before = asyncio.run(go())
        assert [kwargs["mark"] for _, _, kwargs in sessions.entries] == ["b", "c"]
        assert surface.edits[before:] == [
            (7, receipt_text(["alice asked"], cancelled=True))
        ], "only Alice's own line, and only because Alice opened this bubble"
        assert not any("bob asked" in body for _, body in surface.edits[before:])
        assert not queue.has_receipt("unified:agent")

    def test_the_owner_argument_is_required(self) -> None:
        """No default, so a channel added later cannot inherit the whole-queue clear.

        A default would be silent: the new channel would pass its tests and discard other
        people's messages in production.
        """
        with pytest.raises(TypeError):
            asyncio.run(
                stop_running_turn(  # type: ignore[call-arg]
                    _Sessions([]), "s", queue=ReceiptQueue(), surface=_Surface()
                )
            )


# ── the call sites ──────────────────────────────────────────────────────────

#: Each dispatcher that clears a queue on Stop, and the handler that does it. A source
#: check rather than a behavioural one because the point is that NO such handler is
#: missing the argument -- including one added after these tests were written, which no
#: behavioural test can be written for in advance.
_STOP_HANDLERS = (
    ("discord", "_handle_stop"),
    ("telegram", "_handle_stop"),
    ("teams", "_handle_stop"),
    ("webex", "_handle_stop"),
)


def _handler_source(channel: str, name: str) -> str:
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "kiro_crew"
        / channel
        / "transport_dispatch.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{channel}.{name} not found")


class TestEveryStopHandlerNamesItsCaller:
    @pytest.mark.parametrize(("channel", "name"), _STOP_HANDLERS)
    def test_the_handler_passes_an_owner(self, channel: str, name: str) -> None:
        """Whether through the shared helper or its own inline clear.

        Webex clears inline, so a check written only against ``stop_running_turn`` would
        pass while Webex still emptied the whole queue -- and Webex is the channel where
        one session key is shared even without a unified scope, because a group space
        routes every member onto it.
        """
        source = _handler_source(channel, name)
        assert "_entry_owner(" in source, f"{channel} Stop does not name its caller"

    @pytest.mark.parametrize(("channel", "name"), _STOP_HANDLERS)
    def test_the_handler_never_clears_the_whole_queue(self, channel: str, name: str) -> None:
        source = _handler_source(channel, name)
        assert "clear_queue(session_key)" not in source
