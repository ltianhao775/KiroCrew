"""Mid-turn queue receipts — the single collapsing "⏳ Queued (N): …" bubble.

A message that arrives while a turn is running is either folded into that turn
as a steer or queued for after it. Queued messages get ONE receipt bubble that
is edited in place as the burst grows, then flipped to a durable record when the
turn drains it ("▶️ Now answering") or cancelled ("🛑 Cancelled"). Two channels
grew the same subsystem independently -- Telegram and Discord, ~560 duplicated
lines -- and this module is the half of it that is genuinely channel-neutral.

What lives here and what does NOT:

* HERE -- the receipt registry, its lock, and the three lifecycle transitions
  (create/grow, flip-to-answering, finalize-cancelled). These are pure
  bookkeeping over an opaque message id, and every line of them was identical
  across the two channels apart from the address type and the send call.
* NOT here -- ``_handle_busy`` and ``_drain_queue``. They re-enter the channel's
  own ``handle_message`` (whose signature differs per channel: route/chat_id/
  thread vs user_id/channel_id/thread_id) and they own the ``_active_renderers``
  registry. Sharing them would need a ``run_turn`` callback that buys nothing
  and couples this module to turn execution.

Channels reach the transitions through :class:`ReceiptSurface`, whose address is
bound at CONSTRUCTION -- so nothing below ever sees a ``chat_id``, a ``thread``
or a ``channel_id``, which is what let the five address-shaped divergences
between the two copies collapse to zero.

Dependency direction is ``<channel> -> messaging`` (never the reverse), matching
``messaging/dispatch.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Verbatim items shown in a receipt before "…and N more". A large mid-turn
#: burst would otherwise grow the rendered receipt past a channel's message
#: limit; the count prefix still reflects the true total.
RECEIPT_MAX_ITEMS = 5

#: Instant, no-extra-bubble acknowledgement that a mid-turn steer was accepted
#: and folded into the running turn (not merely "seen" — 👀 reads as passive).
STEER_ACK_EMOJI = "🫡"

#: Upper bound on how many queued messages collapse into a single combined turn.
#: A single human will not realistically burst past this mid-turn; anything beyond
#: stays queued and drains after the next turn. Lives here rather than in each
#: dispatcher because it bounds the same collapse in every channel that carries
#: the queue, and three copies had already drifted apart by comment alone.
MAX_COLLAPSE = 50

#: What a receipt SHOWS for a message whose only content was an upload. An
#: attachment-only message has no text, and a blank line in the bubble reads as a
#: message the queue lost. Lives here because every channel that ingests files
#: needs the same substitution in both receipt transitions.
ATTACHMENT_PLACEHOLDER = "[attachment]"


def short(text: str, limit: int = 40) -> str:
    """Collapse whitespace and truncate for compact receipt display."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def receipt_text(
    texts: list[str],
    *,
    answering: bool = False,
    cancelled: bool = False,
) -> str:
    """Render the single collapsing receipt for ``texts`` (order preserved).

    Only the first :data:`RECEIPT_MAX_ITEMS` are listed verbatim; the count
    prefix still reflects the true total.
    """
    count = len(texts)
    items = " · ".join(f"“{short(t)}”" for t in texts[:RECEIPT_MAX_ITEMS])
    if count > RECEIPT_MAX_ITEMS:
        items += f" · …and {count - RECEIPT_MAX_ITEMS} more"
    if cancelled:
        return f"🛑 Cancelled ({count}): {items}"
    if answering:
        return f"▶️ Now answering ({count}): {items}"
    return f"⏳ Queued ({count}): {items}"


@dataclass(frozen=True)
class ReceiptLine:
    """One queued message as the receipt lists it: whose it is, and what it shows.

    The owner is the token :func:`kiro_crew.messaging.queue_drain.owner_token` builds,
    the same value the queue entry itself carries, so the bubble and the queue agree
    about who queued what. Empty for a producer that cannot name its principal, which
    makes the line nobody's to withdraw.
    """

    owner: str
    text: str


@dataclass
class QueueReceipt:
    """The single, in-place receipt bubble tracking messages queued mid-turn.

    ``msg_id`` is deliberately opaque (``Any``): Telegram message ids are ints
    and Discord's are strings, the two are never interleaved in one process, and
    a generic parameter would add ceremony without catching a real mixup -- the
    id is only ever handed straight back to the surface that produced it.

    One registry entry serves a whole session key, and under
    ``messaging.dm_scope = "unified"`` that key spans several chats, so the
    lines on one bubble can belong to several principals while ``msg_id``
    addresses a message in exactly one of their conversations --
    ``opened_by``'s. Two rules follow from that pairing: a transition may only
    render lines back to the principal they came from (:meth:`withdraw` returns
    exactly the caller's own), and only ``opened_by`` may be handed this
    ``msg_id``, because in anybody else's conversation the same number is
    another message.
    """

    msg_id: Any
    opened_by: str = ""
    lines: list[ReceiptLine] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        """What the bubble shows, in order. What :func:`receipt_text` renders."""
        return [line.text for line in self.lines]

    def withdraw(self, owner: str) -> list[str]:
        """Drop *owner*'s lines and return what they showed, in order.

        An empty *owner* drops nothing, matching the queue-side predicate: a caller that
        cannot name its principal withdraws nothing rather than everybody's lines.
        """
        if not owner:
            return []
        taken = [line.text for line in self.lines if line.owner == owner]
        if taken:
            self.lines = [line for line in self.lines if line.owner != owner]
        return taken


class ReceiptSurface(Protocol):
    """One conversation's receipt bubble, with its address already bound.

    Implementations close over whatever addresses their channel needs (Telegram
    binds ``chat_id`` AND the forum ``thread``; Discord binds ``channel_id``), so
    forum routing and channel addressing stay entirely channel-local.
    """

    #: Channel name for log lines only ("telegram" / "discord").
    label: str

    async def send_receipt(self, body: str) -> Any | None:
        """Post a new receipt bubble. Returns an opaque message id, or None."""

    async def edit_receipt(self, msg_id: Any, body: str) -> None:
        """Rewrite the receipt in place. May raise; the queue logs and continues."""


class ReceiptQueue:
    """Owns the per-session receipt registry, its lock, and the transitions.

    The lock is deliberately CALLER-HELD and exposed as :attr:`lock` rather than
    taken inside each method. Holding it across BOTH the enqueue and the receipt
    bookkeeping is what makes the subsystem race-free against the end-of-turn
    drain, which takes the same lock across dequeue + flip: the drain either sees
    a message queued WITH its receipt or sees neither yet -- never a half state
    that would orphan a bubble. ``/stop`` holds it across clear_queue + finalize
    for the same reason. Hiding the lock inside these methods would silently
    reintroduce that race, which is why the ``_locked`` suffixes stay in the
    public names: ugly, and load-bearing.
    """

    def __init__(self) -> None:
        self._receipts: dict[str, QueueReceipt] = {}
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        """The lock callers MUST hold across compound operations (see class doc)."""
        return self._lock

    def has_receipt(self, session_key: str) -> bool:
        """Whether a live receipt exists for this session."""
        return session_key in self._receipts

    async def create_or_grow_locked(
        self,
        session_key: str,
        surface: ReceiptSurface,
        display_text: str,
        owner: str = "",
    ) -> None:
        """Create the receipt, or append to it and edit in place.

        ``display_text`` is what the receipt SHOWS, which is not always the raw
        message: a file-capable channel substitutes :data:`ATTACHMENT_PLACEHOLDER`
        for an attachment-only message so the bubble is not blank. Caller MUST hold
        :attr:`lock`, and MUST have already enqueued the message under that same
        hold.

        ``owner`` is who queued this one line -- the same token the queue entry carries
        -- so a later ``/stop`` for one principal can withdraw that person's lines and
        leave the rest alone. Pass the value the entry was tagged with; empty means the
        line is nobody's to withdraw.
        """
        receipt = self._receipts.get(session_key)
        line = ReceiptLine(owner=owner, text=display_text)
        if receipt is None:
            msg_id = await surface.send_receipt(receipt_text([display_text]))
            if msg_id is not None:
                self._receipts[session_key] = QueueReceipt(
                    msg_id=msg_id, opened_by=owner, lines=[line]
                )
            return
        receipt.lines.append(line)
        try:
            await surface.edit_receipt(receipt.msg_id, receipt_text(receipt.texts))
        except Exception:
            logger.debug("%s: queue receipt grow failed", surface.label, exc_info=True)

    async def flip_answering_locked(
        self,
        session_key: str,
        surface: ReceiptSurface,
        answered: list[str],
        deferred: int = 0,
    ) -> None:
        """Flip the receipt to a durable "▶️ Now answering" record.

        Drops the live entry so the next mid-turn burst opens a fresh receipt.
        ``answered`` is the subset this turn actually answers (the drain caps it),
        so a burst past the cap does not overstate the turn; ``deferred`` (>0 only
        past the cap) is noted so the remainder is not silently implied. Caller
        MUST hold :attr:`lock` across dequeue + this call.
        """
        receipt = self._receipts.pop(session_key, None)
        if receipt is None:
            return
        body = receipt_text(answered, answering=True)
        if deferred:
            body += f" · +{deferred} deferred"
        try:
            await surface.edit_receipt(receipt.msg_id, body)
        except Exception:
            logger.debug("%s: queue receipt flip failed", surface.label, exc_info=True)

    async def finish_cancelled_locked(
        self, session_key: str, surface: ReceiptSurface, owner: str = ""
    ) -> None:
        """Finalize the receipt to a "🛑 Cancelled" record, if present.

        Caller MUST hold :attr:`lock` across clear_queue + this call.

        ``owner`` names the ONE principal whose messages were cleared, and then only
        that person's lines are withdrawn from the record. The registry entry is then
        DROPPED, and whether anything is written depends on who opened the bubble:

        * the caller opened it -- ``surface`` addresses it, so it finalizes as cancelled
          over the caller's OWN withdrawn lines. Not over what remains: those lines
          belong to other principals, and this is their sender's conversation only by
          coincidence of who queued first.
        * somebody else opened it -- nothing is written at all, because ``msg_id``
          belongs to that person's conversation and in the caller's the same number is
          another message entirely.

        Dropping the entry is what keeps a later drain safe. A drain flips using the
        chat of the entry it is answering, so an entry left behind after its opener
        stopped would hand that drain an id minted in a DIFFERENT chat, and the edit
        would land on whatever message happens to hold that number there. The cost is
        that a bubble whose opener stopped goes stale rather than being flipped; the
        next mid-turn burst opens a fresh one, and which conversation a shared bubble
        belongs to is the registry key's own question.

        Omitted, the whole receipt is finalized, which is what the whole-session callers
        mean: the queue they cleared was all of it.
        """
        receipt = self._receipts.get(session_key)
        if receipt is None:
            return
        if owner:
            withdrawn = receipt.withdraw(owner)
            if not withdrawn:
                return
            self._receipts.pop(session_key, None)
            if owner == receipt.opened_by:
                await self._edit(surface, receipt.msg_id, receipt_text(withdrawn, cancelled=True))
            return
        self._receipts.pop(session_key, None)
        await self._edit(surface, receipt.msg_id, receipt_text(receipt.texts, cancelled=True))

    async def _edit(self, surface: ReceiptSurface, msg_id: Any, body: str) -> None:
        """Rewrite the bubble to *body*, logging rather than raising on failure."""
        try:
            await surface.edit_receipt(msg_id, body)
        except Exception:
            logger.debug("%s: queue receipt cancel-finalize failed", surface.label, exc_info=True)
