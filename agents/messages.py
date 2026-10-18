"""Explicit agent messages and a message log (A1/A5, M1 interaction count)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    """One agent-to-agent message: sender, receiver, type, payload."""

    sender: str
    receiver: str
    type: str
    payload: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "payload": dict(self.payload),
        }


class MessageLog:
    """Ordered log of every message exchanged during a scenario."""

    def __init__(self):
        self.messages: list[Message] = []

    def record(self, sender: str, receiver: str, type: str, payload: dict | None = None) -> Message:
        message = Message(sender=sender, receiver=receiver, type=type, payload=dict(payload or {}))
        self.messages.append(message)
        return message

    @property
    def count(self) -> int:
        return len(self.messages)

    def interactions(self) -> int:
        """Number of messages exchanged (M1 agent-interaction counter)."""
        return len(self.messages)

    def by_type(self) -> dict[str, int]:
        return dict(Counter(message.type for message in self.messages))

    def clear(self) -> None:
        self.messages.clear()

    def as_dicts(self) -> list[dict]:
        return [message.as_dict() for message in self.messages]
