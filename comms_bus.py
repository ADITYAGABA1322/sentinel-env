from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(frozen=True)
class CommsMessage:
    sender: str
    receiver: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time)


class CommsBus:
    """
    Lightweight message log for partial-observability experiments.

    The environment keeps hidden specialist metadata internally, while the
    orchestrator-facing view only exposes the public response, confidence, and
    outcome summary. This makes the trust problem behavioral instead of identity
    based.
    """

    def __init__(self, partial_observability: bool = True) -> None:
        self.partial_observability = partial_observability
        self._messages: list[CommsMessage] = []

    def reset(self) -> None:
        self._messages.clear()

    def route(self, sender: str, receiver: str, payload: dict[str, Any]) -> dict[str, Any]:
        visible_payload = self._filter_payload(payload)
        self._messages.append(
            CommsMessage(sender=sender, receiver=receiver, payload=visible_payload)
        )
        return visible_payload

    def history(self, receiver: str | None = None) -> list[dict[str, Any]]:
        messages = self._messages
        if receiver is not None:
            messages = [msg for msg in messages if msg.receiver == receiver]
        return [
            {
                "sender": msg.sender,
                "receiver": msg.receiver,
                "payload": dict(msg.payload),
                "timestamp": round(msg.timestamp, 3),
            }
            for msg in messages
        ]

    def _filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.partial_observability:
            return dict(payload)

        hidden_keys = {
            "internal_id",
            "internal_profile",
            "ground_truth_reliability",
            "adversarial_slot",
        }
        return {key: value for key, value in payload.items() if key not in hidden_keys}
