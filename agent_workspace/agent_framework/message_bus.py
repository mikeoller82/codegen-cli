from collections import defaultdict
from typing import Callable, Dict, Any

class MessageBus:
    def __init__(self):
        self._subscribers: Dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, message: Dict[str, Any]):
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Error handling message for event {event_type}: {e}")
