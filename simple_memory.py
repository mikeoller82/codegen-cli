"""
Simple memory system for the agentic coder
Lightweight alternative to the complex learning engine
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class MemoryEntry:
    """Simple memory entry"""

    task: str
    solution: str
    success: bool
    timestamp: datetime
    tools_used: List[str]
    files_created: List[str]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class SimpleMemory:
    """Lightweight memory system for learning from past tasks"""

    def __init__(self, memory_file: str = "agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.memories: List[MemoryEntry] = []
        self.load_memory()

    def load_memory(self):
        """Load memories from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    self.memories = [MemoryEntry.from_dict(entry) for entry in data]
            except Exception as e:
                print(f"Warning: Could not load memory file: {e}")
                self.memories = []

    def save_memory(self):
        """Save memories to file"""
        try:
            with open(self.memory_file, "w") as f:
                data = [memory.to_dict() for memory in self.memories]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")

    def add_memory(
        self,
        task: str,
        solution: str,
        success: bool,
        tools_used: List[str],
        files_created: List[str],
    ):
        """Add a new memory"""
        memory = MemoryEntry(
            task=task,
            solution=solution,
            success=success,
            timestamp=datetime.now(),
            tools_used=tools_used,
            files_created=files_created,
        )
        self.memories.append(memory)

        # Keep only last 100 memories to avoid bloat
        if len(self.memories) > 100:
            self.memories = self.memories[-100:]

        self.save_memory()

    def find_similar_tasks(self, task: str, limit: int = 3) -> List[MemoryEntry]:
        """Find similar successful tasks"""
        task_lower = task.lower()
        similar = []

        for memory in self.memories:
            if not memory.success:
                continue

            # Simple similarity based on common words
            memory_words = set(memory.task.lower().split())
            task_words = set(task_lower.split())

            common_words = memory_words.intersection(task_words)
            if len(common_words) >= 2:  # At least 2 common words
                similarity = len(common_words) / len(task_words.union(memory_words))
                similar.append((similarity, memory))

        # Sort by similarity and return top matches
        similar.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in similar[:limit]]

    def get_successful_patterns(self) -> Dict[str, Any]:
        """Get patterns from successful tasks"""
        successful = [m for m in self.memories if m.success]

        if not successful:
            return {}

        # Common tools used
        tool_usage = {}
        for memory in successful:
            for tool in memory.tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1

        # Common file types created
        file_types = {}
        for memory in successful:
            for file_path in memory.files_created:
                ext = Path(file_path).suffix
                if ext:
                    file_types[ext] = file_types.get(ext, 0) + 1

        return {
            "total_successful": len(successful),
            "common_tools": sorted(
                tool_usage.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "common_file_types": sorted(
                file_types.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "recent_success_rate": self._calculate_recent_success_rate(),
        }

    def _calculate_recent_success_rate(self, days: int = 7) -> float:
        """Calculate success rate for recent tasks"""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [m for m in self.memories if m.timestamp > cutoff]

        if not recent:
            return 0.0

        successful = sum(1 for m in recent if m.success)
        return successful / len(recent)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.memories:
            return {"total": 0, "successful": 0, "success_rate": 0.0}

        successful = sum(1 for m in self.memories if m.success)
        return {
            "total": len(self.memories),
            "successful": successful,
            "success_rate": successful / len(self.memories),
            "oldest": self.memories[0].timestamp.isoformat() if self.memories else None,
            "newest": self.memories[-1].timestamp.isoformat()
            if self.memories
            else None,
        }
