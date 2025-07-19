from typing import Callable, Any, Dict

class Task:
    def __init__(self, task_id: str, description: str, func: Callable, args: Dict[str, Any]):
        self.task_id = task_id
        self.description = description
        self.func = func
        self.args = args

    def execute(self) -> Any:
        return self.func(**self.args)
