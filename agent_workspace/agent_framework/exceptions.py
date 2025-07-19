class AgentFrameworkException(Exception):
    """Base exception for the Agent Framework."""
    pass

class TaskExecutionError(AgentFrameworkException):
    """Raised when a task encounters an error during execution."""
    def __init__(self, message="Task execution failed", task_id=None, original_exception=None):
        super().__init__(message)
        self.task_id = task_id
        self.original_exception = original_exception

class AgentNotFoundException(AgentFrameworkException):
    """Raised when an agent is not found."""
    def __init__(self, message="Agent not found", agent_id=None):
        super().__init__(message)
        self.agent_id = agent_id
