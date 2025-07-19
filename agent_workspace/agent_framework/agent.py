import threading
import time
from .message_bus import MessageBus
from .task import Task

class Agent:
    def __init__(self, agent_id: str, name: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.name = name
        self.message_bus = message_bus
        self._task_queue = []
        self._thread = None
        self._running = False
        self.message_bus.subscribe(f'task_for_agent_{self.agent_id}', self._handle_incoming_task)

    def _handle_incoming_task(self, message):
        task = message.get('task')
        if task:
            print(f"Agent {self.name} received task: {task.task_id}")
            self._task_queue.append(task)
            if not self._running:
                self.start()

    def receive_task(self, task: Task):
        self._task_queue.append(task)
        if not self._running:
            self.start()

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_tasks)
            self._thread.daemon = True
            self._thread.start()
            print(f"Agent {self.name} started.")

    def stop(self):
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1)
            print(f"Agent {self.name} stopped.")

    def _run_tasks(self):
        while self._running:
            if self._task_queue:
                task = self._task_queue.pop(0)
                print(f"Agent {self.name} executing task: {task.task_id}")
                try:
                    result = task.execute()
                    self.message_bus.publish('TASK_COMPLETE', {'task_id': task.task_id, 'result': result, 'agent_id': self.agent_id})
                except Exception as e:
                    self.message_bus.publish('TASK_FAILED', {'task_id': task.task_id, 'error': str(e), 'agent_id': self.agent_id})
                time.sleep(0.1) # Simulate work
            else:
                time.sleep(0.5) # Wait for tasks
