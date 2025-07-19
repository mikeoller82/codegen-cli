"""
Basic framework usage example
"""
from agent_framework.agent import Agent
from agent_framework.task import Task
from agent_framework.message_bus import MessageBus

def sample_task(name: str):
    print(f"Hello {name}!")
    return {"status": "success", "message": f"Greeted {name}"}

if __name__ == "__main__":
    # Setup components
    bus = MessageBus()
    agent1 = Agent("agent-1", "Worker 1", bus)
    
    # Create sample task
    task = Task("task-1", "Sample greeting task", sample_task, {"name": "Mike"})
    
    # Subscribe to task complete events
    def task_complete_handler(message):
        print(f"Task completed: {message['task_id']}")
        print(f"Result: {message['result']}")
    
    bus.subscribe('TASK_COMPLETE', task_complete_handler)
    
    # Start agent and assign task
    agent1.receive_task(task)
    agent1.start()
