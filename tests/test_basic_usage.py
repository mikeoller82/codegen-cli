import sys
sys.path.insert(0, '/home/mike/codegen-cli/agent_workspace')
from examples.basic_usage import sample_task

# Simple test of task execution
def test_sample_task():
    test_task = sample_task(name="Test User")
    assert "Test User" in test_task["message"]
    assert test_task["status"] == "success"
    print("Basic test passed!")

if __name__ == "__main__":
    test_sample_task()
