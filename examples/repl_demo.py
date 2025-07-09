"""
Demo script showing REPL capabilities
This can be loaded in REPL with: load examples/repl_demo.py
"""

# Example session data that can be loaded into REPL
session_demo = {
    "memory": {
        "session_start": "2024-01-01T10:00:00",
        "generations": [
            {
                "timestamp": "2024-01-01T10:05:00",
                "prompt": "create a simple web scraper",
                "result": "import requests\nfrom bs4 import BeautifulSoup\n\ndef scrape_url(url):\n    response = requests.get(url)\n    soup = BeautifulSoup(response.content, 'html.parser')\n    return soup.get_text()",
                "type": "generation"
            }
        ],
        "file_operations": [
            {
                "timestamp": "2024-01-01T10:06:00", 
                "filename": "scraper.py",
                "operation": "write",
                "type": "file_operation"
            }
        ]
    },
    "variables": {
        "last_generated": "Web scraper code",
        "project_name": "web_scraper_demo"
    },
    "config": {
        "verbose": True,
        "debug": False,
        "default_model": "gpt-3.5-turbo",
        "timeout": 30
    },
    "current_dir": "/home/user/projects"
}

print("Demo session data ready for REPL")
