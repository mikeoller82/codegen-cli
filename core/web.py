"""
Web Manager for fetching and processing web content
"""

import logging
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import json

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

from . import config

logger = logging.getLogger("codegen.web")

class WebManager:
    """Manages web requests and content fetching"""
    
    def __init__(self):
        self.timeout = config.get_config_value('timeout', 30)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CodeGen-CLI/1.0 (Educational Tool)'
        })
    
    def fetch_url(self, url: str, format_type: str = 'auto') -> Dict[str, Any]:
        """Fetch content from a URL"""
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")
            
            # Make request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get('content-type', '').lower()
            
            # Process based on content type or format
            if format_type == 'json' or 'application/json' in content_type:
                try:
                    data = response.json()
                    return {
                        'url': url,
                        'content_type': 'json',
                        'data': data,
                        'raw': response.text,
                        'status_code': response.status_code,
                        'headers': dict(response.headers)
                    }
                except json.JSONDecodeError:
                    pass
            
            # Default to text
            return {
                'url': url,
                'content_type': 'text',
                'data': response.text,
                'raw': response.text,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            raise
    
    def display_content(self, content: Dict[str, Any], format_type: str = 'auto'):
        """Display fetched content"""
        if not HAS_RICH or not console:
            # Fallback display
            print(f"URL: {content['url']}")
            print(f"Status: {content['status_code']}")
            print(f"Content Type: {content['content_type']}")
            print("-" * 40)
            
            if content['content_type'] == 'json':
                print(json.dumps(content['data'], indent=2))
            else:
                print(content['data'][:1000] + "..." if len(content['data']) > 1000 else content['data'])
            return
        
        # Rich display
        console.print(f"[bold blue]URL:[/bold blue] {content['url']}")
        console.print(f"[bold green]Status:[/bold green] {content['status_code']}")
        console.print(f"[bold yellow]Content Type:[/bold yellow] {content['content_type']}")
        
        if content['content_type'] == 'json':
            json_obj = JSON.from_data(content['data'])
            console.print(Panel(json_obj, title="JSON Response"))
        else:
            # Truncate long text content
            text_content = content['data']
            if len(text_content) > 2000:
                text_content = text_content[:2000] + "\n\n... (truncated)"
            
            console.print(Panel(text_content, title="Text Response"))
    
    def fetch_api_data(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Fetch data from an API endpoint"""
        if headers:
            session = requests.Session()
            session.headers.update(headers)
        else:
            session = self.session
        
        try:
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            return {
                'success': True,
                'data': response.json() if 'application/json' in response.headers.get('content-type', '') else response.text,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
