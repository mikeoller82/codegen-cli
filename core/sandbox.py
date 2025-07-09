"""
Code Sandbox for safe execution of generated code
"""

import subprocess
import tempfile
import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

from . import config

logger = logging.getLogger("codegen.sandbox")

class CodeSandbox:
    """Safe execution environment for generated code"""
    
    def __init__(self):
        self.timeout = config.get_config_value('sandbox_timeout', 10)
        self.temp_dir = Path(tempfile.gettempdir()) / "codegen_sandbox"
        self.temp_dir.mkdir(exist_ok=True)
    
    def execute_code(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute Python code safely"""
        timeout = timeout or self.timeout
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.temp_dir) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.temp_dir
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'execution_time': execution_time,
                'timeout': False
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Execution timed out after {timeout} seconds',
                'return_code': -1,
                'execution_time': timeout,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'execution_time': 0,
                'timeout': False
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def execute_file(self, filename: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a Python file safely"""
        timeout = timeout or self.timeout
        
        if not Path(filename).exists():
            return {
                'success': False,
                'stdout': '',
                'stderr': f'File not found: {filename}',
                'return_code': -1,
                'execution_time': 0,
                'timeout': False
            }
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(filename).parent
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'execution_time': execution_time,
                'timeout': False
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Execution timed out after {timeout} seconds',
                'return_code': -1,
                'execution_time': timeout,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'execution_time': 0,
                'timeout': False
            }
    
    def display_results(self, result: Dict[str, Any]):
        """Display execution results"""
        if not HAS_RICH or not console:
            # Fallback display
            print(f"Success: {result['success']}")
            print(f"Return Code: {result['return_code']}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            
            if result['stdout']:
                print("STDOUT:")
                print(result['stdout'])
            
            if result['stderr']:
                print("STDERR:")
                print(result['stderr'])
            return
        
        # Rich display
        status_color = "green" if result['success'] else "red"
        status_text = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        
        info_text = f"[{status_color}]{status_text}[/{status_color}]\n"
        info_text += f"Return Code: {result['return_code']}\n"
        info_text += f"Execution Time: {result['execution_time']:.2f}s"
        
        if result['timeout']:
            info_text += f"\n[red]⏰ TIMEOUT[/red]"
        
        console.print(Panel(info_text, title="Execution Results"))
        
        if result['stdout']:
            console.print(Panel(result['stdout'], title="Output", border_style="green"))
        
        if result['stderr']:
            console.print(Panel(result['stderr'], title="Errors", border_style="red"))
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate Python code syntax without executing"""
        try:
            compile(code, '<string>', 'exec')
            return {
                'valid': True,
                'error': None
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': str(e),
                'line': e.lineno,
                'offset': e.offset
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up sandbox directory")
        except Exception as e:
            logger.warning(f"Error cleaning up sandbox: {e}")
