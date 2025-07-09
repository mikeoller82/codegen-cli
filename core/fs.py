"""
File System Manager for CodeGen CLI
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import mimetypes

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

from . import config

logger = logging.getLogger("codegen.fs")

class FileSystemManager:
    """Manages file system operations with safety checks"""
    
    def __init__(self):
        self.max_file_size = config.get_config_value('max_file_size', 1024 * 1024)
        self.supported_extensions = config.get_config_value('supported_extensions', ['.py', '.txt'])
    
    def read_file(self, filename: str) -> str:
        """Read a file with safety checks"""
        file_path = Path(filename)
        
        # Security checks
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {filename}")
        
        # Size check
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {filename} (max: {self.max_file_size} bytes)")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read file: {filename} ({len(content)} chars)")
            return content
            
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            logger.warning(f"Read file with latin-1 encoding: {filename}")
            return content
    
    def write_file(self, filename: str, content: str, append: bool = False) -> bool:
        """Write content to a file with safety checks"""
        file_path = Path(filename)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extension check
        if file_path.suffix and file_path.suffix not in self.supported_extensions:
            logger.warning(f"Writing to unsupported file type: {file_path.suffix}")
        
        try:
            mode = 'a' if append else 'w'
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            action = "Appended to" if append else "Wrote"
            logger.info(f"{action} file: {filename} ({len(content)} chars)")
            return True
            
        except Exception as e:
            logger.error(f"Error writing file {filename}: {e}")
            raise
    
    def append_file(self, filename: str, content: str) -> bool:
        """Append content to a file"""
        return self.write_file(filename, content, append=True)
    
    def list_files(self, directory: str = ".") -> List[str]:
        """List files in a directory"""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        try:
            files = []
            for item in dir_path.iterdir():
                files.append(str(item))
            
            logger.info(f"Listed {len(files)} items in {directory}")
            return files
            
        except PermissionError:
            raise PermissionError(f"Permission denied: {directory}")
    
    def file_exists(self, filename: str) -> bool:
        """Check if a file exists"""
        return Path(filename).exists()
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Get file information"""
        file_path = Path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'path': str(file_path.absolute()),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'extension': file_path.suffix,
            'mime_type': mimetypes.guess_type(str(file_path))[0]
        }
    
    def display_code(self, code: str, filename: str = None, language: str = None):
        """Display code with syntax highlighting"""
        if not HAS_RICH or not console:
            # Fallback to plain text
            print(code)
            return
        
        # Determine language from filename or content
        if not language and filename:
            ext = Path(filename).suffix.lower()
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.html': 'html',
                '.css': 'css',
                '.json': 'json',
                '.md': 'markdown',
                '.sql': 'sql',
                '.sh': 'bash',
                '.yml': 'yaml',
                '.yaml': 'yaml'
            }
            language = language_map.get(ext, 'text')
        
        if not language:
            language = 'python'  # Default to Python
        
        try:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            console.print(syntax)
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            console.print(code)
    
    def display_file_tree(self, directory: str = ".", max_depth: int = 3):
        """Display a file tree"""
        if not HAS_RICH or not console:
            # Simple fallback
            files = self.list_files(directory)
            for file in sorted(files):
                print(f"  {Path(file).name}")
            return
        
        from rich.tree import Tree
        
        def add_directory(tree, path: Path, current_depth: int = 0):
            if current_depth >= max_depth:
                return
            
            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                for item in items:
                    if item.name.startswith('.'):
                        continue  # Skip hidden files
                    
                    if item.is_dir():
                        branch = tree.add(f"📁 {item.name}")
                        add_directory(branch, item, current_depth + 1)
                    else:
                        icon = "📄" if item.suffix in self.supported_extensions else "📋"
                        tree.add(f"{icon} {item.name}")
            except PermissionError:
                tree.add("❌ Permission denied")
        
        root_path = Path(directory)
        tree = Tree(f"📁 {root_path.name or str(root_path)}")
        add_directory(tree, root_path)
        console.print(tree)
    
    def safe_delete(self, filename: str, confirm: bool = True) -> bool:
        """Safely delete a file with confirmation"""
        file_path = Path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        if confirm:
            if HAS_RICH:
                from rich.prompt import Confirm
                if not Confirm.ask(f"Delete {filename}?"):
                    return False
            else:
                response = input(f"Delete {filename}? (y/N): ")
                if not response.lower().startswith('y'):
                    return False
        
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
            
            logger.info(f"Deleted: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {filename}: {e}")
            raise
    
    def create_backup(self, filename: str) -> str:
        """Create a backup of a file"""
        file_path = Path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Generate backup filename
        timestamp = int(Path(filename).stat().st_mtime)
        backup_name = f"{file_path.stem}.backup.{timestamp}{file_path.suffix}"
        backup_path = file_path.parent / backup_name
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Created backup: {backup_name}")
        return str(backup_path)
