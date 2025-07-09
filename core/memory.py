"""
Memory Manager for session context and history
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
logger = logging.getLogger("codegen.memory")

class MemoryManager:
    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self.generations: List[Dict[str, Any]] = []
        self.file_operations: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
    
    def add_generation(self, prompt: str, result: str) -> None:
        """Add a code generation to memory"""
        entry = {
            'timestamp': datetime.now(),
            'prompt': prompt,
            'result': result,
            'type': 'generation'
        }
        
        self.generations.append(entry)
        
        # Keep only the most recent entries
        if len(self.generations) > self.max_entries:
            self.generations = self.generations[-self.max_entries:]
        
        logger.info(f"Added generation to memory: {prompt[:50]}...")
    
    def add_file_access(self, filename: str, operation: str, content: Optional[str] = None) -> None:
        """Add a file operation to memory"""
        entry = {
            'timestamp': datetime.now(),
            'filename': filename,
            'operation': operation,
            'content_preview': content[:100] + "..." if content and len(content) > 100 else content,
            'type': 'file_operation'
        }
        
        self.file_operations.append(entry)
        
        # Keep only the most recent entries
        if len(self.file_operations) > self.max_entries:
            self.file_operations = self.file_operations[-self.max_entries:]
        
        logger.info(f"Added file operation to memory: {operation} {filename}")
    
    def get_recent_generations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent code generations"""
        return self.generations[-count:] if self.generations else []
    
    def get_recent_file_operations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent file operations"""
        return self.file_operations[-count:] if self.file_operations else []
    
    def get_context_for_file(self, filename: str) -> List[Dict[str, Any]]:
        """Get context related to a specific file"""
        context = []
        
        # Find file operations for this file
        for op in self.file_operations:
            if op['filename'] == filename:
                context.append(op)
        
        # Find generations that might be related to this file
        for gen in self.generations:
            if filename in gen['prompt'] or filename in gen['result']:
                context.append(gen)
        
        # Sort by timestamp
        context.sort(key=lambda x: x['timestamp'])
        return context
    
    def display_memory(self) -> None:
        """Display current session memory"""
        session_duration = datetime.now() - self.session_start
        
        # Session info
        info_table = Table(title="🧠 Session Memory")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Session Duration", str(session_duration).split('.')[0])
        info_table.add_row("Code Generations", str(len(self.generations)))
        info_table.add_row("File Operations", str(len(self.file_operations)))
        
        console.print(info_table)
        console.print()
        
        # Recent generations
        if self.generations:
            console.print("[bold blue]📝 Recent Code Generations:[/bold blue]")
            gen_table = Table()
            gen_table.add_column("Time", style="dim")
            gen_table.add_column("Prompt", style="white")
            gen_table.add_column("Preview", style="green")
            
            for gen in self.get_recent_generations():
                time_str = gen['timestamp'].strftime("%H:%M:%S")
                prompt_preview = gen['prompt'][:40] + "..." if len(gen['prompt']) > 40 else gen['prompt']
                result_preview = gen['result'][:50].replace('\n', ' ') + "..." if len(gen['result']) > 50 else gen['result'].replace('\n', ' ')
                
                gen_table.add_row(time_str, prompt_preview, result_preview)
            
            console.print(gen_table)
            console.print()
        
        # Recent file operations
        if self.file_operations:
            console.print("[bold yellow]📁 Recent File Operations:[/bold yellow]")
            file_table = Table()
            file_table.add_column("Time", style="dim")
            file_table.add_column("Operation", style="cyan")
            file_table.add_column("File", style="white")
            file_table.add_column("Preview", style="green")
            
            for op in self.get_recent_file_operations():
                time_str = op['timestamp'].strftime("%H:%M:%S")
                preview = op.get('content_preview', '') or ''
                
                file_table.add_row(
                    time_str,
                    op['operation'],
                    op['filename'],
                    preview
                )
            
            console.print(file_table)
    
    def clear_memory(self) -> None:
        """Clear all session memory"""
        self.generations.clear()
        self.file_operations.clear()
        logger.info("Session memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'session_start': self.session_start,
            'session_duration': datetime.now() - self.session_start,
            'total_generations': len(self.generations),
            'total_file_operations': len(self.file_operations),
            'memory_usage': {
                'generations': len(self.generations),
                'file_operations': len(self.file_operations),
                'max_entries': self.max_entries
            }
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """Export memory for persistence"""
        return {
            'session_start': self.session_start.isoformat(),
            'generations': [
                {
                    'timestamp': gen['timestamp'].isoformat(),
                    'prompt': gen['prompt'],
                    'result': gen['result'],
                    'type': gen['type']
                }
                for gen in self.generations
            ],
            'file_operations': [
                {
                    'timestamp': op['timestamp'].isoformat(),
                    'filename': op['filename'],
                    'operation': op['operation'],
                    'content_preview': op.get('content_preview'),
                    'type': op['type']
                }
                for op in self.file_operations
            ]
        }
