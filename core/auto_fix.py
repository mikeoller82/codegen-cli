"""
Auto-fix engine for testing and automatically correcting generated code
Enhanced with learning capabilities
"""

import logging
import tempfile
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from .sandbox import CodeSandbox
from .fs import FileSystemManager

console = Console()
logger = logging.getLogger("codegen.autofix")

class TestResult(Enum):
    """Test result types"""
    PASS = "pass"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"

@dataclass
class TestReport:
    """Report from testing code"""
    result: TestResult
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

@dataclass
class FixAttempt:
    """Record of a fix attempt"""
    attempt_number: int
    original_error: str
    fix_strategy: str
    fixed_code: str
    test_result: TestReport
    success: bool

class AutoFixEngine:
    """Engine for automatically testing and fixing generated code with learning"""
    
    def __init__(self, ai_engine=None, learning_engine=None, max_fix_attempts=3):
        self.ai_engine = ai_engine
        self.learning_engine = learning_engine
        self.sandbox = CodeSandbox()
        self.fs_manager = FileSystemManager()
        self.max_fix_attempts = max_fix_attempts
        
        # Fix strategies in order of preference
        self.fix_strategies = [
            self._apply_learned_fixes,  # Try learned fixes first
            self._fix_syntax_errors,
            self._fix_import_errors,
            self._fix_runtime_errors,
            self._add_error_handling,
            self._add_basic_structure
        ]
        
        # Common error patterns and fixes
        self.error_patterns = {
            r"ModuleNotFoundError: No module named '(\w+)'": self._suggest_import_fix,
            r"NameError: name '(\w+)' is not defined": self._suggest_variable_fix,
            r"IndentationError": self._suggest_indentation_fix,
            r"SyntaxError": self._suggest_syntax_fix,
            r"TypeError: (\w+)$$$$ missing \d+ required positional argument": self._suggest_argument_fix,
        }
    
    def test_and_fix_code(self, code: str, description: str = "", 
                         show_progress: bool = True) -> Tuple[str, List[FixAttempt], TestReport]:
        """Test code and automatically fix issues if found"""
        
        if show_progress:
            console.print(f"\n[bold blue]🧪 Testing Generated Code[/bold blue]")
            if description:
                console.print(f"[dim]Description: {description}[/dim]")
        
        fix_attempts = []
        current_code = code
        
        # Initial test
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            if show_progress:
                task = progress.add_task("Running initial test...", total=None)
            
            initial_test = self._test_code(current_code)
        
        if initial_test.result == TestResult.PASS:
            if show_progress:
                console.print("[green]✅ Code passed all tests![/green]")
            
            # Learn from successful generation (no fixes needed)
            if self.learning_engine:
                self.learning_engine.learn_from_generation(description, current_code, [], True)
            
            return current_code, fix_attempts, initial_test
        
        if show_progress:
            console.print(f"[yellow]⚠️  Issues found: {initial_test.result.value}[/yellow]")
            self._display_test_report(initial_test)
        
        # Attempt fixes
        for attempt in range(1, self.max_fix_attempts + 1):
            if show_progress:
                console.print(f"\n[bold yellow]🔧 Fix Attempt {attempt}/{self.max_fix_attempts}[/bold yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                if show_progress:
                    task = progress.add_task(f"Attempting fix {attempt}...", total=None)
                
                fixed_code, fix_strategy = self._attempt_fix(current_code, initial_test)
                
                if fixed_code == current_code:
                    if show_progress:
                        console.print("[red]❌ No fix could be applied[/red]")
                    break
                
                # Test the fixed code
                test_result = self._test_code(fixed_code)
                
                fix_attempt = FixAttempt(
                    attempt_number=attempt,
                    original_error=initial_test.error_message,
                    fix_strategy=fix_strategy,
                    fixed_code=fixed_code,
                    test_result=test_result,
                    success=test_result.result == TestResult.PASS
                )
                
                fix_attempts.append(fix_attempt)
                
                # Learn from this fix attempt
                if self.learning_engine:
                    self.learning_engine.learn_from_fix(
                        current_code, fixed_code, initial_test.error_message,
                        fix_strategy, fix_attempt.success
                    )
                
                if test_result.result == TestResult.PASS:
                    if show_progress:
                        console.print(f"[green]✅ Fix successful with strategy: {fix_strategy}[/green]")
                    current_code = fixed_code
                    break
                else:
                    if show_progress:
                        console.print(f"[yellow]⚠️  Fix attempt failed: {test_result.result.value}[/yellow]")
                    current_code = fixed_code
                    initial_test = test_result
        
        # Final result
        final_test = fix_attempts[-1].test_result if fix_attempts else initial_test
        final_success = final_test.result == TestResult.PASS
        
        # Learn from the overall generation process
        if self.learning_engine:
            self.learning_engine.learn_from_generation(description, current_code, fix_attempts, final_success)
        
        if show_progress:
            if final_success:
                console.print(f"\n[bold green]🎉 Code fixed successfully after {len(fix_attempts)} attempts![/bold green]")
                
                # Show learning insights
                if self.learning_engine and fix_attempts:
                    console.print(f"[dim]💡 Learning: Recorded {len(fix_attempts)} fix patterns for future use[/dim]")
            else:
                console.print(f"\n[bold red]❌ Could not fix code after {len(fix_attempts)} attempts[/bold red]")
                self._display_test_report(final_test)
        
        return current_code, fix_attempts, final_test
    
    def _test_code(self, code: str) -> TestReport:
        """Test code and return detailed report"""
        try:
            # First check syntax
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return TestReport(
                    result=TestResult.SYNTAX_ERROR,
                    error_message=str(e),
                    suggestions=self._get_syntax_suggestions(str(e))
                )
            
            # Execute the code
            execution_result = self.sandbox.execute_code(code, timeout=10)
            
            if execution_result['timeout']:
                return TestReport(
                    result=TestResult.TIMEOUT,
                    error_message="Code execution timed out",
                    execution_time=10.0
                )
            
            if execution_result['success']:
                return TestReport(
                    result=TestResult.PASS,
                    stdout=execution_result['stdout'],
                    stderr=execution_result['stderr'],
                    execution_time=execution_result.get('execution_time', 0)
                )
            else:
                # Analyze the error
                stderr = execution_result['stderr']
                
                if 'ModuleNotFoundError' in stderr or 'ImportError' in stderr:
                    result_type = TestResult.IMPORT_ERROR
                elif any(error in stderr for error in ['NameError', 'AttributeError', 'TypeError']):
                    result_type = TestResult.RUNTIME_ERROR
                else:
                    result_type = TestResult.LOGIC_ERROR
                
                return TestReport(
                    result=result_type,
                    error_message=stderr,
                    stdout=execution_result['stdout'],
                    stderr=stderr,
                    suggestions=self._get_error_suggestions(stderr)
                )
                
        except Exception as e:
            return TestReport(
                result=TestResult.RUNTIME_ERROR,
                error_message=str(e)
            )
    
    def _attempt_fix(self, code: str, test_report: TestReport) -> Tuple[str, str]:
        """Attempt to fix code based on test report"""
        
        # Try each fix strategy
        for strategy_func in self.fix_strategies:
            try:
                fixed_code = strategy_func(code, test_report)
                if fixed_code != code:
                    return fixed_code, strategy_func.__name__
            except Exception as e:
                logger.warning(f"Fix strategy {strategy_func.__name__} failed: {e}")
                continue
        
        return code, "no_fix_applied"
    
    def _apply_learned_fixes(self, code: str, test_report: TestReport) -> str:
        """Apply learned fixes from the learning engine"""
        if not self.learning_engine:
            return code
        
        # Get learned fixes for this error
        learned_fixes = self.learning_engine.get_learned_fixes(test_report.error_message)
        
        if not learned_fixes:
            return code
        
        # Apply the best learned fix
        best_fix = learned_fixes[0]  # Already sorted by confidence
        
        try:
            # Apply the learned fix pattern
            fixed_code = self._apply_fix_pattern(code, best_fix)
            
            if fixed_code != code:
                logger.info(f"Applied learned fix: {best_fix.fix_strategy} (confidence: {best_fix.confidence:.1%})")
                return fixed_code
        except Exception as e:
            logger.warning(f"Failed to apply learned fix: {e}")
        
        return code
    
    def _apply_fix_pattern(self, code: str, fix_pattern) -> str:
        """Apply a specific learned fix pattern"""
        # This is a simplified implementation
        # In practice, this would be more sophisticated based on the fix pattern
        
        if fix_pattern.error_type == 'import_error':
            return self._fix_import_errors(code, None)
        elif fix_pattern.error_type == 'syntax_error':
            return self._fix_syntax_errors(code, None)
        elif fix_pattern.error_type == 'runtime_error':
            return self._fix_runtime_errors(code, None)
        else:
            return code
    
    def _fix_syntax_errors(self, code: str, test_report: TestReport) -> str:
        """Fix common syntax errors"""
        if test_report and test_report.result != TestResult.SYNTAX_ERROR:
            return code
        
        fixed_code = code
        error_msg = test_report.error_message.lower() if test_report else ""
        
        # Fix common indentation issues
        if 'indentation' in error_msg:
            lines = fixed_code.split('\n')
            fixed_lines = []
            for line in lines:
                if line.strip():  # Non-empty line
                    # Ensure proper indentation (4 spaces)
                    stripped = line.lstrip()
                    if line != stripped:  # Was indented
                        fixed_lines.append('    ' + stripped)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            fixed_code = '\n'.join(fixed_lines)
        
        # Fix missing colons
        if 'invalid syntax' in error_msg:
            # Add colons to function definitions, if statements, etc.
            fixed_code = re.sub(r'(def \w+$$[^)]*$$)\s*$', r'\1:', fixed_code, flags=re.MULTILINE)
            fixed_code = re.sub(r'(if .+)\s*$', r'\1:', fixed_code, flags=re.MULTILINE)
            fixed_code = re.sub(r'(for .+ in .+)\s*$', r'\1:', fixed_code, flags=re.MULTILINE)
            fixed_code = re.sub(r'(while .+)\s*$', r'\1:', fixed_code, flags=re.MULTILINE)
            fixed_code = re.sub(r'(class \w+.*)\s*$', r'\1:', fixed_code, flags=re.MULTILINE)
        
        return fixed_code
    
    def _fix_import_errors(self, code: str, test_report: TestReport) -> str:
        """Fix import errors by adding common imports or replacing with alternatives"""
        if test_report and test_report.result != TestResult.IMPORT_ERROR:
            return code
        
        error_msg = test_report.error_message if test_report else ""
        
        # Extract missing module name
        import_match = re.search(r"No module named '(\w+)'", error_msg)
        if not import_match:
            return code
        
        missing_module = import_match.group(1)
        
        # Common module replacements
        replacements = {
            'requests': 'urllib.request',
            'bs4': 'html.parser',
            'numpy': 'math',
            'pandas': 'csv',
        }
        
        if missing_module in replacements:
            # Replace the import
            replacement = replacements[missing_module]
            fixed_code = re.sub(
                rf'import {missing_module}',
                f'import {replacement}',
                code
            )
            fixed_code = re.sub(
                rf'from {missing_module} import',
                f'from {replacement} import',
                fixed_code
            )
            return fixed_code
        
        # Remove the problematic import and add a comment
        fixed_code = re.sub(
            rf'import {missing_module}.*\n',
            f'# import {missing_module}  # Module not available\n',
            code
        )
        
        return fixed_code
    
    def _fix_runtime_errors(self, code: str, test_report: TestReport) -> str:
        """Fix common runtime errors"""
        if test_report and test_report.result != TestResult.RUNTIME_ERROR:
            return code
        
        error_msg = test_report.error_message if test_report else ""
        fixed_code = code
        
        # Fix undefined variables
        name_error_match = re.search(r"NameError: name '(\w+)' is not defined", error_msg)
        if name_error_match:
            undefined_var = name_error_match.group(1)
            
            # Add variable definition at the beginning
            if 'def ' in fixed_code:
                # Add inside the first function
                func_match = re.search(r'(def \w+$$[^)]*$$:\s*\n)', fixed_code)
                if func_match:
                    insertion_point = func_match.end()
                    fixed_code = (fixed_code[:insertion_point] + 
                                f'    {undefined_var} = None  # Auto-generated variable\n' +
                                fixed_code[insertion_point:])
            else:
                # Add at the beginning
                fixed_code = f'{undefined_var} = None  # Auto-generated variable\n\n' + fixed_code
        
        return fixed_code
    
    def _add_error_handling(self, code: str, test_report: TestReport) -> str:
        """Add basic error handling to code"""
        if 'def ' not in code:
            return code
        
        # Wrap main function calls in try-except
        lines = code.split('\n')
        fixed_lines = []
        in_main = False
        
        for line in lines:
            if line.strip() == 'if __name__ == "__main__":':
                in_main = True
                fixed_lines.append(line)
                fixed_lines.append('    try:')
            elif in_main and line.strip() and not line.startswith('    '):
                # End of main block
                fixed_lines.append('    except Exception as e:')
                fixed_lines.append('        print(f"Error: {e}")')
                fixed_lines.append(line)
                in_main = False
            elif in_main and line.strip():
                # Inside main block, add extra indentation
                fixed_lines.append('    ' + line)
            else:
                fixed_lines.append(line)
        
        if in_main:
            # Close the try-except if we're still in main
            fixed_lines.append('    except Exception as e:')
            fixed_lines.append('        print(f"Error: {e}")')
        
        return '\n'.join(fixed_lines)
    
    def _add_basic_structure(self, code: str, test_report: TestReport) -> str:
        """Add basic structure to code if missing"""
        if 'def ' in code or 'class ' in code:
            return code
        
        # Wrap loose code in a main function
        lines = code.split('\n')
        structured_lines = ['def main():']
        
        for line in lines:
            if line.strip():
                structured_lines.append('    ' + line)
            else:
                structured_lines.append(line)
        
        structured_lines.extend([
            '',
            'if __name__ == "__main__":',
            '    main()'
        ])
        
        return '\n'.join(structured_lines)
    
    def _get_syntax_suggestions(self, error_msg: str) -> List[str]:
        """Get suggestions for syntax errors"""
        suggestions = []
        
        if 'indentation' in error_msg.lower():
            suggestions.append("Check indentation - use 4 spaces consistently")
        if 'invalid syntax' in error_msg.lower():
            suggestions.append("Check for missing colons (:) after function/class definitions")
            suggestions.append("Check for missing parentheses or brackets")
        
        return suggestions
    
    def _get_error_suggestions(self, error_msg: str) -> List[str]:
        """Get suggestions based on error patterns"""
        suggestions = []
        
        for pattern, suggestion_func in self.error_patterns.items():
            match = re.search(pattern, error_msg)
            if match:
                suggestions.extend(suggestion_func(match))
        
        return suggestions
    
    def _suggest_import_fix(self, match) -> List[str]:
        module = match.group(1)
        return [f"Install missing module: pip install {module}",
                f"Replace {module} with a built-in alternative"]
    
    def _suggest_variable_fix(self, match) -> List[str]:
        var = match.group(1)
        return [f"Define variable {var} before using it",
                f"Check spelling of variable name {var}"]
    
    def _suggest_indentation_fix(self, match) -> List[str]:
        return ["Use consistent indentation (4 spaces recommended)",
                "Check for mixing tabs and spaces"]
    
    def _suggest_syntax_fix(self, match) -> List[str]:
        return ["Check for missing colons, parentheses, or brackets",
                "Verify proper Python syntax"]
    
    def _suggest_argument_fix(self, match) -> List[str]:
        func = match.group(1)
        return [f"Provide required arguments to {func}()",
                f"Check {func}() function signature"]
    
    def _display_test_report(self, report: TestReport):
        """Display a formatted test report"""
        
        # Status panel
        status_color = "green" if report.result == TestResult.PASS else "red"
        status_text = "✅ PASS" if report.result == TestResult.PASS else f"❌ {report.result.value.upper()}"
        
        panel_content = f"[{status_color}]{status_text}[/{status_color}]"
        
        if report.error_message:
            panel_content += f"\n\n[red]Error:[/red] {report.error_message}"
        
        if report.stdout:
            panel_content += f"\n\n[green]Output:[/green]\n{report.stdout}"
        
        if report.suggestions:
            panel_content += f"\n\n[yellow]Suggestions:[/yellow]"
            for suggestion in report.suggestions:
                panel_content += f"\n• {suggestion}"
        
        console.print(Panel(panel_content, title="Test Report", border_style=status_color))
    
    def generate_fix_summary(self, original_code: str, final_code: str, 
                           fix_attempts: List[FixAttempt], final_test: TestReport) -> str:
        """Generate a summary of what was fixed"""
        
        if not fix_attempts:
            return "No fixes were needed - code worked on first try! ✅"
        
        summary_parts = []
        
        # Overall result
        if final_test.result == TestResult.PASS:
            summary_parts.append(f"✅ Successfully fixed code after {len(fix_attempts)} attempts")
        else:
            summary_parts.append(f"❌ Could not fully fix code after {len(fix_attempts)} attempts")
        
        # List each fix attempt
        summary_parts.append("\n📋 Fix attempts:")
        for attempt in fix_attempts:
            status = "✅" if attempt.success else "❌"
            strategy_name = attempt.fix_strategy.replace('_', ' ').title()
            summary_parts.append(f"  {status} Attempt {attempt.attempt_number}: {strategy_name}")
            if attempt.success:
                summary_parts.append(f"     → Fixed {attempt.test_result.result.value}")
        
        # Learning insights
        if self.learning_engine and fix_attempts:
            learned_patterns = len([a for a in fix_attempts if a.success])
            if learned_patterns > 0:
                summary_parts.append(f"\n🧠 Learning: Recorded {learned_patterns} successful fix patterns")
        
        # Code changes summary
        if final_code != original_code:
            original_lines = len(original_code.split('\n'))
            final_lines = len(final_code.split('\n'))
            line_diff = final_lines - original_lines
            
            summary_parts.append(f"\n📊 Code changes:")
            summary_parts.append(f"  • Lines: {original_lines} → {final_lines} ({line_diff:+d})")
            
            # Detect types of changes
            changes = []
            if 'import' in final_code and 'import' not in original_code:
                changes.append("Added imports")
            if 'try:' in final_code and 'try:' not in original_code:
                changes.append("Added error handling")
            if 'def ' in final_code and 'def ' not in original_code:
                changes.append("Added function structure")
            
            if changes:
                summary_parts.append(f"  • Changes: {', '.join(changes)}")
        
        return '\n'.join(summary_parts)
