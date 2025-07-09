"""
Unit tests for REPL functionality
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from core.repl import REPLSession, CodeGenCompleter

class TestREPLSession:
    def setup_method(self):
        """Set up test fixtures"""
        self.repl = REPLSession()
        self.repl.running = False  # Prevent actual REPL loop in tests
    
    def test_command_aliases(self):
        """Test command aliases work correctly"""
        assert self.repl.aliases['gen'] == 'generate'
        assert self.repl.aliases['mem'] == 'memory'
        assert self.repl.aliases['cls'] == 'clear'
        assert self.repl.aliases['q'] == 'quit'
    
    def test_config_management(self):
        """Test configuration management"""
        # Test default config
        assert 'verbose' in self.repl.config
        assert 'debug' in self.repl.config
        assert 'default_model' in self.repl.config
        
        # Test config update
        original_verbose = self.repl.config['verbose']
        self.repl.config['verbose'] = not original_verbose
        assert self.repl.config['verbose'] != original_verbose
    
    def test_variables_storage(self):
        """Test session variables storage"""
        self.repl.variables['test_var'] = 'test_value'
        assert self.repl.variables['test_var'] == 'test_value'
        
        # Test clearing variables
        self.repl.variables.clear()
        assert len(self.repl.variables) == 0
    
    @patch('core.repl.console.print')
    def test_cmd_pwd(self, mock_print):
        """Test pwd command"""
        self.repl.cmd_pwd([])
        mock_print.assert_called()
    
    @patch('core.repl.console.print')
    def test_cmd_models(self, mock_print):
        """Test models command"""
        self.repl.cmd_models([])
        mock_print.assert_called()
    
    def test_get_prompt_text(self):
        """Test prompt text generation"""
        prompt = self.repl.get_prompt_text()
        assert 'codegen' in str(prompt)
        assert self.repl.config['default_model'] in str(prompt)

class TestCodeGenCompleter:
    def setup_method(self):
        """Set up test fixtures"""
        self.completer = CodeGenCompleter()
    
    def test_commands_list(self):
        """Test that completer has expected commands"""
        expected_commands = ['generate', 'read', 'write', 'help', 'quit']
        for cmd in expected_commands:
            assert cmd in self.completer.commands
    
    def test_completer_initialization(self):
        """Test completer initializes correctly"""
        assert self.completer.commands is not None
        assert self.completer.path_completer is not None
        assert self.completer.word_completer is not None
