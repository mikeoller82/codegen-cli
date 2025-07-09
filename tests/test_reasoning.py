"""
Unit tests for Reasoning Engine
"""

import pytest
from unittest.mock import Mock, patch
from core.reasoning import ReasoningEngine, ReasoningStep, ReasoningChain, ThinkingStep

class TestReasoningEngine:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_ai_engine = Mock()
        self.reasoning_engine = ReasoningEngine(self.mock_ai_engine)
    
    def test_reasoning_engine_initialization(self):
        """Test reasoning engine initializes correctly"""
        assert self.reasoning_engine.ai_engine == self.mock_ai_engine
        assert len(self.reasoning_engine.reasoning_history) == 0
        assert 'step_by_step' in self.reasoning_engine.reasoning_strategies
        assert 'problem_decomposition' in self.reasoning_engine.reasoning_strategies
    
    def test_reasoning_step_creation(self):
        """Test creating reasoning steps"""
        step = ReasoningStep(
            step_type=ThinkingStep.ANALYSIS,
            title="Test Analysis",
            content="This is a test analysis step",
            confidence=0.8
        )
        
        assert step.step_type == ThinkingStep.ANALYSIS
        assert step.title == "Test Analysis"
        assert step.confidence == 0.8
        assert step.dependencies == []
        assert step.outputs == []
    
    def test_reasoning_chain_creation(self):
        """Test creating reasoning chains"""
        steps = [
            ReasoningStep(ThinkingStep.ANALYSIS, "Analysis", "Test analysis"),
            ReasoningStep(ThinkingStep.PLANNING, "Planning", "Test planning")
        ]
        
        chain = ReasoningChain(
            problem="Test problem",
            steps=steps,
            final_solution="Test solution",
            confidence=0.85
        )
        
        assert chain.problem == "Test problem"
        assert len(chain.steps) == 2
        assert chain.confidence == 0.85
        assert chain.metadata == {}
    
    @patch('core.reasoning.console.print')
    def test_step_by_step_reasoning(self, mock_print):
        """Test step-by-step reasoning strategy"""
        problem = "Create a simple calculator"
        
        steps = self.reasoning_engine._step_by_step_reasoning(problem, show_thinking=False)
        
        assert len(steps) == 4  # Analysis, Planning, Implementation, Validation
        assert steps[0].step_type == ThinkingStep.ANALYSIS
        assert steps[1].step_type == ThinkingStep.PLANNING
        assert steps[2].step_type == ThinkingStep.IMPLEMENTATION
        assert steps[3].step_type == ThinkingStep.VALIDATION
    
    def test_problem_categorization(self):
        """Test problem categorization"""
        web_problem = "create a REST API server"
        algo_problem = "implement bubble sort algorithm"
        data_problem = "design a database schema"
        
        assert "web development" in self.reasoning_engine._categorize_problem(web_problem)
        assert "algorithmic" in self.reasoning_engine._categorize_problem(algo_problem)
        assert "data management" in self.reasoning_engine._categorize_problem(data_problem)
    
    @patch('core.reasoning.console.print')
    def test_reason_through_problem(self, mock_print):
        """Test complete reasoning process"""
        problem = "Create a web scraper"
        
        chain = self.reasoning_engine.reason_through_problem(
            problem, strategy='step_by_step', show_thinking=False
        )
        
        assert isinstance(chain, ReasoningChain)
        assert chain.problem == problem
        assert len(chain.steps) > 0
        assert chain.confidence > 0
        assert chain.reasoning_time >= 0
        
        # Check that it's stored in history
        assert len(self.reasoning_engine.reasoning_history) == 1
        assert self.reasoning_engine.reasoning_history[0] == chain
    
    def test_export_reasoning(self):
        """Test exporting reasoning chain"""
        steps = [ReasoningStep(ThinkingStep.ANALYSIS, "Test", "Content")]
        chain = ReasoningChain("Problem", steps, "Solution", 0.8)
        
        exported = self.reasoning_engine.export_reasoning(chain)
        
        assert exported['problem'] == "Problem"
        assert exported['final_solution'] == "Solution"
        assert exported['confidence'] == 0.8
        assert len(exported['steps']) == 1
        assert exported['steps'][0]['type'] == 'analysis'
    
    def test_reasoning_strategies_exist(self):
        """Test all reasoning strategies are implemented"""
        expected_strategies = [
            'step_by_step',
            'problem_decomposition', 
            'first_principles',
            'analogical',
            'critical_thinking'
        ]
        
        for strategy in expected_strategies:
            assert strategy in self.reasoning_engine.reasoning_strategies
            # Test that the strategy function exists and is callable
            strategy_func = self.reasoning_engine.reasoning_strategies[strategy]
            assert callable(strategy_func)
