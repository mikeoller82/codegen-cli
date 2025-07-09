"""
Learning Engine for improving AI code generation based on successful fixes
"""

import logging
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger("codegen.learning")

@dataclass
class FixPattern:
    """A pattern learned from successful fixes"""
    pattern_id: str
    error_type: str
    error_pattern: str
    fix_strategy: str
    fix_pattern: str
    success_count: int = 1
    failure_count: int = 0
    confidence: float = 0.0
    last_used: datetime = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = datetime.now()
        self.update_confidence()
    
    def update_confidence(self):
        """Update confidence based on success/failure ratio"""
        total = self.success_count + self.failure_count
        if total > 0:
            self.confidence = self.success_count / total
        else:
            self.confidence = 0.0

@dataclass
class GenerationImprovement:
    """An improvement learned from generation patterns"""
    improvement_id: str
    prompt_pattern: str
    original_issue: str
    improved_approach: str
    code_template: str
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class LearningMetrics:
    """Metrics about the learning system"""
    total_patterns: int
    active_patterns: int
    avg_confidence: float
    patterns_by_type: Dict[str, int]
    recent_improvements: int
    learning_rate: float

class LearningDatabase:
    """Database for storing learning patterns and improvements"""
    
    def __init__(self, db_path: str = "learning.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fix patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fix_patterns (
                pattern_id TEXT PRIMARY KEY,
                error_type TEXT NOT NULL,
                error_pattern TEXT NOT NULL,
                fix_strategy TEXT NOT NULL,
                fix_pattern TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                failure_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Generation improvements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_improvements (
                improvement_id TEXT PRIMARY KEY,
                prompt_pattern TEXT NOT NULL,
                original_issue TEXT NOT NULL,
                improved_approach TEXT NOT NULL,
                code_template TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Learning sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                session_start TIMESTAMP,
                patterns_learned INTEGER DEFAULT 0,
                improvements_discovered INTEGER DEFAULT 0,
                total_fixes INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_fix_pattern(self, pattern: FixPattern):
        """Save or update a fix pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO fix_patterns 
            (pattern_id, error_type, error_pattern, fix_strategy, fix_pattern,
             success_count, failure_count, confidence, last_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id, pattern.error_type, pattern.error_pattern,
            pattern.fix_strategy, pattern.fix_pattern, pattern.success_count,
            pattern.failure_count, pattern.confidence, pattern.last_used,
            pattern.created_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_fix_patterns(self, error_type: str = None, min_confidence: float = 0.5) -> List[FixPattern]:
        """Get fix patterns, optionally filtered by error type and confidence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT pattern_id, error_type, error_pattern, fix_strategy, fix_pattern,
                   success_count, failure_count, confidence, last_used, created_at
            FROM fix_patterns
            WHERE confidence >= ?
        '''
        params = [min_confidence]
        
        if error_type:
            query += ' AND error_type = ?'
            params.append(error_type)
        
        query += ' ORDER BY confidence DESC, success_count DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            pattern = FixPattern(
                pattern_id=row[0], error_type=row[1], error_pattern=row[2],
                fix_strategy=row[3], fix_pattern=row[4], success_count=row[5],
                failure_count=row[6], confidence=row[7], last_used=row[8],
                created_at=row[9]
            )
            patterns.append(pattern)
        
        return patterns
    
    def save_generation_improvement(self, improvement: GenerationImprovement):
        """Save a generation improvement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO generation_improvements
            (improvement_id, prompt_pattern, original_issue, improved_approach,
             code_template, usage_count, success_rate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            improvement.improvement_id, improvement.prompt_pattern,
            improvement.original_issue, improvement.improved_approach,
            improvement.code_template, improvement.usage_count,
            improvement.success_rate, improvement.created_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_generation_improvements(self, prompt_pattern: str = None) -> List[GenerationImprovement]:
        """Get generation improvements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT improvement_id, prompt_pattern, original_issue, improved_approach,
                   code_template, usage_count, success_rate, created_at
            FROM generation_improvements
        '''
        params = []
        
        if prompt_pattern:
            query += ' WHERE prompt_pattern LIKE ?'
            params.append(f'%{prompt_pattern}%')
        
        query += ' ORDER BY success_rate DESC, usage_count DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        improvements = []
        for row in rows:
            improvement = GenerationImprovement(
                improvement_id=row[0], prompt_pattern=row[1], original_issue=row[2],
                improved_approach=row[3], code_template=row[4], usage_count=row[5],
                success_rate=row[6], created_at=row[7]
            )
            improvements.append(improvement)
        
        return improvements

class LearningEngine:
    """Engine for learning from successful fixes and improving future generations"""
    
    def __init__(self, ai_engine=None):
        self.ai_engine = ai_engine
        self.db = LearningDatabase()
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now()
        self.patterns_learned = 0
        self.improvements_discovered = 0
        
        # Pattern extraction rules
        self.error_extractors = {
            'syntax_error': self._extract_syntax_patterns,
            'import_error': self._extract_import_patterns,
            'runtime_error': self._extract_runtime_patterns,
            'logic_error': self._extract_logic_patterns
        }
        
        # Code improvement templates
        self.improvement_templates = {
            'web_scraper': self._get_web_scraper_template,
            'api_server': self._get_api_server_template,
            'data_processing': self._get_data_processing_template,
            'algorithm': self._get_algorithm_template,
            'class_design': self._get_class_design_template
        }
    
    def learn_from_fix(self, original_code: str, fixed_code: str, error_message: str, 
                      fix_strategy: str, success: bool) -> Optional[FixPattern]:
        """Learn from a successful or failed fix attempt"""
        
        if not success:
            # Update failure count for existing patterns
            self._update_pattern_failure(error_message, fix_strategy)
            return None
        
        # Extract error and fix patterns
        error_type = self._classify_error(error_message)
        error_pattern = self._extract_error_pattern(error_message, error_type)
        fix_pattern = self._extract_fix_pattern(original_code, fixed_code, fix_strategy)
        
        if not error_pattern or not fix_pattern:
            return None
        
        # Create or update pattern
        pattern_id = self._generate_pattern_id(error_type, error_pattern, fix_strategy)
        existing_pattern = self._get_existing_pattern(pattern_id)
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.success_count += 1
            existing_pattern.last_used = datetime.now()
            existing_pattern.update_confidence()
            self.db.save_fix_pattern(existing_pattern)
            logger.info(f"Updated fix pattern: {pattern_id}")
            return existing_pattern
        else:
            # Create new pattern
            new_pattern = FixPattern(
                pattern_id=pattern_id,
                error_type=error_type,
                error_pattern=error_pattern,
                fix_strategy=fix_strategy,
                fix_pattern=fix_pattern
            )
            self.db.save_fix_pattern(new_pattern)
            self.patterns_learned += 1
            logger.info(f"Learned new fix pattern: {pattern_id}")
            return new_pattern
    
    def learn_from_generation(self, prompt: str, generated_code: str, 
                            fix_attempts: List, final_success: bool) -> Optional[GenerationImprovement]:
        """Learn from the entire generation process to improve future generations"""
        
        if not final_success or not fix_attempts:
            return None
        
        # Analyze what went wrong initially and how it was fixed
        prompt_category = self._categorize_prompt(prompt)
        common_issues = self._analyze_common_issues(fix_attempts)
        
        if not common_issues:
            return None
        
        # Create improvement based on learned patterns
        improvement_id = self._generate_improvement_id(prompt_category, common_issues)
        improved_approach = self._generate_improved_approach(prompt, fix_attempts)
        code_template = self._extract_code_template(generated_code, fix_attempts)
        
        improvement = GenerationImprovement(
            improvement_id=improvement_id,
            prompt_pattern=prompt_category,
            original_issue='; '.join(common_issues),
            improved_approach=improved_approach,
            code_template=code_template
        )
        
        self.db.save_generation_improvement(improvement)
        self.improvements_discovered += 1
        logger.info(f"Discovered generation improvement: {improvement_id}")
        return improvement
    
    def get_learned_fixes(self, error_message: str) -> List[FixPattern]:
        """Get learned fix patterns that might apply to an error"""
        error_type = self._classify_error(error_message)
        patterns = self.db.get_fix_patterns(error_type=error_type, min_confidence=0.6)
        
        # Filter patterns that match the specific error
        matching_patterns = []
        for pattern in patterns:
            if self._pattern_matches_error(pattern.error_pattern, error_message):
                matching_patterns.append(pattern)
        
        # Sort by confidence and recency
        matching_patterns.sort(key=lambda p: (p.confidence, p.last_used), reverse=True)
        return matching_patterns[:3]  # Return top 3 matches
    
    def improve_prompt(self, original_prompt: str) -> Tuple[str, Optional[str]]:
        """Improve a prompt based on learned patterns"""
        prompt_category = self._categorize_prompt(original_prompt)
        improvements = self.db.get_generation_improvements(prompt_pattern=prompt_category)
        
        if not improvements:
            return original_prompt, None
        
        # Find the best improvement
        best_improvement = max(improvements, key=lambda i: i.success_rate * i.usage_count + 1)
        
        # Apply the improvement
        improved_prompt = self._apply_prompt_improvement(original_prompt, best_improvement)
        
        # Update usage count
        best_improvement.usage_count += 1
        self.db.save_generation_improvement(best_improvement)
        
        return improved_prompt, best_improvement.improved_approach
    
    def get_code_template(self, prompt: str) -> Optional[str]:
        """Get a learned code template for a prompt"""
        prompt_category = self._categorize_prompt(prompt)
        
        # Check learned templates first
        improvements = self.db.get_generation_improvements(prompt_pattern=prompt_category)
        if improvements:
            best_template = max(improvements, key=lambda i: i.success_rate)
            if best_template.success_rate > 0.7:  # High confidence threshold
                return best_template.code_template
        
        # Fall back to built-in templates
        template_func = self.improvement_templates.get(prompt_category)
        if template_func:
            return template_func(prompt)
        
        return None
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Get metrics about the learning system"""
        patterns = self.db.get_fix_patterns(min_confidence=0.0)  # Get all patterns
        
        total_patterns = len(patterns)
        active_patterns = len([p for p in patterns if p.confidence >= 0.5])
        avg_confidence = sum(p.confidence for p in patterns) / total_patterns if total_patterns > 0 else 0.0
        
        patterns_by_type = Counter(p.error_type for p in patterns)
        
        # Recent improvements (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_improvements = len([p for p in patterns if p.created_at >= week_ago])
        
        # Learning rate (patterns learned per session)
        learning_rate = self.patterns_learned / max(1, (datetime.now() - self.session_start).total_seconds() / 3600)
        
        return LearningMetrics(
            total_patterns=total_patterns,
            active_patterns=active_patterns,
            avg_confidence=avg_confidence,
            patterns_by_type=dict(patterns_by_type),
            recent_improvements=recent_improvements,
            learning_rate=learning_rate
        )
    
    def display_learning_status(self):
        """Display current learning status"""
        metrics = self.get_learning_metrics()
        
        # Main metrics table
        metrics_table = Table(title="🧠 AI Learning Status")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        
        metrics_table.add_row("Total Patterns Learned", str(metrics.total_patterns))
        metrics_table.add_row("Active Patterns", str(metrics.active_patterns))
        metrics_table.add_row("Average Confidence", f"{metrics.avg_confidence:.1%}")
        metrics_table.add_row("Recent Improvements", str(metrics.recent_improvements))
        metrics_table.add_row("Learning Rate", f"{metrics.learning_rate:.2f}/hour")
        
        console.print(metrics_table)
        
        # Patterns by type
        if metrics.patterns_by_type:
            console.print("\n[bold blue]📊 Patterns by Error Type:[/bold blue]")
            type_table = Table()
            type_table.add_column("Error Type", style="cyan")
            type_table.add_column("Patterns", style="white")
            type_table.add_column("Percentage", style="green")
            
            total = sum(metrics.patterns_by_type.values())
            for error_type, count in sorted(metrics.patterns_by_type.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total * 100 if total > 0 else 0
                type_table.add_row(error_type, str(count), f"{percentage:.1f}%")
            
            console.print(type_table)
        
        # Recent high-confidence patterns
        recent_patterns = self.db.get_fix_patterns(min_confidence=0.8)[:5]
        if recent_patterns:
            console.print("\n[bold green]🎯 Top Learned Patterns:[/bold green]")
            pattern_table = Table()
            pattern_table.add_column("Error Type", style="cyan")
            pattern_table.add_column("Fix Strategy", style="yellow")
            pattern_table.add_column("Confidence", style="green")
            pattern_table.add_column("Uses", style="white")
            
            for pattern in recent_patterns:
                confidence_bar = "█" * int(pattern.confidence * 10) + "░" * (10 - int(pattern.confidence * 10))
                pattern_table.add_row(
                    pattern.error_type,
                    pattern.fix_strategy,
                    f"{confidence_bar} {pattern.confidence:.1%}",
                    str(pattern.success_count)
                )
            
            console.print(pattern_table)
    
    # Helper methods
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error"""
        error_lower = error_message.lower()
        
        if 'syntaxerror' in error_lower or 'indentationerror' in error_lower:
            return 'syntax_error'
        elif 'modulenotfounderror' in error_lower or 'importerror' in error_lower:
            return 'import_error'
        elif any(err in error_lower for err in ['nameerror', 'attributeerror', 'typeerror']):
            return 'runtime_error'
        else:
            return 'logic_error'
    
    def _extract_error_pattern(self, error_message: str, error_type: str) -> str:
        """Extract a pattern from the error message"""
        extractor = self.error_extractors.get(error_type)
        if extractor:
            return extractor(error_message)
        return error_message[:100]  # Fallback to first 100 chars
    
    def _extract_syntax_patterns(self, error_message: str) -> str:
        """Extract patterns from syntax errors"""
        # Common syntax error patterns
        patterns = [
            r"invalid syntax.*line (\d+)",
            r"expected ':' after '(.*)'",
            r"IndentationError: (.*)",
            r"unexpected EOF while parsing"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return pattern
        
        return "generic_syntax_error"
    
    def _extract_import_patterns(self, error_message: str) -> str:
        """Extract patterns from import errors"""
        # Extract module name
        match = re.search(r"No module named '(\w+)'", error_message)
        if match:
            return f"missing_module_{match.group(1)}"
        
        return "generic_import_error"
    
    def _extract_runtime_patterns(self, error_message: str) -> str:
        """Extract patterns from runtime errors"""
        # Common runtime error patterns
        if "NameError" in error_message:
            match = re.search(r"name '(\w+)' is not defined", error_message)
            if match:
                return f"undefined_variable_{match.group(1)}"
        
        if "AttributeError" in error_message:
            return "attribute_error"
        
        if "TypeError" in error_message:
            return "type_error"
        
        return "generic_runtime_error"
    
    def _extract_logic_patterns(self, error_message: str) -> str:
        """Extract patterns from logic errors"""
        return "logic_error"
    
    def _extract_fix_pattern(self, original_code: str, fixed_code: str, fix_strategy: str) -> str:
        """Extract the pattern of what was fixed"""
        # Analyze the differences between original and fixed code
        original_lines = original_code.split('\n')
        fixed_lines = fixed_code.split('\n')
        
        # Simple diff analysis
        if len(fixed_lines) > len(original_lines):
            return f"added_lines_{fix_strategy}"
        elif len(fixed_lines) < len(original_lines):
            return f"removed_lines_{fix_strategy}"
        else:
            return f"modified_lines_{fix_strategy}"
    
    def _generate_pattern_id(self, error_type: str, error_pattern: str, fix_strategy: str) -> str:
        """Generate a unique pattern ID"""
        content = f"{error_type}_{error_pattern}_{fix_strategy}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_existing_pattern(self, pattern_id: str) -> Optional[FixPattern]:
        """Get an existing pattern by ID"""
        patterns = self.db.get_fix_patterns(min_confidence=0.0)
        for pattern in patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None
    
    def _update_pattern_failure(self, error_message: str, fix_strategy: str):
        """Update failure count for patterns that didn't work"""
        error_type = self._classify_error(error_message)
        patterns = self.db.get_fix_patterns(error_type=error_type)
        
        for pattern in patterns:
            if pattern.fix_strategy == fix_strategy:
                pattern.failure_count += 1
                pattern.update_confidence()
                self.db.save_fix_pattern(pattern)
                break
    
    def _categorize_prompt(self, prompt: str) -> str:
        """Categorize a prompt to determine its type"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['scraper', 'scrape', 'crawl', 'web']):
            return 'web_scraper'
        elif any(word in prompt_lower for word in ['api', 'server', 'flask', 'fastapi', 'rest']):
            return 'api_server'
        elif any(word in prompt_lower for word in ['data', 'csv', 'json', 'process', 'parse']):
            return 'data_processing'
        elif any(word in prompt_lower for word in ['algorithm', 'sort', 'search', 'tree', 'graph']):
            return 'algorithm'
        elif any(word in prompt_lower for word in ['class', 'object', 'inherit', 'method']):
            return 'class_design'
        else:
            return 'general'
    
    def _analyze_common_issues(self, fix_attempts: List) -> List[str]:
        """Analyze common issues from fix attempts"""
        issues = []
        for attempt in fix_attempts:
            if hasattr(attempt, 'original_error'):
                error_type = self._classify_error(attempt.original_error)
                issues.append(error_type)
        
        # Return unique issues
        return list(set(issues))
    
    def _generate_improvement_id(self, prompt_category: str, common_issues: List[str]) -> str:
        """Generate improvement ID"""
        content = f"{prompt_category}_{'_'.join(sorted(common_issues))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_improved_approach(self, prompt: str, fix_attempts: List) -> str:
        """Generate an improved approach based on fix attempts"""
        improvements = []
        
        for attempt in fix_attempts:
            if hasattr(attempt, 'fix_strategy'):
                if attempt.fix_strategy == '_fix_import_errors':
                    improvements.append("Use built-in modules instead of external dependencies")
                elif attempt.fix_strategy == '_fix_syntax_errors':
                    improvements.append("Add proper indentation and syntax structure")
                elif attempt.fix_strategy == '_add_error_handling':
                    improvements.append("Include comprehensive error handling from the start")
        
        return "; ".join(improvements) if improvements else "Generate more robust code structure"
    
    def _extract_code_template(self, generated_code: str, fix_attempts: List) -> str:
        """Extract a reusable code template"""
        # This is a simplified template extraction
        # In practice, this would be more sophisticated
        lines = generated_code.split('\n')
        
        # Extract function/class structure
        template_lines = []
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                template_lines.append(line)
            elif line.strip().startswith('#'):
                template_lines.append(line)
        
        return '\n'.join(template_lines) if template_lines else generated_code[:200]
    
    def _pattern_matches_error(self, pattern: str, error_message: str) -> bool:
        """Check if a pattern matches an error message"""
        # Simple pattern matching - could be enhanced with regex
        return pattern.lower() in error_message.lower()
    
    def _apply_prompt_improvement(self, original_prompt: str, improvement: GenerationImprovement) -> str:
        """Apply an improvement to a prompt"""
        # Add the improved approach as context
        improved_prompt = f"{original_prompt}\n\nBased on learned patterns, please: {improvement.improved_approach}"
        return improved_prompt
    
    # Built-in code templates
    def _get_web_scraper_template(self, prompt: str) -> str:
        return '''
import requests
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Optional

class WebScraper:
    def __init__(self, base_url: str, delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; WebScraper/1.0)'
        })
    
    def fetch_page(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)  # Rate limiting
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def scrape(self, urls: List[str]) -> List[Dict]:
        results = []
        for url in urls:
            content = self.fetch_page(url)
            if content:
                # Process content here
                results.append({"url": url, "content": content})
        return results
'''
    
    def _get_api_server_template(self, prompt: str) -> str:
        return '''
from flask import Flask, request, jsonify
from typing import Dict, Any
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    try:
        if request.method == 'GET':
            # Handle GET request
            return jsonify({"message": "GET request handled"})
        elif request.method == 'POST':
            # Handle POST request
            data = request.get_json()
            return jsonify({"message": "POST request handled", "data": data})
    except Exception as e:
        app.logger.error(f"Error handling request: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    def _get_data_processing_template(self, prompt: str) -> str:
        return '''
import csv
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def load_csv(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.data = list(reader)
            return self.data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []
    
    def load_json(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            return self.data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []
    
    def process_data(self) -> List[Dict]:
        # Process the loaded data
        processed = []
        for item in self.data:
            # Add processing logic here
            processed.append(item)
        return processed
    
    def save_results(self, data: List[Dict], output_path: str):
        try:
            if output_path.endswith('.json'):
                with open(output_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2)
            elif output_path.endswith('.csv'):
                if data:
                    with open(output_path, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
        except Exception as e:
            print(f"Error saving results: {e}")
'''
    
    def _get_algorithm_template(self, prompt: str) -> str:
        return '''
from typing import List, Any, Optional, Tuple
import time

class Algorithm:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def reset_stats(self):
        self.comparisons = 0
        self.swaps = 0
    
    def get_stats(self) -> Dict[str, int]:
        return {"comparisons": self.comparisons, "swaps": self.swaps}
    
    def sort(self, data: List[Any]) -> List[Any]:
        """Override this method with specific sorting algorithm"""
        raise NotImplementedError
    
    def search(self, data: List[Any], target: Any) -> Optional[int]:
        """Override this method with specific search algorithm"""
        raise NotImplementedError
    
    def benchmark(self, data: List[Any]) -> Dict[str, Any]:
        """Benchmark the algorithm performance"""
        self.reset_stats()
        start_time = time.time()
        
        result = self.sort(data.copy())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "execution_time": execution_time,
            "result_length": len(result),
            **self.get_stats()
        }
'''
    
    def _get_class_design_template(self, prompt: str) -> str:
        return '''
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BaseModel:
    """Base model with common functionality"""
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def update(self):
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class BaseService(ABC):
    """Base service class with common patterns"""
    
    def __init__(self):
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Override this method with specific processing logic"""
        pass
    
    def validate_input(self, data: Any) -> bool:
        """Override this method with specific validation logic"""
        return True
    
    def handle_error(self, error: Exception) -> None:
        """Common error handling"""
        self.logger.error(f"Error in {self.__class__.__name__}: {error}")
'''
