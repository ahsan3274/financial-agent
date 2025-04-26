"""
Module to Improve categorization
"""

from typing import Dict, Any
from memory.long_term import LongTermMemory

class SelfImprovementEngine:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.performance_metrics = {}
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze historical errors to identify improvement opportunities"""
        errors = self.memory.get_items_by_type('error')
        error_analysis = {}
        
        # Categorize errors by type and source
        for error in errors:
            error_type = error.get('type', 'unknown')
            error_analysis.setdefault(error_type, {
                'count': 0,
                'sources': set(),
                'recent_examples': []
            })
            error_analysis[error_type]['count'] += 1
            error_analysis[error_type]['sources'].add(error.get('source'))
            error_analysis[error_type]['recent_examples'].append(error['message'])
        
        return error_analysis

    def generate_improvements(self) -> Dict[str, Any]:
        """Generate actionable improvement plans"""
        analysis = self.analyze_errors()
        improvements = []
        
        # Example improvement logic
        if 'categorization_error' in analysis:
            improvements.append({
                'type': 'category_rules',
                'action': 'Add new rules based on common miscategorizations',
                'target': 'CategoryManager',
                'priority': 'high'
            })
        
        return {
            'performance_report': analysis,
            'recommended_actions': improvements
        }