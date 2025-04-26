"""
Long-term memory module for the financial agent.
Provides short-term storage for expense data and user preferences.
"""
from typing import List, Dict, Any

class ShortTermMemory:
    def __init__(self, max_items=10):
        self.max_items = max_items
        self.items = []
    
    def add_item(self, item: Dict[str, Any]) -> None:
        if len(self.items) >= self.max_items:
            self.items.pop(0)
        self.items.append(item)
    
    def get_items(self) -> List[Dict[str, Any]]:
        return self.items.copy()
    
    def clear(self) -> None:
        self.items = []




