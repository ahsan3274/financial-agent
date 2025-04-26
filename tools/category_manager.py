"""
/tools/category_manager.py
Category manager module for handling expense categorization and rules.
"""
import pandas as pd
import re
from typing import Dict, List, Optional, Any, Tuple, Set
import json
import os
from datetime import datetime
from utils.llm_utils import LLMManager 
import logging


logger = logging.getLogger(__name__)

class CategoryManager:
    # MODIFY __init__ signature
    def __init__(self, storage_path: str = "data/categories_rules.json", llm_manager: Optional[LLMManager] = None):
        """
        Initialize the CategoryManager, loading state from storage.

        Args:
            storage_path: Path to the JSON file for storing categories and rules.
            llm_manager: Optional LLMManager instance for LLM features.
        """
        self.storage_path = storage_path
        self.llm = llm_manager
        self.categories = {}  # Will be loaded
        self.category_rules = {} # Will be loaded
        self.rule_cache = {}
        self.auto_learn = True # Make this configurable?

        self._load_state() # Load state on initialization

        # LLM Integration
        self.llm = llm_manager
        self.rule_cache = {}
        self.auto_learn = True

    def categorize_transaction(self, description: str, notes: str = "") -> str:
        """
        Hybrid categorization flow:
        1. Check exact rule matches
        2. Try LLM categorization within existing taxonomy
        3. Validate against category definitions
        4. Fallback to 'other' if uncertain
        """
        description = description.lower()
        notes = notes.lower()
        
        # Check rule cache first
        if (description, notes) in self.rule_cache:
            return self.rule_cache[(description, notes)]
        
        # 1. Rule-based categorization
        category = self._rule_based_categorize(description)
        if category != 'other':
            return category
            
        # 2. LLM categorization
        llm_category = self._llm_categorize(f"{description} ({notes})")
        
        # 3. Validation against taxonomy
        if self._is_valid_category(llm_category, description, notes):
            if self.auto_learn:
                self._add_rule_from_llm(llm_category, description)
            return llm_category
            
        # 4. Final fallback
        return 'other'

    def _rule_based_categorize(self, description: str) -> str:
        """Your original rule-based logic"""
        for category, keywords in self.category_rules.items():
            if any(keyword in description for keyword in keywords):
                return category
        return 'other'

    def _llm_categorize(self, transaction_text: str) -> str:
        """LLM-powered categorization constrained to existing taxonomy"""
        prompt = f"""Categorize this transaction using ONLY these categories:
        {self._format_categories_for_prompt()}
        
        Transaction: {transaction_text}
        
        Respond with ONLY the category key from the list above."""
        
        return self.llm.query(prompt, temperature=0.2).strip().lower()

    def _is_valid_category(self, category: str, description: str, notes: str) -> bool:
        """Validate LLM suggestion against category definition"""
        if category not in self.categories:
            return False
            
        validation_prompt = f"""Does this transaction match the category definition?
        Category: {self.categories[category]['name']} ({self.categories[category]['description']})
        Transaction: {description}
        Notes: {notes}
        
        Answer with ONLY 'yes' or 'no'."""
        
        return self.llm.query(validation_prompt).lower().strip() == 'yes'

    def _add_rule_from_llm(self, category: str, description: str):
        """Automatically learn from validated LLM categorizations"""
        new_keyword = self._extract_keyword(description)
        if new_keyword and new_keyword not in self.category_rules[category]:
            self.category_rules[category].append(new_keyword)
            self.rule_cache[(description, '')] = category
            self._save_state()

    def _format_categories_for_prompt(self) -> str:
        """Format categories for LLM prompt"""
        return '\n'.join(
            f"- {key}: {value['description']}" 
            for key, value in self.categories.items()
        )
    
    
    
    def get_categories(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available categories.
        
        Returns:
            Dictionary of category details
        """
        return self.categories
    
    def add_category(self, category_id: str, name: str, description: str = "") -> Dict[str, str]:
        """
        Add a new category.
        
        Args:
            category_id: Unique identifier for the category
            name: Display name for the category
            description: Optional description
            
        Returns:
            Dictionary with the added category details
        """
        category_id = category_id.lower().strip()
        
        if category_id in self.categories:
            return {"error": f"Category '{category_id}' already exists"}
        
        self.categories[category_id] = {
            "name": name,
            "description": description
        }
        
        # Initialize empty rules list for the new category
        self.category_rules[category_id] = []
        
        self._save_state()
        return {"success": f"Category '{name}' added successfully", "category_id": category_id}
    
    def update_category(self, category_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, str]:
        """
        Update an existing category.
        
        Args:
            category_id: Unique identifier for the category
            name: New display name (optional)
            description: New description (optional)
            
        Returns:
            Dictionary with result of the update operation
        """
        category_id = category_id.lower().strip()
        
        if category_id not in self.categories:
            return {"error": f"Category '{category_id}' does not exist"}
        
        if name is not None:
            self.categories[category_id]["name"] = name
        
        if description is not None:
            self.categories[category_id]["description"] = description
        
        self._save_state()
        return {"success": f"Category '{category_id}' updated successfully"}
    
    def delete_category(self, category_id: str) -> Dict[str, str]:
        """
        Delete a category.
        
        Args:
            category_id: Unique identifier for the category
            
        Returns:
            Dictionary with result of the delete operation
        """
        category_id = category_id.lower().strip()
        
        if category_id not in self.categories:
            return {"error": f"Category '{category_id}' does not exist"}
        
        # Don't allow deletion of the "other" category
        if category_id == "other":
            return {"error": "Cannot delete the 'other' category"}
        
        del self.categories[category_id]
        
        # Also remove category rules
        if category_id in self.category_rules:
            del self.category_rules[category_id]
        
        self._save_state()
        return {"success": f"Category '{category_id}' deleted successfully"}
    
    def get_category_rules(self, category_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rules for a specific category or all categories.
        
        Args:
            category_id: Optional category ID to filter rules
            
        Returns:
            Dictionary with category rules
        """
        if category_id:
            category_id = category_id.lower().strip()
            if category_id not in self.category_rules:
                return {"error": f"Category '{category_id}' does not exist"}
            return {category_id: self.category_rules[category_id]}
        
        return self.category_rules
    
    def add_category_rule(self, category_id: str, rule: str) -> Dict[str, str]:
        """
        Add a new rule for a category.
        
        Args:
            category_id: Category ID to add the rule to
            rule: Keyword or pattern to match
            
        Returns:
            Dictionary with result of the operation
        """
        category_id = category_id.lower().strip()
        rule = rule.lower().strip()
        
        if category_id not in self.categories:
            return {"error": f"Category '{category_id}' does not exist"}
        
        if rule in self.category_rules.get(category_id, []):
            return {"error": f"Rule '{rule}' already exists for category '{category_id}'"}
        
        # Initialize category rules list if it doesn't exist
        if category_id not in self.category_rules:
            self.category_rules[category_id] = []
        
        self.category_rules[category_id].append(rule)
        self._save_state()
        return {"success": f"Rule '{rule}' added to category '{category_id}'"}
    
    def delete_category_rule(self, category_id: str, rule: str) -> Dict[str, str]:
        """
        Delete a rule from a category.
        
        Args:
            category_id: Category ID to remove the rule from
            rule: Rule to remove
            
        Returns:
            Dictionary with result of the operation
        """
        category_id = category_id.lower().strip()
        rule = rule.lower().strip()
        
        if category_id not in self.category_rules:
            return {"error": f"Category '{category_id}' does not exist"}
        
        if rule not in self.category_rules[category_id]:
            return {"error": f"Rule '{rule}' does not exist for category '{category_id}'"}
        
        self.category_rules[category_id].remove(rule)
        self._save_state()
        
        return {"success": f"Rule '{rule}' removed from category '{category_id}'"}
    
    def categorize_transaction(self, description: str, amount: float = 0.0) -> str:
        """
        Categorize a transaction based on its description.
        
        Args:
            description: Transaction description
            amount: Transaction amount (used for some rules)
            
        Returns:
            Category ID for the transaction
        """
        description = description.lower()
        
        # Try to match transaction description against rules
        for category_id, rules in self.category_rules.items():
            for rule in rules:
                if rule in description:
                    return category_id
        
        # Default to "other" if no rules match
        return "other"
    
    def bulk_categorize(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize multiple transactions at once.
        
        Args:
            transactions: DataFrame with transactions
            
        Returns:
            DataFrame with added category column
        """
        if transactions.empty:
            return transactions
        
        # Make a copy to avoid modifying the original
        result = transactions.copy()
        
        # Ensure description column exists
        if 'description' not in result.columns:
            return result
        
        # Apply categorization function to each transaction
        result['category'] = result.apply(
            lambda row: self.categorize_transaction(
                row['description'], 
                row.get('amount', 0.0)
            ), 
            axis=1
        )
        
        return result
    
    def get_category_summary(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of categories used in transactions.
        
        Args:
            transactions: DataFrame with categorized transactions
            
        Returns:
            Dictionary with category summary statistics
        """
        if transactions.empty or 'category' not in transactions.columns:
            return {"error": "No categorized transactions available"}
        
        # Count transactions by category
        category_counts = transactions['category'].value_counts().to_dict()
        
        # Calculate total amount by category if amount column exists
        category_amounts = {}
        if 'amount' in transactions.columns:
            category_amounts = transactions.groupby('category')['amount'].sum().to_dict()
        
        # Combine counts and amounts
        summary = {}
        for category_id in set(list(category_counts.keys()) + list(category_amounts.keys())):
            if category_id in self.categories:
                summary[category_id] = {
                    "name": self.categories[category_id]["name"],
                    "description": self.categories[category_id]["description"],
                    "transaction_count": category_counts.get(category_id, 0),
                    "total_amount": category_amounts.get(category_id, 0.0)
                }
        
        return summary

    def _load_state(self) -> None:
        """Loads categories and rules from the storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.categories = data.get("categories", self._get_default_categories())
                self.category_rules = data.get("rules", self._get_default_rules())
                logger.info(f"Loaded category state from {self.storage_path}")
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.error(f"Error loading category state from {self.storage_path}: {e}. Using defaults.")
                self._set_defaults_and_save() # Initialize with defaults if file is corrupt/invalid
        else:
            logger.warning(f"Category state file not found at {self.storage_path}. Initializing with defaults.")
            self._set_defaults_and_save()

    # ADD helper for defaults
    def _set_defaults_and_save(self) -> None:
        """Sets default categories/rules and attempts to save them."""
        self.categories = self._get_default_categories()
        self.category_rules = self._get_default_rules()
        logger.info("Set default categories and rules.")
        self._save_state() 

    def _save_state(self) -> None:
        """Saves the current categories and rules to the storage file."""
        state_data = {
            "categories": self.categories,
            "rules": self.category_rules,
            "updated_at": datetime.now().isoformat()
        }
        try:
            # Ensure directory exists before trying to write the file
            storage_dir = os.path.dirname(self.storage_path)
            if storage_dir: # Only create if path includes a directory
                os.makedirs(storage_dir, exist_ok=True)

            with open(self.storage_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.debug(f"Saved category state to {self.storage_path}")
        except IOError as e:
            logger.error(f"Error saving category state to {self.storage_path}: {e}")
        except TypeError as e:
             logger.error(f"Error serializing category state (potential non-serializable data): {e}")


    # ADD default getters (moved from old init)
    def _get_default_categories(self) -> Dict:
         return {
            "housing": {"name": "Housing", "description": "Rent, mortgage, utilities, repairs"},
            # ... (rest of default categories) ...
             "income": {"name": "Income", "description": "Salary, gifts, refunds"},
             "other": {"name": "Other", "description": "Miscellaneous expenses"}
        }

    def _get_default_rules(self) -> Dict:
         return {
            "housing": ["rent", "mortgage", "electricity", "water", "gas bill", "internet", "cable"],
            # ... (rest of default rules) ...
             "income": ["salary", "paycheck", "dividend", "interest", "refund", "gift received"]
         }
    
    def suggest_category_improvements(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest improvements to categorization rules based on transaction patterns.
        
        Args:
            transactions: DataFrame with categorized transactions
            
        Returns:
            Dictionary with improvement suggestions
        """
        if transactions.empty or 'description' not in transactions.columns:
            return {"error": "No transaction data available"}
        
        # Find transactions without categories or categorized as "other"
        uncategorized = transactions[
            (transactions.get('category', '') == '') | 
            (transactions.get('category', '') == 'other')
        ]
        
        # Extract common words from uncategorized transactions
        common_words = {}
        if not uncategorized.empty:
            descriptions = ' '.join(uncategorized['description'].str.lower())
            # Remove common words that are unlikely to be useful for categorization
            stop_words = ['the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'by']
            words = re.findall(r'\b[a-z]{3,}\b', descriptions)
            words = [word for word in words if word not in stop_words]
            
            for word in words:
                if word not in common_words:
                    common_words[word] = 0
                common_words[word] += 1
        
        # Sort by frequency and get top 10
        common_words = dict(sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Check for patterns that might be miscategorized
        potential_new_rules = {}
        for category_id in self.categories:
            if category_id == 'other':
                continue
                
            # Find transactions with this category
            cat_transactions = transactions[transactions.get('category', '') == category_id]
            
            if not cat_transactions.empty:
                # Extract common words unique to this category
                category_descriptions = ' '.join(cat_transactions['description'].str.lower())
                category_words = set(re.findall(r'\b[a-z]{3,}\b', category_descriptions))
                
                # Check if any common words aren't in rules
                existing_rules = set(self.category_rules.get(category_id, []))
                potential_rules = category_words - existing_rules
                
                if potential_rules:
                    potential_new_rules[category_id] = list(potential_rules)
        
        return {
            "uncategorized_count": len(uncategorized),
            "common_uncategorized_words": common_words,
            "potential_new_rules": potential_new_rules
        }

