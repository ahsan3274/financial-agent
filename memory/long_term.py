"""
Long-term memory module for the financial agent.
Provides persistent storage for expense data and user preferences.
"""
import os
import json
import pandas as pd
import logging 
import time 
from datetime import datetime
from typing import Dict, List, Any, Optional, Union


logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    Long-term memory for persistent storage of financial data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize long-term memory.
        
        Args:
            data_dir: Directory for storing persistent data
        """
        self.data_dir = data_dir
        self.expenses_file = os.path.join(data_dir, "expenses.csv")
        self.categories_file = os.path.join(data_dir, "categories.json")
        self.preferences_file = os.path.join(data_dir, "preferences.json")
        self.goals_file = os.path.join(data_dir, "goals.json")
        
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
       
        self.expenses = self._load_expenses()
        self.categories = self._load_categories()
        self.preferences = self._load_preferences()
        self.goals = self._load_goals()
        logger.info(f"LongTermMemory initialized. Data directory: {self.data_dir}")
        self.general_log_file = os.path.join(data_dir, "agent_log.jsonl")

    def _load_expenses(self) -> pd.DataFrame:
        """
        Load expenses from CSV or create a new DataFrame.

        Returns:
            DataFrame with expenses
        """
        default_cols = {
            "id": [], "date": [], "amount": [], "description": [],
            "category": [], "subcategory": [], "payment_method": [], "notes": []
        }
        if os.path.exists(self.expenses_file):
            try:
                # Specify dtype for id to avoid mixed type warnings if possible
                df = pd.read_csv(self.expenses_file, parse_dates=["date"], dtype={'id': 'Int64'}) # Example Int64
                logger.info(f"Loaded {len(df)} expenses from {self.expenses_file}")
                # Ensure all default columns exist, add if missing
                for col in default_cols:
                    if col not in df.columns:
                        df[col] = pd.NA # Or appropriate default like '' for string, 0 for numeric
                return df
            except Exception as e:
                logger.error(f"Error loading expenses from {self.expenses_file}: {e}. Creating empty DataFrame.")
                return pd.DataFrame(default_cols)
        else:
            logger.warning(f"Expenses file not found: {self.expenses_file}. Creating empty DataFrame.")
            return pd.DataFrame(default_cols)

    def _load_categories(self) -> Dict[str, List[str]]:
        """
        Load categories from JSON or create default categories.

        Returns:
            Dictionary of categories and subcategories
        """
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, "r") as f:
                    categories_data = json.load(f)
                logger.info(f"Loaded categories from {self.categories_file}")
                return categories_data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading categories from {self.categories_file}: {e}. Using default categories.")
                return self._get_default_categories()
        else:
            logger.warning(f"Categories file not found: {self.categories_file}. Using default categories.")
            
            return self._get_default_categories()

    def _load_preferences(self) -> Dict[str, Any]:
        """
        Load user preferences from JSON or create defaults.

        Returns:
            Dictionary of user preferences
        """
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, "r") as f:
                    prefs = json.load(f)
                logger.info(f"Loaded preferences from {self.preferences_file}")
                
                return prefs
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading preferences from {self.preferences_file}: {e}. Using defaults.")
                return self._get_default_preferences()
        else:
            logger.warning(f"Preferences file not found: {self.preferences_file}. Using defaults.")
            # Optionally save defaults if file doesn't exist
            # default_prefs = self._get_default_preferences()
            # self.save_preferences() # Need save_preferences method if doing this
            return self._get_default_preferences()

    def _load_goals(self) -> List[Dict[str, Any]]:
        """
        Load financial goals from JSON or create an empty list.

        Returns:
            List of financial goals
        """
        if os.path.exists(self.goals_file):
            try:
                with open(self.goals_file, "r") as f:
                    goals_data = json.load(f)
                logger.info(f"Loaded {len(goals_data)} goals from {self.goals_file}")
                return goals_data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading goals from {self.goals_file}: {e}. Returning empty list.")
                return []
        else:
            logger.warning(f"Goals file not found: {self.goals_file}. Returning empty list.")
            return []

    def store(self, record: Dict[str, Any]) -> None:
        """
        Store a generic record, potentially routing or logging it.
        For now, appends to a general JSONL log file.
        """
        try:
            # Example: Append to a JSON Lines file
            with open(self.general_log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            
            if record.get("type") == "task_execution":
                 # Maybe store summary in a dedicated task log?
                 pass # Placeholder for specific logic
        except Exception as e:
            print(f"Error storing record: {e}") # Use logger preferably

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Basic search functionality (Placeholder).
        Searches descriptions in expenses and goals for the query term.
        A more robust implementation would use embeddings or a search index.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of relevant records (dictionaries).
        """
        results = []
        query_lower = query.lower()

        # Search Expenses (example: search description)
        if not self.expenses.empty and 'description' in self.expenses.columns:
             # Ensure 'description' is string type and handle NaN
            expense_matches = self.expenses[
                self.expenses['description'].astype(str).str.lower().str.contains(query_lower, na=False)
            ]
            results.extend(expense_matches.head(top_k).to_dict(orient='records'))

        
        if self.goals:
            goal_matches = [
                goal for goal in self.goals
                if query_lower in goal.get('name', '').lower() or query_lower in goal.get('notes', '').lower()
            ]
            results.extend(goal_matches[:max(0, top_k - len(results))]) # Avoid adding more than top_k

        
        print(f"LTM Search for '{query}': Found {len(results)} potential matches (returning up to {top_k}).") # Use logger
        return results[:top_k]

        def get_items_by_type(self, item_type: str) -> List[Dict[str, Any]]:
         """
         Retrieves items from memory based on their type.
         (Basic implementation - searches general log or specific stores).
         """
         items = []
         # Example: Search the general log file if it exists
         if os.path.exists(self.general_log_file):
             try:
                 with open(self.general_log_file, "r") as f:
                     for line in f:
                         try:
                             record = json.loads(line)
                             if record.get("type") == item_type:
                                 items.append(record)
                         except json.JSONDecodeError:
                             continue # Skip corrupted lines
             except Exception as e:
                 print(f"Error reading general log for get_items_by_type: {e}") 

         

         print(f"LTM get_items_by_type '{item_type}': Found {len(items)} items.") 
         return items

    def size(self):
        return len(self.expenses) + len(self.goals)

    def _get_default_categories(self) -> Dict[str, List[str]]:
        """
        Get default expense categories.
        
        Returns:
            Dictionary of default categories and subcategories
        """
        return {
            "Housing": ["Rent", "Mortgage", "Property Tax", "Maintenance", "Utilities", "Insurance"],
            "Transportation": ["Car Payment", "Gas", "Insurance", "Maintenance", "Public Transit"],
            "Food": ["Groceries", "Restaurants", "Takeout", "Coffee Shops"],
            "Entertainment": ["Movies", "Streaming Services", "Events", "Hobbies"],
            "Shopping": ["Clothing", "Electronics", "Home Goods", "Personal Care"],
            "Health": ["Insurance", "Doctor Visits", "Medications", "Gym", "Wellness"],
            "Education": ["Tuition", "Books", "Courses", "Supplies"],
            "Personal": ["Subscriptions", "Gifts", "Donations", "Miscellaneous"],
            "Debt": ["Credit Card", "Student Loans", "Personal Loans"],
            "Income": ["Salary", "Freelance", "Investments", "Other"]
        }
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """
        Get default user preferences.
        
        Returns:
            Dictionary of default preferences
        """
        return {
            "currency": "USD",
            "date_format": "%Y-%m-%d",
            "monthly_budget": {},
            "default_payment_method": "",
            "notifications_enabled": True,
            "first_day_of_week": "Monday",
            "theme": "light"
        }
    
    def _load_expenses(self) -> pd.DataFrame:
        """
        Load expenses from CSV or create a new DataFrame.
        
        Returns:
            DataFrame with expenses
        """
        if os.path.exists(self.expenses_file):
            try:
                return pd.read_csv(self.expenses_file, parse_dates=["date"])
            except Exception as e:
                print(f"Error loading expenses: {e}")
                return pd.DataFrame({
                    "id": [],
                    "date": [],
                    "amount": [],
                    "description": [],
                    "category": [],
                    "subcategory": [],
                    "payment_method": [],
                    "notes": []
                })
        else:
            return pd.DataFrame({
                "id": [],
                "date": [],
                "amount": [],
                "description": [],
                "category": [],
                "subcategory": [],
                "payment_method": [],
                "notes": []
            })
    
    def _load_categories(self) -> Dict[str, List[str]]:
        """
        Load categories from JSON or create default categories.
        
        Returns:
            Dictionary of categories and subcategories
        """
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading categories: {e}")
                return self._get_default_categories()
        else:
            return self._get_default_categories()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """
        Load user preferences from JSON or create defaults.
        
        Returns:
            Dictionary of user preferences
        """
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return self._get_default_preferences()
        else:
            return self._get_default_preferences()
    
    def _load_goals(self) -> List[Dict[str, Any]]:
        """
        Load financial goals from JSON or create an empty list.
        
        Returns:
            List of financial goals
        """
        if os.path.exists(self.goals_file):
            try:
                with open(self.goals_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading goals: {e}")
                return []
        else:
            return []
    
    
    
    def save_all(self) -> None:
        """Save all data to files."""
        self.save_expenses()
        #self.save_categories()
        self.save_preferences()
        self.save_goals()
        logger.info("Finished saving LTM components.")
    
    def save_expenses(self) -> None:
        """Save expenses to CSV file."""
        try:
            self.expenses.to_csv(self.expenses_file, index=False)
        except Exception as e:
            print(f"Error saving expenses: {e}")
    
        
    def save_preferences(self) -> None:
        """Save preferences to JSON file."""
        try:
            with open(self.preferences_file, "w") as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def save_goals(self) -> None:
        """Save goals to JSON file."""
        try:
            with open(self.goals_file, "w") as f:
                json.dump(self.goals, f, indent=2)
        except Exception as e:
            print(f"Error saving goals: {e}")
    
    def add_expense(self, expense_data: Dict[str, Any]) -> int:
        """
        Add a new expense record. Handles date parsing robustly.

        Args:
            expense_data: Expense data dictionary

        Returns:
            ID of the new expense
        """
        # Generate ID if not provided - use max existing ID + 1 or timestamp for uniqueness
        if "id" not in expense_data or pd.isna(expense_data.get("id")):
             if not self.expenses.empty and 'id' in self.expenses.columns and self.expenses['id'].dtype in ['int64', 'float64']:
                  # Be careful if IDs are not purely numeric or sequential
                  try:
                      next_id = int(self.expenses['id'].max()) + 1
                  except ValueError:
                      # Fallback if max ID isn't easily determined (e.g., mixed types, strings)
                      next_id = int(time.time() * 1000) # Use timestamp as a fallback ID
             else:
                  next_id = 1 # First expense
             expense_data["id"] = next_id
        else:
            
            pass 

        
        date_val = expense_data.get("date")
        final_date = None

        if isinstance(date_val, str):
            try:
                
                pref_format = self.preferences.get("date_format")
                if pref_format:
                     final_date = pd.to_datetime(date_val, format=pref_format)
                else:
                    
                    final_date = pd.to_datetime(date_val, format="mixed", dayfirst=False) 
            except ValueError:
                try:
                    
                     final_date = pd.to_datetime(date_val, format="mixed", dayfirst=False)
                except ValueError:
                     
                     logger.warning(f"Could not parse date string '{date_val}' for expense ID {expense_data['id']}. Using current time.")
                     final_date = pd.Timestamp.now()
        elif isinstance(date_val, (datetime, pd.Timestamp)):
             
             final_date = pd.Timestamp(date_val)
        else:
             
             logger.warning(f"Invalid or missing date '{date_val}' for expense ID {expense_data['id']}. Using current time.")
             final_date = pd.Timestamp.now()

        expense_data["date"] = final_date
        

        
        new_expense_df = pd.DataFrame([expense_data])

        
        cols = self.expenses.columns.union(new_expense_df.columns)
        self.expenses = pd.concat(
             [self.expenses.reindex(columns=cols), new_expense_df.reindex(columns=cols)],
             ignore_index=True
         )
        
        self.save_expenses()

        return expense_data["id"]
    
    def update_expense(self, expense_id: int, expense_data: Dict[str, Any]) -> bool:
        """
        Update an existing expense.
        
        Args:
            expense_id: ID of the expense to update
            expense_data: New expense data
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Check if expense exists
        if expense_id not in self.expenses["id"].values:
            return False
        
        # Update expense
        for key, value in expense_data.items():
            if key == "date" and isinstance(value, str):
                try:
                    value = datetime.strptime(value, self.preferences["date_format"])
                except ValueError:
                    try:
                        value = datetime.fromisoformat(value)
                    except ValueError:
                        continue
            
            self.expenses.loc[self.expenses["id"] == expense_id, key] = value
        
        # Save to file
        self.save_expenses()
        
        return True
    
    def delete_expense(self, expense_id: int) -> bool:
        """
        Delete an expense.
        
        Args:
            expense_id: ID of the expense to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if expense_id not in self.expenses["id"].values:
            return False
        
        self.expenses = self.expenses[self.expenses["id"] != expense_id]
        self.save_expenses()
        
        return True
    
    def get_expenses(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    category: Optional[str] = None,
                    min_amount: Optional[float] = None,
                    max_amount: Optional[float] = None,
                    search_term: Optional[str] = None) -> pd.DataFrame:
        """
        Get expenses with optional filters.
        
        Args:
            start_date: Filter expenses after this date
            end_date: Filter expenses before this date
            category: Filter by category
            min_amount: Filter by minimum amount
            max_amount: Filter by maximum amount
            search_term: Search in description
            
        Returns:
            DataFrame with filtered expenses
        """
        filtered_expenses = self.expenses.copy()
        
        # Apply filters
        if start_date:
            filtered_expenses = filtered_expenses[filtered_expenses["date"] >= start_date]
        
        if end_date:
            filtered_expenses = filtered_expenses[filtered_expenses["date"] <= end_date]
        
        if category:
            filtered_expenses = filtered_expenses[filtered_expenses["category"] == category]
        
        if min_amount is not None:
            filtered_expenses = filtered_expenses[filtered_expenses["amount"] >= min_amount]
        
        if max_amount is not None:
            filtered_expenses = filtered_expenses[filtered_expenses["amount"] <= max_amount]
        
        if search_term:
            filtered_expenses = filtered_expenses[
                filtered_expenses["description"].str.contains(search_term, case=False, na=False)
            ]
        
        return filtered_expenses
    
    def get_expense_by_id(self, expense_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an expense by ID.
        
        Args:
            expense_id: The expense ID
            
        Returns:
            Expense dictionary or None if not found
        """
        expense = self.expenses[self.expenses["id"] == expense_id]
        if len(expense) == 0:
            return None
        
        return expense.iloc[0].to_dict()
    
    def add_category(self, category: str, subcategories: Optional[List[str]] = None) -> bool:
        """
        Add a new expense category.
        
        Args:
            category: Category name
            subcategories: Optional list of subcategories
            
        Returns:
            True if added successfully, False if category already exists
        """
        if category in self.categories:
            return False
        
        self.categories[category] = subcategories or []
        self.save_categories()
        
        return True
    
    def update_category(self, category: str, subcategories: List[str]) -> bool:
        """
        Update subcategories for a category.
        
        Args:
            category: Category name
            subcategories: New list of subcategories
            
        Returns:
            True if updated successfully, False if category doesn't exist
        """
        if category not in self.categories:
            return False
        
        self.categories[category] = subcategories
        self.save_categories()
        
        return True
    
    def delete_category(self, category: str) -> bool:
        """
        Delete a category.
        
        Args:
            category: Category name to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        if category not in self.categories:
            return False
        
        del self.categories[category]
        self.save_categories()
        
        return True
    
    def get_categories(self) -> Dict[str, List[str]]:
        """
        Get all categories and subcategories.
        
        Returns:
            Dictionary of categories and subcategories
        """
        return self.categories
    
    def update_preference(self, key: str, value: Any) -> None:
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: New preference value
        """
        self.preferences[key] = value
        self.save_preferences()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Preference value or default
        """
        return self.preferences.get(key, default)
    
    def add_goal(self, goal_data: Dict[str, Any]) -> int:
        """
        Add a financial goal.
        
        Args:
            goal_data: Goal data dictionary
            
        Returns:
            ID of the new goal
        """
        # Generate ID if not provided
        if "id" not in goal_data:
            goal_data["id"] = len(self.goals) + 1
        
        if "created_date" not in goal_data:
            goal_data["created_date"] = datetime.now().isoformat()
        
        self.goals.append(goal_data)
        self.save_goals()
        
        return goal_data["id"]
    
    def update_goal(self, goal_id: int, goal_data: Dict[str, Any]) -> bool:
        """
        Update a financial goal.
        
        Args:
            goal_id: ID of the goal to update
            goal_data: New goal data
            
        Returns:
            True if updated successfully, False otherwise
        """
        for i, goal in enumerate(self.goals):
            if goal.get("id") == goal_id:
                self.goals[i].update(goal_data)
                self.save_goals()
                return True
        
        return False
    
    def delete_goal(self, goal_id: int) -> bool:
        """
        Delete a financial goal.
        
        Args:
            goal_id: ID of the goal to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        for i, goal in enumerate(self.goals):
            if goal.get("id") == goal_id:
                del self.goals[i]
                self.save_goals()
                return True
        
        return False
    
    def get_goals(self) -> List[Dict[str, Any]]:
        """
        Get all financial goals.
        
        Returns:
            List of financial goals
        """
        return self.goals
