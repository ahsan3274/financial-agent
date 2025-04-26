"""
/tools/goal_tracker.py
Tool for tracking financial goals and monitoring progress towards them.
Allows users to set, update, and analyze progress on various financial goals.
"""

import datetime
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt


class GoalTracker:
    """Manages financial goals and tracks progress towards them."""
    
    def __init__(self, storage_path: str = "data/goals.csv"):
        """
        Initialize the GoalTracker.
        
        Args:
            storage_path: Path to store goal data
        """
        self.storage_path = storage_path
        self.goals = self._load_goals()
        
    def _load_goals(self) -> pd.DataFrame:
        """Load goals from storage or create empty DataFrame if none exists."""
        try:
            return pd.read_csv(self.storage_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=[
                'goal_id', 'name', 'category', 'target_amount', 
                'current_amount', 'start_date', 'target_date', 'priority',
                'status', 'milestones', 'notes'
            ])
    
    def _save_goals(self) -> None:
        """Save goals to storage."""
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.goals.to_csv(self.storage_path, index=False)
    
    def create_goal(self, 
                   name: str,
                   category: str, 
                   target_amount: float,
                   target_date: Union[str, datetime.date],
                   priority: str = "medium",
                   initial_amount: float = 0.0,
                   notes: str = "") -> str:
        """
        Create a new financial goal.
        
        Args:
            name: Name of the goal
            category: Category of goal (e.g., 'savings', 'debt_reduction')
            target_amount: Amount to reach
            target_date: Date to reach the goal by
            priority: Priority level ('high', 'medium', 'low')
            initial_amount: Starting amount
            notes: Additional notes
            
        Returns:
            goal_id: ID of the created goal
        """
        
        if target_amount <= 0:
            raise ValueError("Target amount must be positive")
        
        if isinstance(target_date, str):
            target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
            
        today = datetime.date.today()
        if target_date < today:
            raise ValueError("Target date must be in the future")
            
        if priority.lower() not in ['high', 'medium', 'low']:
            raise ValueError("Priority must be 'high', 'medium', or 'low'")
        
        if len(self.goals) == 0:
            goal_id = "GOAL-1"
        else:
            last_id = int(self.goals['goal_id'].iloc[-1].split('-')[1])
            goal_id = f"GOAL-{last_id + 1}"
        
        new_goal = pd.DataFrame([{
            'goal_id': goal_id,
            'name': name,
            'category': category,
            'target_amount': target_amount,
            'current_amount': initial_amount,
            'start_date': today.strftime("%Y-%m-%d"),
            'target_date': target_date.strftime("%Y-%m-%d"),
            'priority': priority.lower(),
            'status': 'active',
            'milestones': '',  
            'notes': notes
        }])
        
        self.goals = pd.concat([self.goals, new_goal], ignore_index=True)
        self._save_goals()
        
        return goal_id
    
    def update_progress(self, goal_id: str, new_amount: float) -> Dict:
        """
        Update the current amount for a goal.
        
        Args:
            goal_id: ID of the goal to update
            new_amount: The new total amount (not incremental)
            
        Returns:
            A dictionary with updated goal information and progress
        """
        if goal_id not in self.goals['goal_id'].values:
            raise ValueError(f"Goal with ID {goal_id} not found")
        
        
        goal_idx = self.goals[self.goals['goal_id'] == goal_id].index[0]
        
        self.goals.at[goal_idx, 'current_amount'] = new_amount
        
        if new_amount >= self.goals.at[goal_idx, 'target_amount']:
            self.goals.at[goal_idx, 'status'] = 'completed'
            
        self._save_goals()
        
        goal_data = self.goals.iloc[goal_idx].to_dict()
        progress_pct = min(100, (new_amount / goal_data['target_amount']) * 100)
        
        return {
            'goal_id': goal_id,
            'name': goal_data['name'],
            'current_amount': new_amount,
            'target_amount': goal_data['target_amount'],
            'progress_percentage': progress_pct,
            'status': goal_data['status']
        }
    
    def get_goal(self, goal_id: str) -> Dict:
        """
        Get detailed information about a specific goal.
        
        Args:
            goal_id: ID of the goal to retrieve
            
        Returns:
            Dictionary with goal details
        """
        if goal_id not in self.goals['goal_id'].values:
            raise ValueError(f"Goal with ID {goal_id} not found")
        
        goal_data = self.goals[self.goals['goal_id'] == goal_id].iloc[0].to_dict()
        
        progress_pct = min(100, (goal_data['current_amount'] / goal_data['target_amount']) * 100)
        goal_data['progress_percentage'] = progress_pct
        
        goal_data['remaining_amount'] = max(0, goal_data['target_amount'] - goal_data['current_amount'])
        
        target_date = datetime.datetime.strptime(goal_data['target_date'], "%Y-%m-%d").date()
        today = datetime.date.today()
        goal_data['days_remaining'] = (target_date - today).days if target_date > today else 0
        
        if goal_data['milestones']:
            goal_data['milestones'] = goal_data['milestones'].split('|')
        else:
            goal_data['milestones'] = []
            
        return goal_data
    
    def list_goals(self, 
                  status: Optional[str] = None, 
                  category: Optional[str] = None,
                  priority: Optional[str] = None) -> List[Dict]:
        """
        List goals with optional filtering.
        
        Args:
            status: Filter by status ('active', 'completed', 'abandoned')
            category: Filter by category
            priority: Filter by priority ('high', 'medium', 'low')
            
        Returns:
            List of goal dictionaries with basic info
        """
        filtered_goals = self.goals.copy()
        
        if status:
            filtered_goals = filtered_goals[filtered_goals['status'] == status]
        if category:
            filtered_goals = filtered_goals[filtered_goals['category'] == category]
        if priority:
            filtered_goals = filtered_goals[filtered_goals['priority'] == priority]
            
        result = []
        for _, goal in filtered_goals.iterrows():
            progress_pct = min(100, (goal['current_amount'] / goal['target_amount']) * 100)
            
            result.append({
                'goal_id': goal['goal_id'],
                'name': goal['name'],
                'category': goal['category'],
                'current_amount': goal['current_amount'],
                'target_amount': goal['target_amount'],
                'progress_percentage': progress_pct,
                'target_date': goal['target_date'],
                'priority': goal['priority'],
                'status': goal['status']
            })
            
        return result
    
    def add_milestone(self, goal_id: str, milestone: str) -> List[str]:
        """
        Add a milestone to a goal.
        
        Args:
            goal_id: ID of the goal
            milestone: Milestone description
            
        Returns:
            Updated list of milestones
        """
        if goal_id not in self.goals['goal_id'].values:
            raise ValueError(f"Goal with ID {goal_id} not found")
        
        goal_idx = self.goals[self.goals['goal_id'] == goal_id].index[0]
        
        current_milestones = self.goals.at[goal_idx, 'milestones']
        
        if current_milestones:
            updated_milestones = f"{current_milestones}|{milestone}"
        else:
            updated_milestones = milestone
            
        self.goals.at[goal_idx, 'milestones'] = updated_milestones
        self._save_goals()
        
        return updated_milestones.split('|')
    
    def update_goal_status(self, goal_id: str, status: str) -> Dict:
        """
        Update the status of a goal.
        
        Args:
            goal_id: ID of the goal
            status: New status ('active', 'completed', 'abandoned', 'paused')
            
        Returns:
            Updated goal information
        """
        valid_statuses = ['active', 'completed', 'abandoned', 'paused']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
            
        if goal_id not in self.goals['goal_id'].values:
            raise ValueError(f"Goal with ID {goal_id} not found")
        
        goal_idx = self.goals[self.goals['goal_id'] == goal_id].index[0]
        self.goals.at[goal_idx, 'status'] = status
        self._save_goals()
        
        return self.get_goal(goal_id)
    
    def update_goal_details(self, 
                           goal_id: str, 
                           **kwargs) -> Dict:
        """
        Update various attributes of a goal.
        
        Args:
            goal_id: ID of the goal to update
            **kwargs: Fields to update (name, target_amount, target_date, priority, notes)
            
        Returns:
            Updated goal information
        """
        if goal_id not in self.goals['goal_id'].values:
            raise ValueError(f"Goal with ID {goal_id} not found")
            
        allowed_fields = ['name', 'target_amount', 'target_date', 'priority', 'notes', 'category']
        
        goal_idx = self.goals[self.goals['goal_id'] == goal_id].index[0]
        
        for field, value in kwargs.items():
            if field not in allowed_fields:
                continue
                
            if field == 'target_amount' and value <= 0:
                raise ValueError("Target amount must be positive")
                
            if field == 'priority' and value.lower() not in ['high', 'medium', 'low']:
                raise ValueError("Priority must be 'high', 'medium', or 'low'")
                
            if field == 'target_date':
                if isinstance(value, str):
                    try:
                        datetime.datetime.strptime(value, "%Y-%m-%d")
                    except ValueError:
                        raise ValueError("Target date must be in YYYY-MM-DD format")
                elif isinstance(value, datetime.date):
                    value = value.strftime("%Y-%m-%d")
                    
            self.goals.at[goal_idx, field] = value
            
        self._save_goals()
        
        return self.get_goal(goal_id)
    
    def calculate_projection(self, goal_id: str) -> Dict:
        """
        Calculate projections for meeting a goal based on current progress.
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            Dictionary with projection details
        """
        goal_data = self.get_goal(goal_id)
        
        start_date = datetime.datetime.strptime(goal_data['start_date'], "%Y-%m-%d").date()
        target_date = datetime.datetime.strptime(goal_data['target_date'], "%Y-%m-%d").date() 
        today = datetime.date.today()
        
        months_remaining = (target_date.year - today.year) * 12 + target_date.month - today.month
        if months_remaining <= 0:
            months_remaining = 1  # Avoid division by zero
            
        amount_remaining = goal_data['target_amount'] - goal_data['current_amount']
        
        required_monthly = amount_remaining / months_remaining
        
        months_elapsed = (today.year - start_date.year) * 12 + today.month - start_date.month
        if months_elapsed < 1:
            months_elapsed = 1
            
        average_monthly = goal_data['current_amount'] / months_elapsed
        
        if average_monthly <= 0:
            projected_months = float('inf')
            projected_completion = "N/A (insufficient progress)"
        else:
            projected_months = amount_remaining / average_monthly
            projected_date = today + datetime.timedelta(days=int(projected_months * 30.4))
            projected_completion = projected_date.strftime("%Y-%m-%d")
            
        return {
            'goal_id': goal_id,
            'name': goal_data['name'],
            'required_monthly_contribution': required_monthly,
            'current_average_monthly': average_monthly,
            'on_track': average_monthly >= required_monthly,
            'projected_completion_date': projected_completion,
            'months_remaining': months_remaining,
            'amount_remaining': amount_remaining
        }
        
    def delete_goal(self, goal_id: str) -> bool:
        """
        Delete a goal.
        
        Args:
            goal_id: ID of the goal to delete
            
        Returns:
            True if successful
        """
        if goal_id not in self.goals['goal_id'].values:
            raise ValueError(f"Goal with ID {goal_id} not found")
            
        self.goals = self.goals[self.goals['goal_id'] != goal_id]
        self._save_goals()
        
        return True
    
    def generate_progress_chart(self, goal_id: str, save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate a chart showing progress towards a goal.
        
        Args:
            goal_id: ID of the goal
            save_path: Path to save the chart (if provided)
            
        Returns:
            Figure and Axes objects
        """
        goal_data = self.get_goal(goal_id)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = ['Progress', 'Remaining']
        sizes = [goal_data['current_amount'], goal_data['remaining_amount']]
        colors = ['#4CAF50', '#E0E0E0']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.title(f"Progress for: {goal_data['name']}")
        
        plt.figtext(0.5, 0.01, 
                   f"Target: ${goal_data['target_amount']:.2f} | "
                   f"Current: ${goal_data['current_amount']:.2f} | "
                   f"Days remaining: {goal_data['days_remaining']}",
                   ha='center', fontsize=10)
        
        if save_path:
            plt.savefig(save_path)
            
        return fig, ax
    
    def get_goal_summary(self) -> Dict:
        """
        Get a summary of all goals and their progress.
        
        Returns:
            Dictionary with summary statistics
        """
        if len(self.goals) == 0:
            return {
                'total_goals': 0,
                'active_goals': 0,
                'completed_goals': 0,
                'total_target_amount': 0,
                'total_current_amount': 0,
                'overall_progress': 0
            }
            
        active_goals = len(self.goals[self.goals['status'] == 'active'])
        completed_goals = len(self.goals[self.goals['status'] == 'completed'])
        
        total_target = self.goals['target_amount'].sum()
        total_current = self.goals['current_amount'].sum()
        
        overall_progress = (total_current / total_target * 100) if total_target > 0 else 0
        
        
        on_track_count = 0
        for goal_id in self.goals['goal_id']:
            try:
                projection = self.calculate_projection(goal_id)
                if projection['on_track']:
                    on_track_count += 1
            except:
               
                pass
                
        return {
            'total_goals': len(self.goals),
            'active_goals': active_goals,
            'completed_goals': completed_goals,
            'abandoned_goals': len(self.goals) - active_goals - completed_goals,
            'total_target_amount': total_target,
            'total_current_amount': total_current,
            'overall_progress': overall_progress,
            'on_track_goals': on_track_count
        }
