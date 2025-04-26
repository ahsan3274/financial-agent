"""
/tools/budget_advisor.py
Budget advisor module for creating and managing budgets and providing recommendations.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging # Use logging
from typing import Dict, List, Optional, Any, Tuple

try:
    from memory.long_term import LongTermMemory
except ImportError:
    LongTermMemory = object 

logger = logging.getLogger(__name__)

class BudgetAdvisor:
    """
    Creates and manages budgets, provides spending recommendations based on LTM data.
    """
    def __init__(self, budget_file: str = "data/budget.json"):
        """
        Initialize the budget advisor.

        Args:
            budget_file: Path to JSON file storing budget data
        """
        self.budget_file = budget_file
        self.budgets = {}
        self.income = 0.0
        self.load_budget()
        logger.info(f"BudgetAdvisor initialized. Loaded budget from {self.budget_file}")

    def load_budget(self) -> None:
        """Load budget data from the budget file."""
        if os.path.exists(self.budget_file):
            try:
                with open(self.budget_file, 'r') as file:
                    data = json.load(file)
                    self.budgets = data.get('categories', {})
                    self.income = data.get('income', 0.0)
                    logger.info(f"Budget loaded: {len(self.budgets)} categories, income {self.income}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading budget file {self.budget_file}: {e}")
                self.budgets = {}
                self.income = 0.0
        else:
            logger.warning(f"Budget file {self.budget_file} not found. Initializing empty budget.")
            self.budgets = {}
            self.income = 0.0

    def save_budget(self) -> None:
        """Save current budget data to the budget file."""
        data = {
            "categories": self.budgets,
            "income": self.income,
            "updated_at": datetime.now().isoformat()
        }
        try:
            os.makedirs(os.path.dirname(self.budget_file), exist_ok=True)
            with open(self.budget_file, 'w') as file:
                json.dump(data, file, indent=2)
            logger.debug(f"Budget saved to {self.budget_file}")
        except IOError as e:
            logger.error(f"Error saving budget file {self.budget_file}: {e}")

    def set_income(self, amount: float) -> Dict[str, Any]:
        """Set monthly income."""
        if amount < 0:
            return {"error": "Income cannot be negative"}
        self.income = amount
        self.save_budget()
        logger.info(f"Monthly income set to {amount}")
        return {"success": f"Monthly income set to {amount}"}

    def get_income(self) -> float:
        """Get current monthly income."""
        return self.income

    def set_category_budget(self, category: str, amount: float) -> Dict[str, Any]:
        """Set budget for a specific category."""
        if amount < 0:
            return {"error": "Budget amount cannot be negative"}
        category_key = category.lower().strip() # Normalize category key
        self.budgets[category_key] = amount
        self.save_budget()
        logger.info(f"Budget for category '{category_key}' set to {amount}")
        return {"success": f"Budget for '{category}' set to {amount}"} # Return original name

    def get_category_budget(self, category: str) -> Dict[str, Any]:
        """Get budget for a specific category."""
        category_key = category.lower().strip()
        if category_key not in self.budgets:
            return {"error": f"No budget set for category '{category}'"}
        return {"category": category, "budget": self.budgets[category_key]}

    def get_all_budgets(self) -> Dict[str, float]:
        """Get all category budgets."""
        return self.budgets.copy()

    def calculate_budget_status(self, expenses: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate budget status for the current month based on provided expenses.

        Args:
            expenses: DataFrame with expense records for the relevant period (e.g., current month).

        Returns:
            Dictionary with budget status for each category and summary.
        """
        if not isinstance(expenses, pd.DataFrame):
             return {"error": "Invalid expense data provided (must be DataFrame)"}
        if expenses.empty:
             logger.warning("Calculating budget status with empty expenses DataFrame.")
             # Return status assuming zero spending if df is empty but valid
             # return {"message": "No expense data provided for calculation period."}


        if 'category' not in expenses.columns or 'amount' not in expenses.columns:
            return {"error": "Expense data must include 'category' and 'amount' columns"}

        expenses['amount'] = pd.to_numeric(expenses['amount'], errors='coerce').fillna(0)

        current_month = datetime.now().strftime('%Y-%m')
        if 'date' in expenses.columns:
            if not pd.api.types.is_datetime64_any_dtype(expenses['date']):
                 try:
                     expenses['date'] = pd.to_datetime(expenses['date'])
                 except Exception:
                     logger.error("Failed to convert 'date' column to datetime in calculate_budget_status.")
                     

            if pd.api.types.is_datetime64_any_dtype(expenses['date']):
                 monthly_expenses = expenses[expenses['date'].dt.strftime('%Y-%m') == current_month].copy()
                 logger.debug(f"Filtered {len(monthly_expenses)} expenses for current month {current_month}")
            else:
                 monthly_expenses = expenses # Use all if no date or conversion failed
                 logger.warning("Using all provided expenses for budget status as date filtering failed/unavailable.")
        else:
             monthly_expenses = expenses # Use all if no date column
             logger.warning("Using all provided expenses for budget status as 'date' column is missing.")

        monthly_expenses['category_key'] = monthly_expenses['category'].str.lower().str.strip()
        category_spending = monthly_expenses.groupby('category_key')['amount'].sum().to_dict()
        logger.debug(f"Spending this month by category key: {category_spending}")

        budget_status = {}
        total_budget = 0.0
        total_spent_in_budgeted = 0.0

        for category_key, budget in self.budgets.items():
            spent = category_spending.get(category_key, 0.0)
            actual_spent = abs(spent) if spent < 0 else 0 # Consider only negative amounts as spending
            
            remaining = budget - actual_spent
            percentage_used = (actual_spent / budget * 100) if budget > 0 else 0

            status = "on track"
            if percentage_used >= 100:
                status = "exceeded"
            elif percentage_used >= 90:
                status = "warning"

            budget_status[category_key] = {
                "budget": budget,
                "spent": actual_spent,
                "remaining": remaining,
                "percentage_used": round(percentage_used, 2),
                "status": status
            }
            total_budget += budget
            total_spent_in_budgeted += actual_spent

        total_remaining = total_budget - total_spent_in_budgeted
        total_percentage = (total_spent_in_budgeted / total_budget * 100) if total_budget > 0 else 0

        overall_status = "on track"
        if total_percentage >= 100:
            overall_status = "exceeded"
        elif total_percentage >= 90:
            overall_status = "warning"

        result = {
            "categories": budget_status,
            "summary": {
                "total_budget": total_budget,
                "total_spent_in_budgeted_categories": total_spent_in_budgeted,
                "total_remaining_in_budgeted_categories": total_remaining,
                "percentage_used": round(total_percentage, 2),
                "status": overall_status
            },
            "month": current_month
        }
        logger.debug(f"Calculated budget status: {result['summary']}")
        return result

    def generate_spending_recommendations(self, ltm: LongTermMemory) -> Dict[str, Any]:
        """
        Generate spending recommendations based on budget and current month's expense history from LTM.

        Args:
            ltm: The LongTermMemory instance to fetch expenses from.

        Returns:
            Dictionary with spending recommendations.
        """
        logger.info("Generating spending recommendations...")
        # Fetch current month's expenses from LTM
        today = datetime.now()
        current_month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        try:
            expenses = ltm.get_expenses(start_date=current_month_start, end_date=today)
            logger.info(f"Fetched {len(expenses)} expenses from LTM for the current month.")
        except Exception as e:
             logger.error(f"Failed to fetch expenses from LTM: {e}", exc_info=True)
             return {"error": "Failed to retrieve expense data from memory."}


        budget_status_result = self.calculate_budget_status(expenses)

        if "error" in budget_status_result:
            logger.error(f"Error calculating budget status: {budget_status_result['error']}")
            return budget_status_result

        daily_limits = {}
        alerts = []
        recommendations = []
        remaining_days = 0

        try:
            days_in_month = (today.replace(month=today.month % 12 + 1, day=1) - timedelta(days=1)).day
            remaining_days = max(0, days_in_month - today.day + 1) # Ensure non-negative

            for category_key, status in budget_status_result.get("categories", {}).items():
                if status["remaining"] <= 0:
                    alerts.append(f"Budget for {category_key} has been exceeded. Avoid further spending.")
                    daily_limits[category_key] = 0
                else:
                    daily_limit = (status["remaining"] / remaining_days) if remaining_days > 0 else status["remaining"]
                    daily_limits[category_key] = round(daily_limit, 2)

                    if status["status"] == "warning":
                        alerts.append(f"Budget for {category_key} is near limit ({status['percentage_used']}% used). Limit daily spending to ~${daily_limit:.2f}.")

            summary_status = budget_status_result.get("summary", {}).get("status", "unknown")
            if summary_status == "exceeded":
                recommendations.append("Overall budget for tracked categories exceeded. Reduce non-essential spending.")
            elif summary_status == "warning":
                recommendations.append("Overall budget for tracked categories is near limit. Monitor spending carefully.")
            else:
                recommendations.append("Overall budget for tracked categories is on track. Keep up the good work!")

            total_budget = budget_status_result.get("summary", {}).get("total_budget", 0)
            if self.income > 0 and total_budget < self.income:
                 potential_savings = self.income - total_budget
                 recommendations.append(f"Income (${self.income:.2f}) exceeds total budgeted amount (${total_budget:.2f}). Consider allocating ${potential_savings:.2f} to savings or other goals.")

        except Exception as e:
            logger.error(f"Error generating recommendation details: {e}", exc_info=True)
            return {
                 "budget_status": budget_status_result,
                 "error": "Failed to generate detailed recommendations."
             }


        logger.info(f"Generated {len(alerts)} alerts and {len(recommendations)} recommendations.")
        return {
            "budget_status": budget_status_result,
            "daily_limits": daily_limits,
            "alerts": alerts,
            "recommendations": recommendations,
            "remaining_days_in_month": remaining_days
        }

    def suggest_budget_adjustments(self, ltm: LongTermMemory, months_of_history: int = 3) -> Dict[str, Any]:
        logger.info(f"Suggesting budget adjustments based on {months_of_history} months history.")
        today = datetime.now()
        start_date = (today - timedelta(days=30 * months_of_history)).replace(day=1)
        try:
             historical_expenses = ltm.get_expenses(start_date=start_date, end_date=today)
        except Exception as e:
             logger.error(f"Failed to fetch historical expenses from LTM: {e}", exc_info=True)
             return {"error": "Failed to retrieve historical expense data from memory."}


        if historical_expenses.empty:
            logger.warning(f"No expense data found for the past {months_of_history} months.")
            return {"error": f"No expense data available for the past {months_of_history} months"}

        if 'category' not in historical_expenses.columns or 'amount' not in historical_expenses.columns or 'date' not in historical_expenses.columns:
             return {"error": "Historical expense data lacks required columns (date, category, amount)."}

        # Ensure types are correct
        historical_expenses['amount'] = pd.to_numeric(historical_expenses['amount'], errors='coerce').fillna(0)
        if not pd.api.types.is_datetime64_any_dtype(historical_expenses['date']):
             historical_expenses['date'] = pd.to_datetime(historical_expenses['date'], errors='coerce')
             historical_expenses.dropna(subset=['date'], inplace=True) # Remove rows where date conversion failed


        historical_expenses['month'] = historical_expenses['date'].dt.strftime('%Y-%m')
        historical_expenses['category_key'] = historical_expenses['category'].str.lower().str.strip()

        monthly_category_spending = historical_expenses[historical_expenses['amount'] < 0].groupby(['month', 'category_key'])['amount'].sum().abs()
        avg_monthly_spending = monthly_category_spending.groupby('category_key').mean().to_dict()
        logger.debug(f"Average monthly spending by category key: {avg_monthly_spending}")

        suggested_adjustments = {}
        for category_key, avg_spent in avg_monthly_spending.items():
            current_budget = self.budgets.get(category_key, 0)
            suggestion = ""
            adjustment = current_budget 

            if current_budget == 0:
                adjustment = avg_spent
                suggestion = f"Set new budget based on average spending of ${avg_spent:.2f}"
            else:
                difference = current_budget - avg_spent
                difference_pct = (difference / current_budget * 100) if current_budget > 0 else 0

                threshold = 20.0
                if abs(difference_pct) >= threshold:
                    if difference < 0: 
                        suggestion = f"Increase budget by ${abs(difference):.2f} (avg spending ${avg_spent:.2f} vs budget ${current_budget:.2f})"
                        adjustment = avg_spent 
                    else: 
                        suggestion = f"Consider decreasing budget by ${difference:.2f} (avg spending ${avg_spent:.2f} vs budget ${current_budget:.2f})"
                        adjustment = avg_spent 
                else:
                    suggestion = "Budget seems aligned with recent spending."
                    adjustment = current_budget

            suggested_adjustments[category_key] = {
                "current_budget": current_budget,
                "average_spending": round(avg_spent, 2),
                "suggested_budget": round(adjustment, 2),
                "suggestion": suggestion
            }

        budgeted_keys_with_spending = set(avg_monthly_spending.keys())
        for category_key, budget in self.budgets.items():
            if category_key not in budgeted_keys_with_spending:
                suggested_adjustments[category_key] = {
                    "current_budget": budget,
                    "average_spending": 0.0,
                    "suggested_budget": 0.0, 
                    "suggestion": "No spending recorded in this category recently. Consider removing budget or reallocating."
                }

        logger.info(f"Generated {len(suggested_adjustments)} budget adjustment suggestions.")
        return {
            "suggested_adjustments": suggested_adjustments,
            "months_analyzed": months_of_history
        }

    def delete_category_budget(self, category: str) -> Dict[str, Any]:
        """Deletes the budget for a specific category."""
        category_key = category.lower().strip()
        if category_key not in self.budgets:
            return {"error": f"No budget set for category '{category}'"}
        del self.budgets[category_key]
        self.save_budget()
        logger.info(f"Budget for category '{category_key}' deleted.")
        return {"success": f"Budget for '{category}' deleted"}

    def generate_investment_recommendation(self, surplus: float) -> Dict[str, list]:
        """Suggest investment options based on budget surplus (simple example)."""
        if surplus <= 0:
            return {"recommendation": "No surplus available for investment."}

        recommendations = {
            'high_risk': ['Technology Stocks', 'Cryptocurrency Index'],
            'medium_risk': ['Broad Market Index Funds (e.g., S&P 500)', 'Real Estate Investment Trusts (REITs)'],
            'low_risk': ['High-Yield Savings Account', 'Government Bonds', 'Certificates of Deposit (CDs)']
        }
        if surplus > 1000:
            level = 'high_risk'
        elif surplus > 250:
            level = 'medium_risk'
        else:
            level = 'low_risk'

        return {
             "risk_level": level.replace('_', ' ').title(),
             "suggestions": recommendations[level],
             "note": "This is a very basic suggestion. Consult a financial advisor for personalized investment advice."
         }