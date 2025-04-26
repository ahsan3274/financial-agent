"""
/tools/financial_planner.py
Financial planning module
"""
from datetime import datetime
from typing import Dict

class FinancialPlanner:
    def __init__(self, net_worth_calculator):
        self.net_worth = net_worth_calculator

    def create_retirement_plan(self, user_data: Dict) -> Dict:
        """Generate retirement savings plan"""
        current_age = user_data['age']
        retirement_age = user_data['target_retirement_age']
        current_savings = user_data['current_savings']
        
        years_to_retire = retirement_age - current_age
        annual_contribution = ((user_data['desired_income'] * 25) - current_savings) / years_to_retire
        
        return {
            'annual_contribution': annual_contribution,
            'recommended_portfolio': self._suggest_portfolio(user_data['risk_tolerance'])
        }

    def _suggest_portfolio(self, risk_tolerance: str) -> Dict:
        """Suggest investment portfolio based on risk tolerance"""
        portfolios = {
            'conservative': {'bonds': 70, 'stocks': 30},
            'moderate': {'bonds': 50, 'stocks': 50},
            'aggressive': {'bonds': 30, 'stocks': 70}
        }
        return portfolios.get(risk_tolerance.lower(), portfolios['moderate'])