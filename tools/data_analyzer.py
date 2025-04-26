"""
/tools/data_analyzer.py
Data analyzer module for processing financial data from LTM, identifying trends and insights.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging # Use logging
from typing import Dict, List, Tuple, Optional, Any

try:
    from memory.long_term import LongTermMemory
except ImportError:
    LongTermMemory = object 

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Analyzes financial data (fetched from LTM) to extract insights, trends, and patterns.
    """

    def __init__(self):
        """Initialize the data analyzer."""
        
        self.expense_data = pd.DataFrame()
        logger.info("DataAnalyzer initialized.")

    def set_data(self, expense_data: pd.DataFrame) -> None:
        """
        Set or update the internal expense data for analysis.
        Ensures 'date' column is datetime.
        """
        if not isinstance(expense_data, pd.DataFrame):
             logger.error("Failed to set data: Input is not a pandas DataFrame.")
             self.expense_data = pd.DataFrame() # Reset to empty
             return

        logger.debug(f"Setting internal data with DataFrame of shape {expense_data.shape}")
        df = expense_data.copy() # Work on a copy

        # Ensure required columns exist and handle date conversion
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # Optionally drop rows where date conversion failed
                    original_len = len(df)
                    df.dropna(subset=['date'], inplace=True)
                    if len(df) < original_len:
                        logger.warning(f"Dropped {original_len - len(df)} rows due to invalid dates during conversion.")
                except Exception as e:
                    logger.error(f"Error converting 'date' column to datetime: {e}. Date filtering might fail.", exc_info=True)
        else:
            logger.warning("Setting data without a 'date' column. Time-based analysis will not be possible.")

        if 'amount' in df.columns:
             df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        else:
             logger.warning("Setting data without an 'amount' column. Value-based analysis might fail.")

        if 'category' not in df.columns:
             logger.warning("Setting data without a 'category' column. Category analysis will not be possible.")
             df['category'] = 'Unknown' # Add placeholder if missing


        self.expense_data = df

    def monthly_spending_summary(self) -> Dict[str, Any]:
        """Generate a summary of monthly spending from the internal data."""
        if self.expense_data.empty:
            return {"error": "No expense data set for analysis"}
        if 'date' not in self.expense_data.columns or not pd.api.types.is_datetime64_any_dtype(self.expense_data['date']):
             return {"error": "Cannot generate monthly summary without a valid 'date' column."}
        if 'amount' not in self.expense_data.columns:
            return {"error": "Cannot generate monthly summary without an 'amount' column."}


        spending_data = self.expense_data[self.expense_data['amount'] < 0].copy()
        spending_data['amount'] = spending_data['amount'].abs() 

        if spending_data.empty:
            return {"message": "No spending transactions found in the data."}


        spending_data['month_year'] = spending_data['date'].dt.strftime('%Y-%m')

        monthly_stats = spending_data.groupby('month_year').agg(
            total_spent=('amount', 'sum'),
            avg_transaction=('amount', 'mean'),
            num_transactions=('amount', 'count')
        ).reset_index()

        if monthly_stats.empty:
             return {"message": "No monthly spending data to summarize."}

        result = {
            'monthly_summary': monthly_stats.to_dict(orient='records'),
            'average_monthly_spend': round(monthly_stats['total_spent'].mean(), 2),
            'highest_spending_month': monthly_stats.loc[monthly_stats['total_spent'].idxmax()].to_dict() if not monthly_stats.empty else {},
            'lowest_spending_month': monthly_stats.loc[monthly_stats['total_spent'].idxmin()].to_dict() if not monthly_stats.empty else {}
        }
        logger.debug("Generated monthly spending summary.")
        return result

    def category_analysis(self) -> Dict[str, Any]:
        """Analyze spending by category from the internal data."""
        if self.expense_data.empty:
            return {"error": "No expense data set for analysis"}
        if 'category' not in self.expense_data.columns:
            return {"error": "Cannot analyze categories without a 'category' column."}
        if 'amount' not in self.expense_data.columns:
            return {"error": "Cannot analyze categories without an 'amount' column."}

        spending_data = self.expense_data[self.expense_data['amount'] < 0].copy()
        spending_data['amount'] = spending_data['amount'].abs()

        if spending_data.empty:
            return {"message": "No spending transactions found for category analysis."}

        category_stats = spending_data.groupby('category').agg(
            total_spent=('amount', 'sum'),
            avg_transaction=('amount', 'mean'),
            num_transactions=('amount', 'count')
        ).reset_index()

        total_spending = category_stats['total_spent'].sum()
        if total_spending > 0:
             category_stats['percentage'] = (category_stats['total_spent'] / total_spending * 100).round(2)
        else:
             category_stats['percentage'] = 0.0

        category_stats = category_stats.sort_values('total_spent', ascending=False)

        logger.debug("Generated category analysis.")
        return {
            'category_breakdown': category_stats.to_dict(orient='records'),
            'top_category': category_stats.iloc[0].to_dict() if not category_stats.empty else {},
            'total_spending': round(total_spending, 2)
        }

    def spending_trends(self, period: str = 'monthly') -> Dict[str, Any]:
        """Analyze spending trends over time from the internal data."""
        if self.expense_data.empty:
            return {"error": "No expense data set for analysis"}
        if 'date' not in self.expense_data.columns or not pd.api.types.is_datetime64_any_dtype(self.expense_data['date']):
             return {"error": "Cannot analyze trends without a valid 'date' column."}
        if 'amount' not in self.expense_data.columns:
            return {"error": "Cannot analyze trends without an 'amount' column."}

        spending_data = self.expense_data[self.expense_data['amount'] < 0].copy()
        spending_data['amount'] = spending_data['amount'].abs()

        if spending_data.empty:
            return {"message": "No spending transactions found for trend analysis."}


        valid_periods = {'daily': '%Y-%m-%d', 'weekly': '%Y-W%W', 'monthly': '%Y-%m'}
        if period not in valid_periods:
            logger.warning(f"Invalid period '{period}'. Defaulting to 'monthly'.")
            period = 'monthly'

        spending_data = spending_data.set_index('date')

        freq_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'} # W-MON starts week on Monday
        pandas_freq = freq_map.get(period, 'MS') # Default to Monthly Start if period is invalid
        trend_data = spending_data['amount'].resample(pandas_freq).sum().reset_index()
        trend_data.rename(columns={'date':'period_start', 'amount':'total_spent'}, inplace=True)
        trend_data['period_start'] = trend_data['period_start'].dt.strftime(valid_periods[period]) # Format period string


        if len(trend_data) >= 3:
            trend_data['moving_avg_3'] = trend_data['total_spent'].rolling(window=3, min_periods=1).mean().round(2)
        if len(trend_data) >= 6:
            trend_data['moving_avg_6'] = trend_data['total_spent'].rolling(window=6, min_periods=1).mean().round(2)

        trend_data['change_pct'] = (trend_data['total_spent'].pct_change() * 100).round(2)

        trend_direction = "insufficient data"
        if len(trend_data) >= 3:
            recent_trend = trend_data['total_spent'].iloc[-3:].values
            # Check if values are reasonably different before declaring trend
            if abs(recent_trend[2] - recent_trend[0]) > 0.01: # Avoid floating point noise
                 if recent_trend[2] > recent_trend[0]:
                     trend_direction = "increasing"
                 elif recent_trend[2] < recent_trend[0]:
                     trend_direction = "decreasing"
                 else:
                      trend_direction = "stable"
            else:
                 trend_direction = "stable"


        logger.debug(f"Generated spending trends for period '{period}'.")
        return {
            'trend_data': trend_data.to_dict(orient='records'),
            'trend_direction': trend_direction,
            'period_type': period
        }

    def anomaly_detection(self, z_threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalous spending transactions using Z-score from internal data."""
        if self.expense_data.empty:
            return []
        if 'amount' not in self.expense_data.columns:
             logger.warning("Cannot perform anomaly detection without 'amount' column.")
             return []

        spending_data = self.expense_data[self.expense_data['amount'] < 0].copy()

        if len(spending_data) < 2: # Need at least 2 points to calculate std dev
             logger.warning("Not enough spending data points for anomaly detection.")
             return []

        mean_amount = spending_data['amount'].mean()
        std_amount = spending_data['amount'].std()

        if std_amount is None or std_amount == 0:
            logger.warning("Standard deviation of spending is zero. Cannot calculate Z-scores.")
            return []

        spending_data['z_score'] = (spending_data['amount'] - mean_amount) / std_amount

        anomalies = spending_data[spending_data['z_score'].abs() > z_threshold].copy()

        anomalies = anomalies.sort_values(by='z_score', key=abs, ascending=False)

        if 'date' in anomalies.columns and pd.api.types.is_datetime64_any_dtype(anomalies['date']):
             anomalies['date'] = anomalies['date'].dt.strftime('%Y-%m-%d')

        result = anomalies.drop(columns=['z_score'], errors='ignore').to_dict(orient='records')
        logger.debug(f"Detected {len(result)} anomalies with Z-score > {z_threshold}.")
        return result

    def generate_spending_report_from_ltm(self, ltm: LongTermMemory, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches data from LongTermMemory and generates a comprehensive spending report.
        Returns a dictionary with a top-level 'status' key ('success' or 'error').

        Args:
            ltm: The LongTermMemory instance.
            start_date: Optional start date string ('YYYY-MM-DD').
            end_date: Optional end date string ('YYYY-MM-DD').

        Returns:
            Dictionary containing the comprehensive report or an error message.
            Example Success: {"status": "success", "report": {...report data...}}
            Example Error: {"status": "error", "message": "Error details...", "details": "Optional traceback"}
        """
        logger.info(f"DataAnalyzer: Generating report from LTM between {start_date} and {end_date}")

        s_date_obj, e_date_obj = None, None
        try:
            if start_date:
                s_date_obj = pd.to_datetime(start_date)
            if end_date:
                e_date_obj = pd.to_datetime(end_date)
        except ValueError as e:
            logger.error(f"Invalid date format provided: {e}")
            # Return clear error status
            return {"status": "error", "message": f"Invalid date format. Please use YYYY-MM-DD. ({e})"}

        try:
            expense_data = ltm.get_expenses(start_date=s_date_obj, end_date=e_date_obj)
            logger.info(f"Fetched {len(expense_data)} expenses from LTM.")
        except Exception as e:
             logger.error(f"Failed to fetch expenses from LTM: {e}", exc_info=True)
             # Return clear error status
             return {"status": "error", "message": "Failed to retrieve expense data from memory.", "details": str(e)}

        if expense_data.empty:
             logger.warning("DataAnalyzer: No expense data found in LTM for the specified criteria.")
             # Return success, but with a message indicating no data
             return {
                 "status": "success",
                 "message": "No expense data found in Long Term Memory for the specified date range.",
                 "report": { # Provide minimal report structure
                    "report_period": {"start_date": start_date, "end_date": end_date},
                    "summary_statistics": {},
                    "monthly_summary": {},
                    "category_analysis": {},
                    "spending_trends": {},
                    "anomalies_detected": []
                 }
            }

        self.set_data(expense_data)

        report_result = self.generate_spending_report(start_date=start_date, end_date=end_date)

        if report_result["status"] == "success":
            logger.info("Successfully generated spending report from LTM data.")
        else:
            # The internal method already logged the specific error
            logger.error(f"Error generating report internally. See previous logs for details.")

        return report_result 
    def generate_spending_report(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive spending report based on the currently set self.expense_data.
        Returns a dictionary with a top-level 'status' key ('success' or 'error').

        Args:
            start_date: Optional start date string ('YYYY-MM-DD') to filter internal data.
            end_date: Optional end date string ('YYYY-MM-DD') to filter internal data.

        Returns:
            Dictionary with comprehensive report data or an error message.
            Example Success: {"status": "success", "report": {...report data...}}
            Example Error: {"status": "error", "message": "Error details...", "details": "Optional traceback"}
        """
        if self.expense_data.empty:
            logger.error("generate_spending_report called but internal expense data is empty.")
            return {"status": "error", "message": "No expense data has been set for the analyzer"}

        original_internal_data = self.expense_data 
        try:
            report_data = self.expense_data.copy() 
            s_date_obj, e_date_obj = None, None

            if 'date' in report_data.columns and pd.api.types.is_datetime64_any_dtype(report_data['date']):
                if start_date:
                    try:
                        s_date_obj = pd.to_datetime(start_date)
                        report_data = report_data[report_data['date'] >= s_date_obj]
                    except ValueError:
                        logger.error(f"Invalid start_date format in generate_spending_report: {start_date}")
                        # Return clear error status from within the try block
                        return {"status": "error", "message": f"Invalid start_date format: {start_date}. Use YYYY-MM-DD."}
                if end_date:
                    try:
                        e_date_obj = pd.to_datetime(end_date)
                        report_data = report_data[report_data['date'] <= e_date_obj]
                    except ValueError:
                        logger.error(f"Invalid end_date format in generate_spending_report: {end_date}")
                        # Return clear error status from within the try block
                        return {"status": "error", "message": f"Invalid end_date format: {end_date}. Use YYYY-MM-DD."}
            elif start_date or end_date:
                logger.warning("Date filtering requested for report, but internal data lacks a valid 'date' column.")

            if report_data.empty:
                logger.warning("No data available for the specified date range within the set data.")
                return {
                    "status": "success",
                    "message": "No data available for the specified date range within the analyzer's current data.",
                     "report": {
                        "report_period": {"start_date": start_date, "end_date": end_date},
                        "summary_statistics": {},
                        "monthly_summary": {},
                        "category_analysis": {},
                        "spending_trends": {},
                        "anomalies_detected": []
                     }
                 }

            self.expense_data = report_data

            monthly_summary = self.monthly_spending_summary()
            category_analysis = self.category_analysis()

            trend_period = 'monthly' # Default
            if s_date_obj and e_date_obj:
                 days_span = (e_date_obj - s_date_obj).days
                 if days_span <= 31: trend_period = 'daily'
                 elif days_span <= 93: trend_period = 'weekly'

            spending_trends = self.spending_trends(period=trend_period) 
            anomalies = self.anomaly_detection()

            self.expense_data = original_internal_data

            report_start = s_date_obj.strftime('%Y-%m-%d') if s_date_obj else report_data['date'].min().strftime('%Y-%m-%d') if not report_data.empty and 'date' in report_data.columns else "N/A"
            report_end = e_date_obj.strftime('%Y-%m-%d') if e_date_obj else report_data['date'].max().strftime('%Y-%m-%d') if not report_data.empty and 'date' in report_data.columns else "N/A"

            total_spent = report_data[report_data['amount'] < 0]['amount'].sum() if 'amount' in report_data.columns else 0
            avg_trans = report_data[report_data['amount'] < 0]['amount'].mean() if 'amount' in report_data.columns and not report_data[report_data['amount'] < 0].empty else 0
            num_trans = len(report_data[report_data['amount'] < 0]) if 'amount' in report_data.columns else len(report_data)

            report_content = {
                "report_period": {
                    "start_date": report_start,
                    "end_date": report_end
                },
                "summary_statistics": {
                    "total_spent": round(abs(total_spent), 2),
                    "average_spending_transaction": round(abs(avg_trans), 2),
                    "spending_transaction_count": num_trans,
                    "unique_categories": report_data['category'].nunique() if 'category' in report_data.columns else 0
                },
                "monthly_summary": monthly_summary,
                "category_analysis": category_analysis,
                "spending_trends": spending_trends,
                "anomalies_detected": anomalies if isinstance(anomalies, list) else []
            }

            logger.info(f"Finished generating spending report content for period {report_start} to {report_end}.")
            return {"status": "success", "report": report_content}

        except Exception as e:
            logger.error(f"Error during generate_spending_report execution: {e}", exc_info=True)
            self.expense_data = original_internal_data
            return {"status": "error", "message": f"Error during report generation: {type(e).__name__}", "details": str(e)}