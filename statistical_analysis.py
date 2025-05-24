import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import pandas as pd

class StatisticalAnalysis:
    @staticmethod
    def mann_kendall_test(data: List[float]) -> Dict[str, float]:
        """
        Perform Mann-Kendall trend test
        Returns: Dictionary containing test statistic, p-value, and trend direction
        """
        if len(data) < 3:
            return {
                "statistic": None,
                "p_value": None,
                "trend": "insufficient_data",
                "confidence": None
            }
        
        try:
            statistic, p_value = stats.kendalltau(range(len(data)), data)
            
            # Determine trend direction and confidence
            trend = "no_trend"
            if p_value < 0.05:  # 95% confidence level
                trend = "increasing" if statistic > 0 else "decreasing"
            
            confidence = (1 - p_value) * 100 if p_value < 1 else 0
            
            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "trend": trend,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "error": str(e),
                "trend": "error_in_calculation"
            }

    @staticmethod
    def calculate_percentile_threshold(data: List[float], percentile: float = 95) -> float:
        """Calculate percentile threshold for extreme event detection"""
        return float(np.percentile(data, percentile))

    @staticmethod
    def analyze_seasonality(data: List[float], frequency: int = 12) -> Dict[str, float]:
        """
        Analyze seasonal patterns in the data
        frequency: number of observations per cycle (e.g., 12 for monthly data)
        """
        if len(data) < frequency * 2:
            return {"error": "Insufficient data for seasonal analysis"}
        
        try:
            # Decompose time series
            series = pd.Series(data)
            seasonal_means = [float(series[i::frequency].mean()) for i in range(frequency)]
            
            # Calculate seasonal strength
            seasonal_variance = float(np.var(seasonal_means))
            total_variance = float(np.var(data))
            seasonal_strength = float(seasonal_variance / total_variance if total_variance > 0 else 0)
            
            return {
                "seasonal_strength": seasonal_strength,
                "seasonal_pattern": seasonal_means,
                "has_seasonality": bool(seasonal_strength > 0.1)  # Convert numpy.bool_ to Python bool
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def detect_change_points(data: List[float], window: int = 10) -> Dict[str, List]:
        """
        Detect significant changes in the time series
        window: size of the rolling window for change detection
        """
        if len(data) < window * 2:
            return {"error": "Insufficient data for change point detection"}
        
        try:
            # Calculate rolling statistics
            series = pd.Series(data)
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            
            # Detect significant changes
            z_scores = (series - rolling_mean) / rolling_std
            change_points = []
            
            for i in range(window, len(data)):
                if abs(z_scores[i]) > 2:  # 2 standard deviations threshold
                    change_points.append({
                        "index": int(i),  # Convert numpy.int64 to Python int
                        "value": float(data[i]),
                        "z_score": float(z_scores[i])
                    })
            
            return {
                "change_points": change_points,
                "total_changes": len(change_points)
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def comprehensive_analysis(data: List[float], frequency: int = 12) -> Dict[str, any]:
        """
        Perform comprehensive statistical analysis including trends, seasonality,
        and change points
        """
        # Convert all numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [float(x) for x in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(x) for x in obj]
            return obj

        result = {
            "trend_analysis": StatisticalAnalysis.mann_kendall_test(data),
            "seasonality": StatisticalAnalysis.analyze_seasonality(data, frequency),
            "change_points": StatisticalAnalysis.detect_change_points(data),
            "basic_stats": {
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data))
            }
        }
        
        # Convert all numpy types to native Python types
        return convert_to_native(result) 