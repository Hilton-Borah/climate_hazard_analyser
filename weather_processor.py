import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
from scipy import stats

class WeatherProcessor:
    def __init__(self):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # Cache for 1 hour
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.cache = {}

    async def fetch_weather_data(self, lat: float, lon: float, start_year: str, end_year: str) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo API using the official client
        """
        try:
            print(f"\nFetching weather data:")
            print(f"Location: ({lat}, {lon})")
            print(f"Period: {start_year} to {end_year}")

            # Setup the API parameters
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_year,
                "end_date": end_year,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "auto"
            }

            print("\nMaking API request with parameters:")
            print(f"URL: {url}")
            print(f"Parameters: {params}")

            # Make the API request
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]  # Process first location

            # Print metadata
            print(f"\nAPI Response Metadata:")
            print(f"Location: {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation: {response.Elevation()} m asl")
            print(f"Timezone: {response.Timezone()} ({response.TimezoneAbbreviation()})")

            # Process daily data
            daily = response.Daily()
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "temp_max": daily.Variables(0).ValuesAsNumpy(),  # temperature_2m_max
                "temp_min": daily.Variables(1).ValuesAsNumpy(),  # temperature_2m_min
                "precipitation": daily.Variables(2).ValuesAsNumpy(),  # precipitation_sum
            }

            # Create DataFrame
            df = pd.DataFrame(data=daily_data)
            
            print(f"\nData Summary:")
            print(f"Total days: {len(df)}")
            print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print("\nTemperature Statistics:")
            print(f"Max temperature range: {df['temp_max'].min():.1f}°C to {df['temp_max'].max():.1f}°C")
            print(f"Min temperature range: {df['temp_min'].min():.1f}°C to {df['temp_min'].max():.1f}°C")
            print("\nPrecipitation Statistics:")
            print(f"Total precipitation days: {(df['precipitation'] > 0).sum()}")
            print(f"Max daily precipitation: {df['precipitation'].max():.1f}mm")
            
            # Validate data
            if df.empty:
                raise Exception("No data returned from API")
            
            if df['temp_max'].isna().all() or df['temp_min'].isna().all():
                raise Exception("Temperature data is missing")
                
            if df['precipitation'].isna().all():
                raise Exception("Precipitation data is missing")
                
            return df

        except Exception as e:
            print(f"Error in fetch_weather_data: {str(e)}")
            raise

    def detect_heatwaves(self, df: pd.DataFrame, percentile: float = 90) -> List[Dict[str, Any]]:
        """
        Detect heatwaves using the 90th percentile method (changed from 95th)
        A heatwave is defined as 3+ consecutive days above the 90th percentile
        """
        print(f"Starting heatwave detection with {len(df)} days of data")
        
        # Calculate monthly temperature thresholds
        df['month'] = df['date'].dt.month
        thresholds = {}
        for month in range(1, 13):
            month_data = df[df['month'] == month]['temp_max']
            if not month_data.empty:
                thresholds[month] = np.percentile(month_data, percentile)
                print(f"Month {month} threshold: {thresholds[month]:.1f}°C")
        
        # Find days above monthly threshold
        hot_days = df.apply(lambda row: row['temp_max'] > thresholds.get(row['month'], 0), axis=1)
        hot_days_count = hot_days.sum()
        print(f"Found {hot_days_count} days above threshold")
        
        # Find sequences of 3+ consecutive hot days
        heatwaves = []
        current_streak = 0
        start_date = None
        
        for date, is_hot in zip(df['date'], hot_days):
            if is_hot:
                if current_streak == 0:
                    start_date = date
                current_streak += 1
            else:
                if current_streak >= 3:
                    max_temp = df.loc[
                        (df['date'] >= start_date) & 
                        (df['date'] < date),
                        'temp_max'
                    ].max()
                    heatwaves.append({
                        'start_date': start_date,
                        'end_date': date - timedelta(days=1),
                        'duration': current_streak,
                        'max_temp': max_temp
                    })
                    print(f"Detected heatwave: {start_date.date()} to {(date - timedelta(days=1)).date()}, duration: {current_streak} days, max temp: {max_temp}°C")
                current_streak = 0
        
        # Check for heatwave at the end of the data
        if current_streak >= 3:
            max_temp = df.loc[
                (df['date'] >= start_date),
                'temp_max'
            ].max()
            heatwaves.append({
                'start_date': start_date,
                'end_date': df['date'].iloc[-1],
                'duration': current_streak,
                'max_temp': max_temp
            })
            print(f"Detected final heatwave: {start_date.date()} to {df['date'].iloc[-1].date()}, duration: {current_streak} days, max temp: {max_temp}°C")
        
        print(f"Total heatwaves detected: {len(heatwaves)}")
        return heatwaves

    def detect_drought(self, df: pd.DataFrame, window: int = 30) -> List[Dict[str, Any]]:
        """
        Detect drought periods using precipitation data
        A drought is defined as a period with significantly below-average precipitation
        """
        # Calculate rolling average precipitation
        df['rolling_precip'] = df['precipitation'].rolling(window=window, center=True).mean()
        
        # Calculate historical average
        historical_avg = df['precipitation'].mean()
        drought_threshold = historical_avg * 0.5  # 50% of historical average
        
        # Identify drought periods
        drought_periods = []
        in_drought = False
        start_date = None
        
        for date, precip in zip(df['date'], df['rolling_precip']):
            if pd.notna(precip):  # Check for non-NaN values
                if precip < drought_threshold and not in_drought:
                    in_drought = True
                    start_date = date
                elif (precip >= drought_threshold or pd.isna(precip)) and in_drought:
                    in_drought = False
                    if (date - start_date).days >= 30:  # Only count droughts lasting at least 30 days
                        drought_periods.append({
                            'start_date': start_date,
                            'end_date': date,
                            'duration': (date - start_date).days,
                            'avg_precipitation': df.loc[
                                (df['date'] >= start_date) & 
                                (df['date'] <= date),
                                'precipitation'
                            ].mean()
                        })
        
        return drought_periods

    def detect_heavy_rainfall(self, df: pd.DataFrame, percentile: float = 95) -> List[Dict[str, Any]]:
        """
        Detect heavy rainfall events
        Heavy rainfall is defined as precipitation above the 95th percentile
        """
        # Remove days with no precipitation
        rain_days = df[df['precipitation'] > 0]
        if len(rain_days) == 0:
            return []
            
        threshold = np.percentile(rain_days['precipitation'], percentile)
        
        heavy_rain_events = []
        for date, precip in zip(df['date'], df['precipitation']):
            if precip > threshold:
                heavy_rain_events.append({
                    'date': date,
                    'precipitation': precip
                })
        
        return heavy_rain_events

    def calculate_trends(self, events: List[Dict[str, Any]], start_year: int, end_year: int) -> Dict[str, float]:
        """
        Calculate trends in frequency and intensity of events
        """
        print(f"\nCalculating trends for period {start_year}-{end_year}")
        print(f"Total events to process: {len(events)}")
        
        # Convert events to yearly statistics
        years = list(range(start_year, end_year + 1))
        yearly_stats = {year: {'count': 0, 'intensity': []} for year in years}
        
        # Process each event and aggregate by year
        for event in events:
            event_year = None
            intensity_value = None
            
            # Handle different event types
            if isinstance(event.get('start_date'), datetime):
                event_year = event['start_date'].year
                if 'max_temp' in event:
                    intensity_value = float(event['max_temp'])
                elif 'avg_precipitation' in event:
                    intensity_value = float(event['avg_precipitation'])
            else:
                event_year = event['date'].year
                if 'precipitation' in event:
                    intensity_value = float(event['precipitation'])
            
            if event_year in yearly_stats and intensity_value is not None:
                yearly_stats[event_year]['count'] += 1
                yearly_stats[event_year]['intensity'].append(intensity_value)
                print(f"Added event for year {event_year}: count={yearly_stats[event_year]['count']}, intensity={intensity_value}")
        
        # Calculate year-wise averages and prepare data for trend analysis
        year_list = []
        frequency_list = []
        intensity_list = []
        
        print("\nYearly statistics:")
        for year in years:
            year_list.append(year)
            frequency_list.append(yearly_stats[year]['count'])
            intensities = yearly_stats[year]['intensity']
            avg_intensity = np.mean(intensities) if intensities else 0
            intensity_list.append(avg_intensity)
            print(f"Year {year}: {yearly_stats[year]['count']} events, avg intensity: {avg_intensity:.2f}")
        
        # Calculate trends using linear regression
        if len(year_list) > 1:  # Need at least 2 points for trend
            # Frequency trend
            slope_freq, _, _, _, _ = stats.linregress(year_list, frequency_list)
            
            # Intensity trend
            slope_int, _, _, _, _ = stats.linregress(year_list, intensity_list)
            
            # Calculate percent change from first to last year
            first_freq = frequency_list[0] if frequency_list[0] != 0 else 1
            first_int = intensity_list[0] if intensity_list[0] != 0 else 1
            
            last_freq = frequency_list[-1]
            last_int = intensity_list[-1]
            
            freq_percent_change = ((last_freq - first_freq) / first_freq) * 100
            int_percent_change = ((last_int - first_int) / first_int) * 100
            
            print(f"\nTrend analysis:")
            print(f"Frequency trend: {freq_percent_change:.2f}%")
            print(f"Intensity trend: {int_percent_change:.2f}%")
            
            return {
                'trend_coefficient': freq_percent_change,
                'percent_change': int_percent_change,
                'yearly_data': {
                    'years': year_list,
                    'frequencies': frequency_list,
                    'intensities': intensity_list
                }
            }
        else:
            print("\nNot enough data points for trend analysis")
            return {
                'trend_coefficient': 0,
                'percent_change': 0,
                'yearly_data': {
                    'years': year_list,
                    'frequencies': frequency_list,
                    'intensities': intensity_list
                }
            }

    async def analyze_region(self, lat: float, lon: float, start_year: str, end_year: str, hazard_type: str) -> Dict[str, Any]:
        """
        Analyze climate hazards for a specific region
        """
        try:
            # Convert date strings to datetime objects
            start_date = datetime.strptime(start_year, "%Y-%m-%d")
            end_date = datetime.strptime(end_year, "%Y-%m-%d")
            
            # Fetch weather data
            df = await self.fetch_weather_data(lat, lon, start_year, end_year)
            
            # Initialize variables for analysis
            events = []
            event_type = ""
            
            # Perform analysis based on hazard type
            if hazard_type == "heatwave":
                events = self.detect_heatwaves(df)
                event_type = "heatwaves"
            elif hazard_type == "drought":
                events = self.detect_drought(df)
                event_type = "drought periods"
            elif hazard_type == "rainfall":
                events = self.detect_heavy_rainfall(df)
                event_type = "heavy rainfall events"
            else:
                raise ValueError(f"Unsupported hazard type: {hazard_type}")
            
            # Calculate trends
            trends = self.calculate_trends(events, start_date.year, end_date.year)
            
            # Calculate years difference
            years_diff = end_date.year - start_date.year + 1
            
            # Prepare summary statistics
            total_events = len(events)
            avg_per_year = total_events / years_diff if years_diff > 0 else 0
            avg_duration = np.mean([event.get('duration', 1) for event in events]) if events else 0
            
            if hazard_type == "heatwave":
                avg_intensity = np.mean([event['max_temp'] for event in events]) if events else 0
                intensity_unit = "°C"
            elif hazard_type in ["drought", "rainfall"]:
                avg_intensity = np.mean([event.get('precipitation', 0) for event in events]) if events else 0
                intensity_unit = "mm"
            
            summary = {
                "total_events": total_events,
                "avg_per_year": round(float(avg_per_year), 2),
                "average_duration": round(float(avg_duration), 2),
                "average_intensity": round(float(avg_intensity), 2),
                "intensity_unit": intensity_unit,
                "trend_direction": "increasing" if trends['trend_coefficient'] > 0 else "decreasing",
                "significant": abs(trends['trend_coefficient']) > 10,
                "yearly_statistics": trends['yearly_data']
            }
            
            return {
                "trends": trends,
                "summary": summary
            }
            
        except Exception as e:
            print(f"Error in analyze_region: {str(e)}")
            raise

    def _get_trend_description(self, trend_coefficient: float) -> str:
        """
        Convert trend coefficient to human-readable description
        """
        if trend_coefficient > 20:
            return "Strong increasing trend"
        elif trend_coefficient > 5:
            return "Moderate increasing trend"
        elif trend_coefficient > 1:
            return "Slight increasing trend"
        elif trend_coefficient < -20:
            return "Strong decreasing trend"
        elif trend_coefficient < -5:
            return "Moderate decreasing trend"
        elif trend_coefficient < -1:
            return "Slight decreasing trend"
        else:
            return "No significant trend" 