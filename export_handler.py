import pandas as pd
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Union
import json

class ExportHandler:
    @staticmethod
    def create_excel(data: Dict[str, Any]) -> BytesIO:
        """Convert analysis data to Excel format with proper formatting"""
        try:
            buffer = BytesIO()
            
            # Create Excel writer with xlsxwriter engine for better formatting
            writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
            workbook = writer.book

            # Create formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'border': 1
            })

            # Summary Sheet
            summary_data = {
                'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Region': [str(data.get('region', {}).get('name', 'Not specified'))],
                'Hazard Type': [str(data.get('hazard_type', 'Not specified'))],
                'Period': [str(data.get('period', 'Not specified'))]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            summary_sheet = writer.sheets['Summary']
            
            # Format Summary sheet
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
                summary_sheet.set_column(col_num, col_num, 20)

            # Trends Sheet
            if 'trends' in data and isinstance(data['trends'], dict):
                trends_data = {k: [float(v) if isinstance(v, (int, float)) else v] 
                             for k, v in data['trends'].items()}
                trends_df = pd.DataFrame(trends_data)
                trends_df.to_excel(writer, sheet_name='Trends', index=False)
                trends_sheet = writer.sheets['Trends']
                
                # Format Trends sheet
                for col_num, value in enumerate(trends_df.columns.values):
                    trends_sheet.write(0, col_num, value, header_format)
                    trends_sheet.set_column(col_num, col_num, 15)

            # Statistical Analysis Sheet
            if 'statistical_analysis' in data and isinstance(data['statistical_analysis'], dict):
                stats = data['statistical_analysis']
                if 'basic_stats' in stats and isinstance(stats['basic_stats'], dict):
                    stats_data = {k: [float(v) if isinstance(v, (int, float)) else v] 
                                for k, v in stats['basic_stats'].items()}
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                    stats_sheet = writer.sheets['Statistics']
                    
                    # Format Statistics sheet
                    for col_num, value in enumerate(stats_df.columns.values):
                        stats_sheet.write(0, col_num, value, header_format)
                        stats_sheet.set_column(col_num, col_num, 15)

            # Yearly Data Sheet
            if 'yearly_statistics' in data and isinstance(data['yearly_statistics'], dict):
                yearly_stats = data['yearly_statistics']
                yearly_data = {
                    'Year': yearly_stats.get('years', []),
                    'Frequency': [float(f) for f in yearly_stats.get('frequencies', [])],
                    'Intensity': [float(i) for i in yearly_stats.get('intensities', [])]
                }
                yearly_df = pd.DataFrame(yearly_data)
                yearly_df.to_excel(writer, sheet_name='Yearly Data', index=False)
                yearly_sheet = writer.sheets['Yearly Data']
                
                # Format Yearly Data sheet
                for col_num, value in enumerate(yearly_df.columns.values):
                    yearly_sheet.write(0, col_num, value, header_format)
                    yearly_sheet.set_column(col_num, col_num, 12)

            # Save and close
            writer.close()
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            raise Exception(f"Error creating Excel file: {str(e)}")

    @staticmethod
    def format_data_for_export(data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for export by flattening nested structures and ensuring proper types"""
        try:
            # Handle potential MongoDB ObjectId
            if '_id' in data:
                data['id'] = str(data['_id'])
                del data['_id']

            # Ensure all nested dictionaries are properly converted
            def convert_values(obj: Union[Dict, list, Any]) -> Union[Dict, list, Any]:
                if isinstance(obj, dict):
                    return {k: convert_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_values(item) for item in obj]
                elif hasattr(obj, 'to_dict'):  # Handle Pydantic models
                    return convert_values(obj.to_dict())
                elif str(type(obj)).startswith("<class 'numpy"):  # Handle numpy types
                    return float(obj)
                return obj

            export_data = convert_values(data)
            
            # Ensure required sections exist
            if 'trends' not in export_data:
                export_data['trends'] = {}
            if 'statistical_analysis' not in export_data:
                export_data['statistical_analysis'] = {'basic_stats': {}}
            if 'yearly_statistics' not in export_data:
                export_data['yearly_statistics'] = {'years': [], 'frequencies': [], 'intensities': []}

            return export_data
        except Exception as e:
            raise Exception(f"Error formatting data for export: {str(e)}") 