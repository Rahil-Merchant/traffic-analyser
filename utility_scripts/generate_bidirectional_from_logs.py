import pandas as pd
import os
import sys
from datetime import datetime

# --- Configuration ---
# Define the input and output directories.
# The script will scan for source Excel files in INPUT_DIRECTORY.
# The newly generated bidirectional reports will be saved in OUTPUT_DIRECTORY.
INPUT_DIRECTORY = r'C:\Users\swest\Desktop\Link Counts'
OUTPUT_DIRECTORY = r'C:\Users\swest\Desktop\Link Counts - Bidirectional'
IGNORE_EXISTING = True

# Ensure the main project directory is in the Python path
# This allows the script to import modules from the 'reporting' and 'core' packages
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from reporting.excel_report import create_bidirectional_link_count_report


def reverse_get_display_direction(display_name):
    """Maps a display name (e.g., 'North') back to its key (e.g., 'north')."""
    return display_name.lower()

def process_log_file(filepath):
    """
    Processes a single comprehensive log file to generate a bidirectional report.
    """
    print(f"\n--- Processing file: {os.path.basename(filepath)} ---")

    try:
        # 1. --- File Identification & Validation ---
        # Check for the required sheets to ensure it's a valid source file.
        with pd.ExcelFile(filepath) as xls:
            if 'Vehicle Logs' not in xls.sheet_names or 'RSA Output' not in xls.sheet_names:
                print("Skipping: File does not contain 'Vehicle Logs' and 'RSA Output' sheets.")
                return

        # 2. --- Data Loading ---
        df_logs = pd.read_excel(filepath, sheet_name='Vehicle Logs')
        if df_logs.empty:
            print("Skipping: 'Vehicle Logs' sheet is empty.")
            return
        
        original_filename_base = os.path.splitext(os.path.basename(filepath))[0]
        custom_filename = f"{original_filename_base}_bidirectional_link_counts.xlsx"
        if IGNORE_EXISTING and os.path.isfile(os.path.join(OUTPUT_DIRECTORY,custom_filename)):
            print("Processed already")
            return
        
        df_intervals = pd.read_excel(filepath, sheet_name='Intervals')
        start_date = datetime.strptime(df_intervals['Date'].tolist()[0], "%m/%d/%Y")
        #* Assuming 6AM start time - info not in sheet
        start_time = start_date.replace(hour=6, minute=0, second=0)

        # 3. --- Direction Filtering & Extraction ---
        # Find all unique exit directions present in the logs.
        unique_exit_dirs_display = df_logs['Exit Direction'].dropna().unique()
        unique_exit_dirs = [reverse_get_display_direction(d) for d in unique_exit_dirs_display]

        # Proceed only if there are exactly two unique exit directions.
        if len(unique_exit_dirs) != 2:
            print(f"Skipping: Found {len(unique_exit_dirs)} exit directions. Expected 2. Directions: {unique_exit_dirs}")
            return

        print(f"Found 2 exit directions: {unique_exit_dirs}")

        # 4. --- Determine Primary Direction using Cardinal Order ---
        # The most reliable method without replicating complex RSA logic is to use a fixed cardinal order.
        # This ensures consistency across all generated reports.
        cardinal_order = ['north', 'east', 'south', 'west']
        
        # Sort the identified unique directions based on the cardinal order.
        # This makes the assignment deterministic.
        unique_exit_dirs.sort(key=lambda d: cardinal_order.index(d) if d in cardinal_order else -1)
        
        primary_direction = unique_exit_dirs[0]
        secondary_direction = unique_exit_dirs[1]

        print(f"Assigned Primary (by cardinal order): {primary_direction}, Secondary: {secondary_direction}")

        # 5. --- Reconstruct `completed_paths_data` ---
        # The reporting function needs data in a specific list-of-dicts format.
        # We reconstruct this from the 'Vehicle Logs' DataFrame.
        
        df_logs['exit_time_dt_obj'] = df_logs['Exit Time'].apply(
            lambda x: datetime.combine(start_date, pd.to_datetime(x, format='%H:%M:%S.%f', errors='coerce').time())
        )

        completed_paths_data = []
        for _, row in df_logs.iterrows():
            completed_paths_data.append({
                'id': row['Vehicle Tracker ID'],
                'type': row['Object Name'],
                'entry_direction': reverse_get_display_direction(row['Entry Direction']),
                'exit_direction': reverse_get_display_direction(row['Exit Direction']),
                'entry_time': None, # Not strictly needed for this report
                'exit_time': row['exit_time_dt_obj'],
                'time_in_intersection': row['Time in Intersection (s)'],
                'status': 'exited'
            })
        
        if not completed_paths_data:
            print("Skipping: No valid data reconstructed from logs.")
            return

        # 6. --- Call the Report Generation Function ---
        print(f"Calling bidirectional report generator for '{original_filename_base}'...")
        
        # The station number might be part of the filename or metadata, extract if possible
        station_number_str = "" # Placeholder

        create_bidirectional_link_count_report(
            completed_paths_data=completed_paths_data,
            start_datetime_video=start_time,
            video_path="", # Not needed when custom_filename is used
            primary_direction=primary_direction,
            secondary_direction=secondary_direction,
            video_duration_formatted="24 hours",
            station_number_str="",
            speed_limit_kmh=None,
            night_hours_start=None,
            night_hours_end=None,
            output_dir=OUTPUT_DIRECTORY,
            custom_filename=custom_filename
        )

    except Exception as e:
        print(f"An error occurred while processing {os.path.basename(filepath)}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to find and process all relevant log files.
    """
    print("Starting bidirectional report generation from logs...")
    
    # Create the output directory if it doesn't exist.
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    found_files = 0
    # Scan all .xlsx files in the input directory.
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(INPUT_DIRECTORY, filename)
            process_log_file(filepath)
            found_files += 1

    if found_files == 0:
        print(f"No .xlsx files found in '{INPUT_DIRECTORY}'.")
    
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
