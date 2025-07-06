# project/reporting/excel_report.py
import pandas as pd
from datetime import datetime, timedelta, time
import os
from collections import OrderedDict
from config import VEHICLE_ID_MAP, REPORT_OUTPUT_DIR, VALID_MOVEMENTS, BIDIRECTIONAL_REPORT_CATEGORIES
from core.utils import get_display_direction
from loguru import logger

# --- Helper Function: Build RSA Direction Map (Corrected Exit Order) ---
def _build_rsa_direction_map(primary_direction, active_directions, valid_movements_config):
    """Builds a map from (entry_key, exit_key) to RSA movement number."""
    rsa_movement_map = OrderedDict()
    movement_number = 1

    STANDARD_ORDER = ['north', 'west', 'south', 'east'] # Reference CCW order
    OPPOSITE = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}

    if not active_directions:
        logger.debug("Error: No active directions provided for RSA map generation.")
        return rsa_movement_map

    if not primary_direction or primary_direction not in active_directions:
        primary_direction = active_directions[0]
        logger.debug(f"Warning: Primary direction invalid/inactive. Using first active: {primary_direction}")

    try:
        start_index = STANDARD_ORDER.index(primary_direction)
    except ValueError: # Fallback if primary_direction isn't in standard order
        start_index = STANDARD_ORDER.index(active_directions[0]) if active_directions else 0

    ordered_active_entry_dirs = []
    processed_dirs = set()
    if active_directions:
        # Build the ordered list of active entry directions starting from primary
        for i in range(len(STANDARD_ORDER)):
            current_dir = STANDARD_ORDER[(start_index + i) % len(STANDARD_ORDER)]
            if current_dir in active_directions and current_dir not in processed_dirs:
                ordered_active_entry_dirs.append(current_dir)
                processed_dirs.add(current_dir)

    logger.debug(f"RSA Numbering: Ordered Active Entry Dirs = {ordered_active_entry_dirs}")

    # Iterate through ordered entry dirs
    for entry_dir in ordered_active_entry_dirs:
        logger.debug(f"  Processing Entry: {entry_dir}")
        try:
            entry_idx = STANDARD_ORDER.index(entry_dir)
        except ValueError:
            logger.debug(f"    Skipping invalid entry_dir: {entry_dir}")
            continue

        # Determine relative directions based on the standard CCW order
        straight_dir = OPPOSITE.get(entry_dir)
        right_dir = STANDARD_ORDER[(entry_idx + 1) % len(STANDARD_ORDER)] # Next in CCW order is right
        left_dir = STANDARD_ORDER[(entry_idx - 1 + len(STANDARD_ORDER)) % len(STANDARD_ORDER)] # Previous is left

        # Define the relative exit order: Left -> Straight -> Right
        relative_exit_order = [left_dir, straight_dir, right_dir]
        logger.debug(f"    Relative Exit Order Used: {relative_exit_order}")

        allowed_exits = valid_movements_config.get(entry_dir, [])
        processed_exits_for_entry = set() # Track exits processed for this entry

        # Iterate through relative exit order
        for exit_dir in relative_exit_order:
            # Check if this exit is valid
            is_valid = (
                exit_dir is not None and
                exit_dir in active_directions and      # Is it an active zone?
                exit_dir != entry_dir and             # Not a U-turn?
                exit_dir in allowed_exits and         # Allowed by config?
                exit_dir not in processed_exits_for_entry # Not already numbered for this entry?
            )

            if is_valid:
                if (entry_dir, exit_dir) not in rsa_movement_map:
                    rsa_movement_map[(entry_dir, exit_dir)] = movement_number
                    logger.debug(f"    Mapping: {entry_dir} -> {exit_dir} = #{movement_number}")
                    movement_number += 1
                    processed_exits_for_entry.add(exit_dir) # Mark as processed

    logger.debug(f"RSA Direction Map created with {len(rsa_movement_map)} entries: {rsa_movement_map}")
    return rsa_movement_map


# --- Main Report Function ---
def create_excel_report(completed_paths_data, start_datetime, primary_direction, video_path):
    """
    Creates a multi-sheet Excel report: Vehicle Logs (1 row/path), Intervals (24hr),
    Directional Counts (multi-header), RSA Output.
    """
    print(f"\n--- create_excel_report (Complete Code - Final Structure) ---")
    if not completed_paths_data:
         print("WARNING: No completed paths data received!")
         return None
    print(f"Received {len(completed_paths_data)} completed path summaries.")

    all_valid_vehicle_types = list(VEHICLE_ID_MAP.keys())
    if not all_valid_vehicle_types:
        print("ERROR: VEHICLE_ID_MAP in config appears empty.")
        return None

    # --- 1. Prepare Base Data ---
    valid_path_ids = set()
    valid_paths_structured = [] # For directional counts and RSA sheet
    vehicle_log_summary_data = [] # For Vehicle Logs sheet

    active_directions_set = set()
    if completed_paths_data:
        for path in completed_paths_data:
            v_id = path.get('id'); entry_t = path.get('entry_time'); exit_t = path.get('exit_time'); status = path.get('status')
            entry_dir = path.get('entry_direction'); exit_dir = path.get('exit_direction'); t_in_int = path.get('time_in_intersection')
            is_valid_entry_dir = entry_dir and entry_dir not in ['UNKNOWN', None]; is_valid_exit_dir = exit_dir and exit_dir not in ['UNKNOWN', 'TIMEOUT', 'FORCED', None, 'IN_PROGRESS', 'N/A']
            is_valid_status = status == 'exited'; has_valid_times = entry_t and exit_t and exit_t >= entry_t

            if is_valid_status and is_valid_entry_dir and is_valid_exit_dir and has_valid_times and v_id is not None:
                v_type = path.get('type', 'UnkType')
                if v_type not in all_valid_vehicle_types: continue
                valid_path_ids.add(v_id)
                entry_dir_disp = get_display_direction(entry_dir); exit_dir_disp = get_display_direction(exit_dir)
                entry_time_str = entry_t.strftime('%H:%M:%S.%f')[:-3]; exit_time_str = exit_t.strftime('%H:%M:%S.%f')[:-3]
                # Data for Vehicle Logs sheet
                vehicle_log_summary_data.append({
                    'Vehicle Tracker ID': v_id, 'Object ID': VEHICLE_ID_MAP.get(v_type, ''), 'Object Name': v_type,
                    'Entry Direction': entry_dir_disp, 'Exit Direction': exit_dir_disp,
                    'Entry Time': entry_time_str, 'Exit Time': exit_time_str,
                    'Time in Intersection (s)': round(t_in_int, 2) if t_in_int is not None else 'N/A' })
                # Data for other sheets (use raw keys for direction map lookup)
                valid_paths_structured.append({
                     'vehicle_type': v_type, 'direction_from': entry_dir, 'direction_to': exit_dir,
                     'timestamp': exit_t })
                active_directions_set.add(entry_dir); active_directions_set.add(exit_dir)
    logger.debug(f"Processed {len(valid_path_ids)} valid completed paths.")
    active_directions = sorted(list(active_directions_set)) # Use only directions involved in valid paths
    logger.debug(f"Active directions derived from valid paths: {active_directions}")

    # --- Build RSA Direction Map ---
    rsa_direction_map = _build_rsa_direction_map(primary_direction, active_directions, VALID_MOVEMENTS)

    # --- 2. Create Intervals (Full 24 Hours) ---
    intervals = []; report_date = start_datetime.date(); day_start_dt = datetime.combine(report_date, time.min); day_end_dt = day_start_dt + timedelta(days=1); curr_interval_start = day_start_dt
    interval_bins = [day_start_dt]; interval_labels = []
    while curr_interval_start < day_end_dt:
        interval_end = curr_interval_start + timedelta(minutes=15); interval_labels.append(f"{curr_interval_start.strftime('%H:%M:%S')} - {interval_end.strftime('%H:%M:%S')}")
        interval_bins.append(interval_end); intervals.append({'Date': curr_interval_start.strftime('%m/%d/%Y'), 'Time Intervals': interval_labels[-1], 'Start Time': curr_interval_start.strftime('%H:%M:%S'), 'End Time': interval_end.strftime('%H:%M:%S')}); curr_interval_start += timedelta(minutes=15)

    # --- 3. Prepare DataFrames ---
    df_log_summary = pd.DataFrame(vehicle_log_summary_data); df_intervals = pd.DataFrame(intervals)
    if not valid_paths_structured:
        df_moves = pd.DataFrame(columns=['vehicle_type', 'direction_from', 'direction_to', 'timestamp', 'TIME'])
    else:
        # Create display names needed for pivot table columns
        for path in valid_paths_structured:
            path['direction_from_disp'] = get_display_direction(path['direction_from'])
            path['direction_to_disp'] = get_display_direction(path['direction_to'])
        df_moves = pd.DataFrame(valid_paths_structured); df_moves['timestamp'] = pd.to_datetime(df_moves['timestamp'])
        time_cat = pd.CategoricalDtype(categories=interval_labels, ordered=True); df_moves['TIME'] = pd.cut(df_moves['timestamp'], bins=interval_bins, labels=interval_labels, right=False, ordered=True).astype(time_cat)
        vehicle_cat = pd.CategoricalDtype(categories=all_valid_vehicle_types, ordered=False); df_moves['vehicle_type'] = df_moves['vehicle_type'].astype(vehicle_cat)

    # --- Create Excel File ---
    excel_created_path = None
    try:
        output_report_dir = REPORT_OUTPUT_DIR
        os.makedirs(output_report_dir, exist_ok=True)
        output_filename_base=f"detection_logs_{os.path.splitext(os.path.basename(video_path))[0]}"
        excel_full_path = os.path.join(output_report_dir, f"{output_filename_base}.xlsx")


        with pd.ExcelWriter(excel_full_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- Sheet: Vehicle Logs ---
            if not df_log_summary.empty:
                 log_cols = ['Vehicle Tracker ID', 'Object ID', 'Object Name', 'Entry Direction', 'Exit Direction', 'Entry Time', 'Exit Time', 'Time in Intersection (s)']
                 df_logs = df_log_summary[[col for col in log_cols if col in df_log_summary.columns]]
                 df_logs.to_excel(writer, sheet_name='Vehicle Logs', index=False)
                 logger.debug(f"Writing 'Vehicle Logs' sheet ({len(df_logs)} rows).")
            else: logger.debug("No data for 'Vehicle Logs' sheet.")

            # --- Sheet: Intervals ---
            if not df_intervals.empty:
                df_intervals[['Date', 'Time Intervals', 'Start Time', 'End Time']].to_excel(writer, sheet_name='Intervals', index=False)
                logger.debug(f"Writing 'Intervals' sheet ({len(df_intervals)} rows).")
            else: logger.debug("No data for 'Intervals' sheet.")

            # --- Sheets: Directional Counts ---
            if not df_moves.empty and not df_intervals.empty and active_directions:
                # Define formats
                header_format_main = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
                header_format_sub = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
                col_header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'rotation': 90, 'border': 1})
                time_col_format = workbook.add_format({'align': 'left'})
                data_format = workbook.add_format({'align': 'center'})

                for from_dir_key in active_directions:
                    from_dir_sheet_name = f"From {from_dir_key.capitalize()}"
                    from_dir_match_display = from_dir_key.capitalize()
                    worksheet = writer.book.add_worksheet(from_dir_sheet_name)
                    writer.sheets[from_dir_sheet_name] = worksheet
                    logger.debug(f"  Generating sheet: {from_dir_sheet_name}")

                    df_from = df_moves[df_moves['direction_from_disp'] == from_dir_match_display].copy()
                    possible_to_dirs_keys = sorted([d for d in active_directions if d != from_dir_key])
                    if not possible_to_dirs_keys: continue

                    # Pivot, Reindex, Calculate Totals
                    pivot_df = pd.pivot_table(df_from, index='TIME', columns=['direction_to_disp', 'vehicle_type'], aggfunc='size', fill_value=0, observed=False, dropna=False)
                    all_dest_display = [get_display_direction(d) for d in possible_to_dirs_keys]
                    multi_cols = pd.MultiIndex.from_product([all_dest_display, all_valid_vehicle_types], names=['direction_to', 'vehicle_type'])
                    pivot_df = pivot_df.reindex(columns=multi_cols, fill_value=0); pivot_df = pivot_df.reindex(interval_labels, fill_value=0)
                    for to_dir_disp in all_dest_display:
                        vehicle_cols_tuples = [(to_dir_disp, vtype) for vtype in all_valid_vehicle_types]; cols_to_sum = [col for col in vehicle_cols_tuples if col in pivot_df.columns]
                        if cols_to_sum: pivot_df[(to_dir_disp, 'TOTAL')] = pivot_df[cols_to_sum].sum(axis=1)
                        else: pivot_df[(to_dir_disp, 'TOTAL')] = 0
                    final_col_order = [];
                    for to_dir_disp in all_dest_display:
                        for vtype in all_valid_vehicle_types: final_col_order.append((to_dir_disp, vtype)); final_col_order.append((to_dir_disp, 'TOTAL'))
                    final_col_order = [col for col in final_col_order if col in pivot_df.columns];
                    if not final_col_order: continue
                    df_sheet = pivot_df[final_col_order].copy(); df_sheet.index.name = 'TIME'; df_sheet.reset_index(inplace=True)

                    # Write Headers Manually
                    num_vehicle_types = len(all_valid_vehicle_types); cols_per_dest = num_vehicle_types + 1; last_data_col_index = 0
                    if possible_to_dirs_keys: last_data_col_index = (cols_per_dest * len(possible_to_dirs_keys))
                    worksheet.merge_range(0, 0, 0, last_data_col_index, from_dir_sheet_name, header_format_main); worksheet.write(2, 0, "TIME", col_header_format)
                    current_col_idx = 1; header_col_tuples_written = []
                    for to_dir_key in possible_to_dirs_keys:
                        to_dir_display_hdr = f"To {to_dir_key.capitalize()}"; start_merge_col_idx = current_col_idx; num_cols_this_dest = 0
                        for vtype in all_valid_vehicle_types:
                            col_tuple = (get_display_direction(to_dir_key), vtype)
                            if col_tuple in final_col_order: worksheet.write(2, current_col_idx, vtype, col_header_format); header_col_tuples_written.append(col_tuple); current_col_idx += 1; num_cols_this_dest += 1
                        total_tuple = (get_display_direction(to_dir_key), 'TOTAL')
                        if total_tuple in final_col_order: worksheet.write(2, current_col_idx, "TOTAL", col_header_format); header_col_tuples_written.append(total_tuple); current_col_idx += 1; num_cols_this_dest += 1
                        if num_cols_this_dest > 0: worksheet.merge_range(1, start_merge_col_idx, 1, current_col_idx - 1, to_dir_display_hdr, header_format_sub)

                    # Write Data Manually with Explicit Casting
                    start_row = 3
                    for row_idx, time_label in enumerate(df_sheet['TIME']):
                        try: worksheet.write(start_row + row_idx, 0, str(time_label))
                        except Exception: worksheet.write(start_row + row_idx, 0, "Write Error")
                        for col_idx, col_tuple in enumerate(header_col_tuples_written):
                             try: value = df_sheet.loc[row_idx, col_tuple]; worksheet.write(start_row + row_idx, col_idx + 1, int(value))
                             except KeyError: worksheet.write(start_row + row_idx, col_idx + 1, 0)
                             except (ValueError, TypeError): worksheet.write(start_row + row_idx, col_idx + 1, 0)

                    # Apply Formatting
                    worksheet.set_column(0, 0, 18, time_col_format)
                    current_col_idx = 1
                    for col_tuple in header_col_tuples_written: col_width = 10 if col_tuple[1] == 'TOTAL' else 5; worksheet.set_column(current_col_idx, current_col_idx, col_width, data_format); current_col_idx += 1
                    logger.debug(f"  Writing '{from_dir_sheet_name}' sheet ({len(df_sheet)} rows).")

            # Condition checks
            elif df_moves.empty: logger.debug("Skipping directional sheets: No valid movements.")
            elif not intervals: logger.debug("Skipping directional sheets: Intervals list empty.")
            elif not active_directions: logger.debug("Skipping directional sheets: No active directions.")

            # --- Sheet: RSA Format Output ---
            logger.debug(f"Checking conditions for RSA sheet: Map size={len(rsa_direction_map)}, Valid paths count={len(valid_paths_structured)}")
            if rsa_direction_map and valid_paths_structured:
                logger.debug("Generating RSA Format Output sheet...")
                rsa_sheet_name = "RSA Output"
                rsa_worksheet = writer.book.add_worksheet(rsa_sheet_name)
                writer.sheets[rsa_sheet_name] = rsa_worksheet

                # Write Header
                L0_val_1 = len(rsa_direction_map); L0_val_2 = len(rsa_direction_map); L0_val_3 = len(active_directions)
                rsa_header = [
                    "H0,1,320,3,RSA Standard Format Version 3.20",
                    f"S0,ME055,ME055, Intersection Name Placeholder,-31.365299,29.572933", # Example
                    f"I0,00001,Emaan_Traffic_Capture_Application", "D0,M,L",
                    f"D1,{start_datetime.strftime('%y%m%d')},0000000,{start_datetime.strftime('%y%m%d')},235959999,{start_datetime.strftime('%y%m%d')},0000000",
                    f"L0,{L0_val_1},{L0_val_2},{L0_val_3}" ]
                for r, row_str in enumerate(rsa_header): rsa_worksheet.write_string(r, 0, row_str)

                # Prepare and Write Data Lines
                rsa_data_lines = []; valid_paths_structured.sort(key=lambda x: x['timestamp'])
                for path in valid_paths_structured:
                    entry_key = path['direction_from']; exit_key = path['direction_to']; ts_obj = path['timestamp']; v_type = path['vehicle_type']
                    dir_num = rsa_direction_map.get((entry_key, exit_key), 0)
                    base_id = VEHICLE_ID_MAP.get(v_type, '?')
                    v_code = f"{base_id},1" if not base_id.startswith('2') else base_id
                    ts_str = ts_obj.strftime('%y%m%d,%H%M%S%f')[:-3]
                    rsa_line = f"10,9,1,0,{ts_str},{dir_num},{dir_num},1,{v_code}"
                    rsa_data_lines.append(rsa_line)
                start_data_row = len(rsa_header)
                for r, row_str in enumerate(rsa_data_lines): rsa_worksheet.write_string(start_data_row + r, 0, row_str)
                rsa_worksheet.set_column(0, 0, 60); logger.debug(f"Writing 'RSA Output' sheet ({len(rsa_data_lines)} rows).")

            elif not rsa_direction_map: logger.debug("Skipping 'RSA Output' sheet: RSA Direction map is empty.")
            else: logger.debug("Skipping 'RSA Output' sheet: No valid paths processed.")

        print(f"Excel report generated: {excel_full_path}")
        excel_created_path = excel_full_path

    except Exception as e:
        print(f"Error creating Excel report: {e}")
        import traceback; traceback.print_exc()
        return None

    return excel_created_path

def create_bidirectional_link_count_report(completed_paths_data, start_datetime_video, video_path, primary_direction, secondary_direction, video_duration_formatted="24 hours", station_number_str=""):
    """
    Creates a separate Excel report for bi-directional link counts based on user-selected "To" directions.
    The report includes individual and cumulative counts for specific vehicle categories over a 24-hour period.
    """

    if not completed_paths_data:
        print("WARNING (Bi-directional Report): No completed paths data received. Report will be empty or not generated.")
        return None

    # This is the order they will appear under each direction in the Excel sheet
    NEW_REPORT_COLUMN_ORDER = list(BIDIRECTIONAL_REPORT_CATEGORIES.keys())

    # --- 2. Prepare 24-Hour Time Bins ---
    # Report starts from 00:00 of the video's start day
    report_start_date = start_datetime_video.date()
    day_start_dt = datetime.combine(report_start_date, time.min) # 00:00:00
    
    time_bins_24hr = []
    time_bin_labels_24hr = []
    current_interval_start_dt = day_start_dt
    for _ in range(96): # 24 hours * 4 intervals/hour
        interval_end_dt = current_interval_start_dt + timedelta(minutes=15)
        time_bins_24hr.append(current_interval_start_dt)
        # Label for the start of the interval, as in the template (e.g., "6:00")
        time_bin_labels_24hr.append(current_interval_start_dt.strftime('%H:%M'))
        current_interval_start_dt = interval_end_dt
    time_bins_24hr.append(current_interval_start_dt) # Add the final end time for pd.cut

    # --- 3. Filter and Categorize Data for the Two Selected "To" Directions ---
    processed_for_report = []
    for path in completed_paths_data:
        if path.get('status') != 'exited' or not path.get('exit_time'):
            continue

        exit_time_dt = path['exit_time']
        vehicle_type_orig = path.get('type')
        exit_direction_norm = path.get('exit_direction', '').lower()

        matched_to_direction_label = None
        if exit_direction_norm == primary_direction:
            matched_to_direction_label = 'Dir1' # Internal label
        elif exit_direction_norm == secondary_direction:
            matched_to_direction_label = 'Dir2' # Internal label
        
        if matched_to_direction_label:
            assigned_new_category = None
            for new_cat, orig_types_list in NEW_REPORT_CATEGORIES.items():
                if vehicle_type_orig in orig_types_list:
                    assigned_new_category = new_cat
                    break
            
            if assigned_new_category:
                processed_for_report.append({
                    'timestamp': exit_time_dt,
                    'to_direction_label': matched_to_direction_label,
                    'new_category': assigned_new_category
                })

    if not processed_for_report:
        print("WARNING (Bi-directional Report): No relevant vehicle movements found for the selected 'To' directions and categories.")
        # Still create an empty report structure if desired, or return None
        # For now, let's proceed to create the structure.
        
    df_processed = pd.DataFrame(processed_for_report)

    # --- 4. Create DataFrame with Time Bins and Pivot for Individual Counts ---
    if not df_processed.empty:
        # Bin timestamps into 15-minute intervals using the generated labels
        # We use the start times (time_bin_labels_24hr) as the labels for the bins
        df_processed['Time Bin Start'] = pd.cut(
            df_processed['timestamp'],
            bins=time_bins_24hr,
            labels=time_bin_labels_24hr, # Use the H:MM labels
            right=False, # [start, end) interval
            ordered=True
        )
        
        # Pivot table for individual counts
        individual_counts_pivot = pd.pivot_table(
            df_processed,
            index='Time Bin Start',
            columns=['to_direction_label', 'new_category'],
            aggfunc='size',
            fill_value=0
        )
    else: # Create an empty pivot table structure if no data
        multi_index_cols = pd.MultiIndex.from_product(
            [['Dir1', 'Dir2'], NEW_REPORT_COLUMN_ORDER],
            names=['to_direction_label', 'new_category']
        )
        empty_idx = pd.CategoricalIndex(time_bin_labels_24hr, categories=time_bin_labels_24hr, ordered=True, name='Time Bin Start')
        individual_counts_pivot = pd.DataFrame(0, index=empty_idx, columns=multi_index_cols)


    # Reindex to ensure all 96 time bins and all categories/directions are present
    # And ensure correct column order for Dir1 and Dir2 categories
    final_columns_ordered = []
    for d_label, d_name_user in [('Dir1', primary_direction), ('Dir2', secondary_direction)]:
        for cat in NEW_REPORT_COLUMN_ORDER:
            final_columns_ordered.append((d_label, cat))
    
    individual_counts_pivot = individual_counts_pivot.reindex(
        index=time_bin_labels_24hr, # Ensure all time bins are rows
        columns=pd.MultiIndex.from_tuples(final_columns_ordered, names=['to_direction_label', 'new_category']),
        fill_value=0
    )

    # Calculate Totals for Individual Counts sheet
    df_individual_with_totals = individual_counts_pivot.copy()
    totals_row = df_individual_with_totals.sum(axis=0)
    totals_row.name = "Total" # Name for the new row's index
    # df_individual_with_totals = df_individual_with_totals.append(totals_row) # Older pandas
    df_individual_with_totals.loc["Total"] = totals_row # Newer pandas


    # --- 5. Calculate Cumulative Counts ---
    # Cumulative sum is calculated on the pivot table *without* the 'Total' row
    df_cumulative_counts = individual_counts_pivot.cumsum(axis=0)


    # --- 6. Write to Excel using xlsxwriter ---
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_str = start_datetime_video.strftime("%Y%m%d_%H%M%S")
    # New filename format
    excel_filename = f"bidirectional_path_analysis_{video_basename}_{timestamp_str}.xlsx"
    excel_full_path = os.path.join(REPORT_OUTPUT_DIR, excel_filename)

    try:
        with pd.ExcelWriter(excel_full_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Common header data
            header_info_static = [
                (" Link Count Sheet ",), # A1, merged later
                ("Station Number", station_number_str), ("Road Number", ""), ("KM", ""), ("Region", ""),
                ("Location Description", ""),
                ("Direction 1", ''),
                ("Direction 2", ''),
                ("Date", start_datetime_video.strftime("%Y-%m-%d")),
                ("Day of week", start_datetime_video.strftime("%A")),
                ("Duration of count", video_duration_formatted),
                ("Supervisor", ""), ("Classification System", ""),
                ("", "", "", "", "", "Link layout and travel directions", "", ""), # Row for this text
                ("",) # Empty row
            ]
            # Column headers for the data part
            data_column_headers_L1 = ["Start Time"]
            data_column_headers_L2 = [""] # For "Start Time"
            
            # Build data column headers dynamically
            dir1_display_name = get_display_direction(primary_direction)
            dir2_display_name = get_display_direction(secondary_direction)

            for _ in NEW_REPORT_COLUMN_ORDER: data_column_headers_L1.append(f"To {dir1_display_name}")
            for cat_name in NEW_REPORT_COLUMN_ORDER: data_column_headers_L2.append(cat_name)
            
            for _ in NEW_REPORT_COLUMN_ORDER: data_column_headers_L1.append(f"To {dir2_display_name}")
            for cat_name in NEW_REPORT_COLUMN_ORDER: data_column_headers_L2.append(cat_name)
            
            num_data_cols = 1 + len(NEW_REPORT_COLUMN_ORDER) * 2 # StartTime + (Cats * 2 Dirs)

            # --- Helper to write a sheet ---
            def write_sheet_content(sheet_name_ws, df_to_write, include_total_row):
                ws = workbook.add_worksheet(sheet_name_ws)
                
                # Formatting
                bold_format = workbook.add_format({'bold': True})
                center_format = workbook.add_format({'align': 'center'})
                header_merge_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})

                # Write static header block
                current_row_excel = 0
                ws.merge_range(current_row_excel, 0, current_row_excel, num_data_cols -1 , header_info_static[0][0], header_merge_format) # Title
                current_row_excel += 1
                for i in range(1, len(header_info_static) -2): # Up to "Link layout"
                    ws.write(current_row_excel, 0, header_info_static[i][0], bold_format)
                    if len(header_info_static[i]) > 1 and header_info_static[i][1]:
                        ws.write(current_row_excel, 1, header_info_static[i][1])
                    current_row_excel += 1
                # "Link layout..." row - merge across several cells
                ws.merge_range(current_row_excel, 0, current_row_excel, num_data_cols -1, header_info_static[-2][5], center_format) # Link layout
                current_row_excel += 2 # Skip one empty row and move to data headers

                # Write data column headers (L1 - Merged Direction Names)
                header_row_l1_excel = current_row_excel
                ws.write(header_row_l1_excel, 0, data_column_headers_L1[0], bold_format) # "Start Time"
                ws.write(header_row_l1_excel + 1, 0, data_column_headers_L2[0], bold_format) # Empty under "Start Time"
                
                # "To Direction 1" merged header
                ws.merge_range(header_row_l1_excel, 1, header_row_l1_excel, len(NEW_REPORT_COLUMN_ORDER), 
                               f"To {dir1_display_name}", header_merge_format)
                # "To Direction 2" merged header
                ws.merge_range(header_row_l1_excel, 1 + len(NEW_REPORT_COLUMN_ORDER), 
                               header_row_l1_excel, len(NEW_REPORT_COLUMN_ORDER) * 2,
                               f"To {dir2_display_name}", header_merge_format)
                
                # Write data column headers (L2 - Vehicle Categories)
                header_row_l2_excel = current_row_excel + 1
                for col_idx, header_text in enumerate(data_column_headers_L2[1:], start=1):
                    ws.write(header_row_l2_excel, col_idx, header_text, bold_format)
                
                current_row_excel += 2 # Move to data start

                # Write data
                # df_to_write index is 'Time Bin Start' (HH:MM labels)
                # df_to_write columns are MultiIndex: (('Dir1'/'Dir2'), ('Light'/'Heavy'/...))
                for time_bin_label, row_data in df_to_write.iterrows():
                    if not include_total_row and time_bin_label == "Total": # Skip Total for cumulative
                        continue
                    ws.write(current_row_excel, 0, time_bin_label) # Time bin label (e.g., "06:00")
                    
                    col_offset = 1
                    # Dir1 data
                    for cat in NEW_REPORT_COLUMN_ORDER:
                        ws.write(current_row_excel, col_offset, row_data[('Dir1', cat)])
                        col_offset += 1
                    # Dir2 data
                    for cat in NEW_REPORT_COLUMN_ORDER:
                        ws.write(current_row_excel, col_offset, row_data[('Dir2', cat)])
                        col_offset += 1
                    current_row_excel += 1
                
                # Autofit column widths (basic)
                ws.set_column(0, 0, 12) # Start Time column
                for i in range(1, num_data_cols):
                    ws.set_column(i, i, 10) # Data columns

            # --- Sheet 1: Individual Counts ---
            write_sheet_content("Individual Counts", df_individual_with_totals, include_total_row=True)
            logger.debug("(Bi-directional Report) Writing 'Individual Counts' sheet.")

            # --- Sheet 2: Cumulative Counts ---
            write_sheet_content("Cumulative Counts", df_cumulative_counts, include_total_row=False)
            logger.debug("(Bi-directional Report) Writing 'Cumulative Counts' sheet.")

        print(f"Bi-directional Link Count report generated: {excel_full_path}")
        return excel_full_path

    except Exception as e:
        print(f"Error creating Bi-directional Link Count report: {e}")
        import traceback
        traceback.print_exc()
        return None