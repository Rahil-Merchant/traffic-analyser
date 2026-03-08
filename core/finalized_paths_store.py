# project/core/finalized_paths_store.py
import sqlite3
import tempfile
import os
from datetime import datetime

class FinalizedPathsStore:
    def __init__(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.sqlite')
        os.close(self.db_fd)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS finalized_paths (
                id TEXT,
                type TEXT,
                entry_direction TEXT,
                exit_direction TEXT,
                entry_time TEXT,
                exit_time TEXT,
                time_in_intersection REAL,
                status TEXT,
                average_speed_kmh REAL
            )
        ''')
        self.conn.commit()

    def _dt_converter(self, dt):
        if isinstance(dt, datetime):
            return dt.isoformat()
        return dt

    def _dt_parser(self, dt_str):
        if not dt_str: return None
        try:
            return datetime.fromisoformat(dt_str)
        except Exception:
            return None

    def append(self, path_data):
        speed_val = path_data.get('average_speed_kmh')
        if speed_val is not None:
            try:
                speed_val = float(speed_val)
            except Exception:
                speed_val = None

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO finalized_paths 
            (id, type, entry_direction, exit_direction, entry_time, exit_time, time_in_intersection, status, average_speed_kmh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            path_data.get('id'),
            path_data.get('type'),
            path_data.get('entry_direction'),
            path_data.get('exit_direction'),
            self._dt_converter(path_data.get('entry_time')),
            self._dt_converter(path_data.get('exit_time')),
            path_data.get('time_in_intersection'),
            path_data.get('status'),
            speed_val
        ))
        self.conn.commit()

    def get_all(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM finalized_paths')
        rows = cursor.fetchall()
        results = []
        for row in rows:
            speed_val = row[8]
            if speed_val is not None:
                try:
                    speed_val = float(speed_val)
                except Exception:
                    speed_val = None
                    
            results.append({
                'id': row[0],
                'type': row[1],
                'entry_direction': row[2],
                'exit_direction': row[3],
                'entry_time': self._dt_parser(row[4]),
                'exit_time': self._dt_parser(row[5]),
                'time_in_intersection': row[6],
                'status': row[7],
                'average_speed_kmh': speed_val
            })
        return results

    def count_by_status(self, status):
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM finalized_paths WHERE status = ?', (status,))
        row = cursor.fetchone()
        return row[0] if row else 0

    def clear(self):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM finalized_paths')
        self.conn.commit()

    def close(self):
        self.conn.close()
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception:
            pass
