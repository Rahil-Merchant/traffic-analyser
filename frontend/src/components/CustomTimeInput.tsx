import React from 'react';
import { TimePicker, ConfigProvider } from 'antd';
import dayjs from 'dayjs';
import customParseFormat from 'dayjs/plugin/customParseFormat';

dayjs.extend(customParseFormat);

interface CustomTimeInputProps {
  date: Date;
  setDate: (date: Date) => void;
}

const CustomTimeInput: React.FC<CustomTimeInputProps> = ({ date, setDate }) => {
  return (
    <div className="bg-gray-800/50 rounded-md border border-gray-600/30 p-2">
      <ConfigProvider
        theme={{
          token: {
            colorBgContainer: 'rgba(17, 24, 39, 0.5)',
            colorBorder: 'rgba(75, 85, 99, 0.5)',
            colorText: '#ffffff',
            colorTextPlaceholder: '#9ca3af',
            colorPrimary: '#3b82f6',
            colorBgElevated: 'rgba(17, 24, 39, 0.95)',
            borderRadius: 6,
            fontSize: 14,
            controlHeight: 32,
            fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
          },
          components: {
            DatePicker: {
              colorBgContainer: 'rgba(17, 24, 39, 0.5)',
              colorBorder: 'rgba(75, 85, 99, 0.5)',
              activeBorderColor: '#3b82f6',
              hoverBorderColor: 'rgba(59, 130, 246, 0.5)',
              colorText: '#ffffff',
              colorTextPlaceholder: '#9ca3af',
            },
          },
        }}
      >
        <div className="flex items-center justify-center space-x-1">
          <TimePicker
            value={dayjs(date)}
            onChange={(time) => {
              if (time) {
                const newDate = new Date(date);
                newDate.setHours(time.hour(), time.minute(), time.second(), time.millisecond());
                setDate(newDate);
              }
            }}
            format="HH:mm:ss.SSS"
            showNow={false}
            size="small"
            className="custom-time-picker"
            style={{
              width: '160px',
              height: '32px',
              fontFamily: 'ui-monospace, SFMono-Regular, monospace'
            }}
            popupClassName="custom-time-popup"
          />

          <button
            onClick={() => {
              const now = new Date();
              now.setFullYear(date.getFullYear(), date.getMonth(), date.getDate());
              setDate(now);
            }}
            className="px-2 py-1 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded text-xs font-medium hover:bg-blue-500/30 transition-all duration-200"
          >
            Now
          </button>
        </div>

        <style>{`
          .custom-time-picker {
            background: rgba(17, 24, 39, 0.5) !important;
            border: 1px solid rgba(75, 85, 99, 0.5) !important;
            border-radius: 6px !important;
          }
          
          .custom-time-picker .ant-picker-input > input {
            color: #ffffff !important;
            background: transparent !important;
            font-family: ui-monospace, SFMono-Regular, monospace !important;
            font-size: 14px !important;
            text-align: center !important;
          }
          
          .custom-time-picker:hover {
            border-color: rgba(59, 130, 246, 0.5) !important;
          }
          
          .custom-time-picker.ant-picker-focused {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.25) !important;
          }
          
          .custom-time-picker .ant-picker-suffix {
            color: #9ca3af !important;
          }
          
          .custom-time-popup .ant-picker-dropdown {
            background: rgba(17, 24, 39, 0.95) !important;
            border: 1px solid rgba(75, 85, 99, 0.3) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(8px) !important;
          }
          
          .custom-time-popup .ant-picker-time-panel {
            background: transparent !important;
          }
          
          .custom-time-popup .ant-picker-time-panel-column {
            background: rgba(31, 41, 55, 0.8) !important;
            border-right: 1px solid rgba(75, 85, 99, 0.3) !important;
          }
          
          .custom-time-popup .ant-picker-time-panel-column:last-child {
            border-right: none !important;
          }
          
          .custom-time-popup .ant-picker-time-panel-cell {
            color: #ffffff !important;
            font-family: ui-monospace, SFMono-Regular, monospace !important;
            font-size: 13px !important;
          }
          
          .custom-time-popup .ant-picker-time-panel-cell:hover {
            background: rgba(59, 130, 246, 0.2) !important;
          }
          
          .custom-time-popup .ant-picker-time-panel-cell-selected {
            background: rgba(59, 130, 246, 0.4) !important;
            color: #ffffff !important;
          }
          
          .custom-time-popup .ant-picker-time-panel-cell-disabled {
            color: #6b7280 !important;
          }

          .custom-time-popup .ant-picker-time-panel-cell-selected .ant-picker-time-panel-cell-inner {
            color: #ffffff !important;
            background: transparent !important;
            border: 0.5px solid rgba(170, 218, 184, 0.3) !important;
          }

        `}</style>
      </ConfigProvider>
    </div>
  );
};

export default CustomTimeInput;
