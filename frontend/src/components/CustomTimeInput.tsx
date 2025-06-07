import React, { useState, useEffect } from 'react';

interface CustomTimeInputProps {
  date: Date;
  setDate: (date: Date) => void;
}

const CustomTimeInput: React.FC<CustomTimeInputProps> = ({ date, setDate }) => {
  const [hours, setHours] = useState(date.getHours().toString().padStart(2, '0'));
  const [minutes, setMinutes] = useState(date.getMinutes().toString().padStart(2, '0'));
  const [seconds, setSeconds] = useState(date.getSeconds().toString().padStart(2, '0'));
  const [milliseconds, setMilliseconds] = useState(date.getMilliseconds().toString().padStart(3, '0'));

  useEffect(() => {
    setHours(date.getHours().toString().padStart(2, '0'));
    setMinutes(date.getMinutes().toString().padStart(2, '0'));
    setSeconds(date.getSeconds().toString().padStart(2, '0'));
    setMilliseconds(date.getMilliseconds().toString().padStart(3, '0'));
  }, [date]);

  const handleTimeChange = (part: 'h' | 'm' | 's' | 'ms', value: string) => {
    const numericValue = parseInt(value, 10);
    const newDate = new Date(date.getTime());

    if (part === 'h' && numericValue >= 0 && numericValue <= 23) {
      setHours(value);
      newDate.setHours(numericValue);
      setDate(newDate);
    } else if (part === 'm' && numericValue >= 0 && numericValue <= 59) {
      setMinutes(value);
      newDate.setMinutes(numericValue);
      setDate(newDate);
    } else if (part === 's' && numericValue >= 0 && numericValue <= 59) {
      setSeconds(value);
      newDate.setSeconds(numericValue);
      setDate(newDate);
    } else if (part === 'ms' && numericValue >= 0 && numericValue <= 999) {
      setMilliseconds(value);
      newDate.setMilliseconds(numericValue);
      setDate(newDate);
    }
  };
  
  const handleBlur = (part: 'h' | 'm' | 's' | 'ms') => {
    if (part === 'h') setHours(prev => prev.padStart(2, '0'));
    if (part === 'm') setMinutes(prev => prev.padStart(2, '0'));
    if (part === 's') setSeconds(prev => prev.padStart(2, '0'));
    if (part === 'ms') setMilliseconds(prev => prev.padStart(3, '0'));
  };

  return (
    <div className="bg-gray-800/50 rounded-md border border-gray-600/30 p-2">
      {/* Compact Time Input */}
      <div className="flex items-center justify-center space-x-1">
        <input 
          type="text" 
          value={hours} 
          onChange={e => handleTimeChange('h', e.target.value)} 
          onBlur={() => handleBlur('h')} 
          className="w-8 h-8 text-center bg-gray-900/50 border border-gray-600/50 rounded text-white font-mono text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500/25 transition-all duration-200 outline-none"
          maxLength={2}
          placeholder="00"
        />
        <span className="text-gray-400 font-mono">:</span>
        <input 
          type="text" 
          value={minutes} 
          onChange={e => handleTimeChange('m', e.target.value)} 
          onBlur={() => handleBlur('m')} 
          className="w-8 h-8 text-center bg-gray-900/50 border border-gray-600/50 rounded text-white font-mono text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500/25 transition-all duration-200 outline-none"
          maxLength={2}
          placeholder="00"
        />
        <span className="text-gray-400 font-mono">:</span>
        <input 
          type="text" 
          value={seconds} 
          onChange={e => handleTimeChange('s', e.target.value)} 
          onBlur={() => handleBlur('s')} 
          className="w-8 h-8 text-center bg-gray-900/50 border border-gray-600/50 rounded text-white font-mono text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500/25 transition-all duration-200 outline-none"
          maxLength={2}
          placeholder="00"
        />
        <span className="text-gray-400 font-mono">.</span>
        <input 
          type="text" 
          value={milliseconds} 
          onChange={e => handleTimeChange('ms', e.target.value)} 
          onBlur={() => handleBlur('ms')} 
          className="w-10 h-8 text-center bg-gray-900/50 border border-gray-600/50 rounded text-white font-mono text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500/25 transition-all duration-200 outline-none"
          maxLength={3}
          placeholder="000"
        />
      </div>

      {/* Quick Set Buttons */}
      <div className="flex justify-center space-x-2 mt-2">
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
        <button
          onClick={() => {
            const midnight = new Date(date);
            midnight.setHours(0, 0, 0, 0);
            setDate(midnight);
          }}
          className="px-2 py-1 bg-purple-500/20 border border-purple-500/30 text-purple-400 rounded text-xs font-medium hover:bg-purple-500/30 transition-all duration-200"
        >
          00:00
        </button>
      </div>
    </div>
  );
};

export default CustomTimeInput;
