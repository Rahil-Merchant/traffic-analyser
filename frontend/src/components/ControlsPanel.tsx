import React, { useState } from 'react';

interface ControlsPanelProps {
  activeDirection: string | null;
  drawnZones: string[];
  primaryDirection: string | null;
  isLaunchable: boolean;
  roadWidthMeters: number | '';
  roadLengthMeters: number | '';
  lengthSkewFactor: number | '';
  widthSkewFactor: number | '';
  speedLimitKmh: number | '';
  nightHoursStart: string;
  nightHoursEnd: string;
  onSetDirection: (direction: string) => void;
  onFinishZone: () => void;
  onUndoPoint: () => void;
  onClearAllZones: () => void;
  onPrimaryDirectionChange: (direction: string) => void;
  onRoadWidthChange: (val: number | '') => void;
  onRoadLengthChange: (val: number | '') => void;
  onLengthSkewFactorChange: (val: number | '') => void;
  onWidthSkewFactorChange: (val: number | '') => void;
  onSpeedLimitChange: (val: number | '') => void;
  onNightHoursStartChange: (val: string) => void;
  onNightHoursEndChange: (val: string) => void;
  onLaunch: () => void;
  hasPreviousResults?: boolean;
  onReopenResults?: () => void;
  canRecalibrate?: boolean;
  onRecalibrate?: (width: number, length: number, lengthSkew?: number, widthSkew?: number) => Promise<void>;
}

const ControlsPanel: React.FC<ControlsPanelProps> = (props) => {
  const {
    activeDirection,
    drawnZones,
    primaryDirection,
    isLaunchable,
    roadWidthMeters,
    roadLengthMeters,
    lengthSkewFactor,
    widthSkewFactor,
    speedLimitKmh,
    nightHoursStart,
    nightHoursEnd,
    onSetDirection,
    onFinishZone,
    onUndoPoint,
    onClearAllZones,
    onPrimaryDirectionChange,
    onRoadWidthChange,
    onRoadLengthChange,
    onLengthSkewFactorChange,
    onWidthSkewFactorChange,
    onSpeedLimitChange,
    onNightHoursStartChange,
    onNightHoursEndChange,
    onLaunch,
    hasPreviousResults,
    onReopenResults,
    canRecalibrate,
    onRecalibrate
  } = props;

  const [isRecalibrating, setIsRecalibrating] = useState(false);

  const directions = [
    { key: 'north', label: 'North', icon: '↑', color: 'emerald' },
    { key: 'south', label: 'South', icon: '↓', color: 'amber' },
    { key: 'east', label: 'East', icon: '→', color: 'blue' },
    { key: 'west', label: 'West', icon: '←', color: 'red' }
  ];

  const colorClasses = {
    emerald: 'from-emerald-500 to-emerald-600 border-emerald-400/50 text-emerald-50',
    amber: 'from-amber-500 to-amber-600 border-amber-400/50 text-amber-50',
    blue: 'from-blue-500 to-blue-600 border-blue-400/50 text-blue-50',
    red: 'from-red-500 to-red-600 border-red-400/50 text-red-50'
  };

  return (
    <div className="h-full bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-sm border border-gray-700/50 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
      {/* Premium Header - Fixed */}
      <div className="bg-gradient-to-r from-purple-600/20 via-blue-600/20 to-purple-600/20 p-4 border-b border-gray-700/50 flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Analysis Controls</h2>
            <p className="text-gray-400 text-xs">Configure detection zones</p>
          </div>
        </div>
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        <div className="p-3 space-y-3">
          {/* Zone Drawing Section */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500/20 rounded-md flex items-center justify-center">
                <span className="text-blue-400 font-bold text-xs">1</span>
              </div>
              <p className="font-semibold text-gray-200 text-sm">Select Detection Zone</p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {directions.map(dir => {
                const dirLower = dir.key;
                const isDrawn = drawnZones.includes(dirLower);
                const isActive = activeDirection === dirLower;
                const colorClass = colorClasses[dir.color as keyof typeof colorClasses];

                return (
                  <button
                    key={dir.key}
                    onClick={() => onSetDirection(dirLower)}
                    className={`
                relative p-2 rounded-lg border-2 transition-all duration-300 transform hover:scale-105
                ${isActive
                        ? `bg-gradient-to-br ${colorClass} shadow-lg ring-2 ring-white/20`
                        : isDrawn
                          ? `bg-gradient-to-br ${colorClass} opacity-75 hover:opacity-100`
                          : 'bg-gray-800/50 border-gray-600/50 text-gray-300 hover:bg-gray-700/50 hover:border-gray-500/50'
                      }
              `}
                  >
                    <div className="flex flex-col items-center space-y-1">
                      <div className="text-base font-mono">{dir.icon}</div>
                      <div className="text-xs font-semibold">{dir.label}</div>
                      {isDrawn && !isActive && (
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border border-gray-900 flex items-center justify-center">
                          <svg className="w-2 h-2 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Polygon Editing Section */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-yellow-500/20 rounded-md flex items-center justify-center">
                <span className="text-yellow-400 font-bold text-xs">2</span>
              </div>
              <p className="font-semibold text-gray-200 text-sm">Edit Polygon</p>
            </div>

            <div className="space-y-2">
              <button
                onClick={onFinishZone}
                className="w-full p-2 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-md hover:from-green-500 hover:to-green-600 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-green-500/25 font-semibold flex items-center justify-center space-x-2 text-xs"
                disabled={!activeDirection}
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Finish Zone</span>
              </button>

              <button
                onClick={onUndoPoint}
                className="w-full p-2 bg-gradient-to-r from-yellow-600 to-yellow-700 text-white rounded-md hover:from-yellow-500 hover:to-yellow-600 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-yellow-500/25 font-semibold flex items-center justify-center space-x-2 text-xs"
                disabled={!activeDirection}
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                </svg>
                <span>Undo Point</span>
              </button>

              <button
                onClick={onClearAllZones}
                className="w-full p-2 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-md hover:from-red-500 hover:to-red-600 transition-all duration-200 shadow-lg hover:shadow-red-500/25 font-semibold flex items-center justify-center space-x-2 text-xs"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                <span>Clear All</span>
              </button>
            </div>
          </div>

          {/* Speed Calibration Section */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-orange-500/20 rounded-md flex items-center justify-center">
                <span className="text-orange-400 font-bold text-xs">🏎️</span>
              </div>
              <p className="font-semibold text-gray-200 text-sm">Speed Calibration</p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => onSetDirection('road')}
                className={`
                  col-span-2 relative p-2 rounded-lg border-2 transition-all duration-300 transform hover:scale-105 flex flex-col items-center justify-center space-y-1
                  ${activeDirection === 'road'
                    ? 'bg-gradient-to-br from-orange-500 to-orange-600 shadow-lg ring-2 ring-white/20 border-orange-400/50 text-white'
                    : drawnZones.includes('road')
                      ? 'bg-gradient-to-br from-orange-500/80 to-orange-600/80 hover:opacity-100 border-orange-400/50 text-white'
                      : 'bg-gray-800/50 border-gray-600/50 text-gray-300 hover:bg-gray-700/50 hover:border-gray-500/50'
                  }
                `}
              >
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8v8M20 8v8M4 16l16-8M4 8l16 8" />
                  </svg>
                  <span className="text-xs font-semibold">{drawnZones.includes('road') ? 'Road Corners Marked' : 'Mark Road Corners'}</span>
                </div>
                <span className="text-[10px] opacity-70">TL → TR → BR → BL (4 points)</span>
              </button>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Road Width (m)</label>
                <input
                  type="number"
                  min="0" step="0.1"
                  value={roadWidthMeters}
                  onChange={e => onRoadWidthChange(e.target.value ? parseFloat(e.target.value) : '')}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                  placeholder="e.g. 7.0"
                />
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Width Skew Factor</label>
                <input
                  type="number"
                  min="0" step="0.01"
                  value={widthSkewFactor}
                  onChange={e => onWidthSkewFactorChange(e.target.value ? parseFloat(e.target.value) : '')}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                  placeholder="e.g. 1.0"
                />
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Road Length (m)</label>
                <input
                  type="number"
                  min="0" step="0.1"
                  value={roadLengthMeters}
                  onChange={e => onRoadLengthChange(e.target.value ? parseFloat(e.target.value) : '')}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                  placeholder="e.g. 30.0"
                />
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Length Skew Factor</label>
                <input
                  type="number"
                  min="0" step="0.01"
                  value={lengthSkewFactor}
                  onChange={e => onLengthSkewFactorChange(e.target.value ? parseFloat(e.target.value) : '')}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                  placeholder="e.g. 1.0"
                />
              </div>

              <div className="col-span-2 bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Speed Limit (km/h)</label>
                <input
                  type="number"
                  min="0"
                  value={speedLimitKmh}
                  onChange={e => onSpeedLimitChange(e.target.value ? parseFloat(e.target.value) : '')}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                  placeholder="e.g. 100"
                />
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Night Start</label>
                <input
                  type="time"
                  value={nightHoursStart}
                  onChange={e => onNightHoursStartChange(e.target.value)}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                />
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Night End</label>
                <input
                  type="time"
                  value={nightHoursEnd}
                  onChange={e => onNightHoursEndChange(e.target.value)}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:ring-1 focus:ring-blue-500"
                />
              </div>

              {/* Recalibrate Speed Button */}
              {hasPreviousResults && canRecalibrate && onRecalibrate && (
                <button
                  onClick={async () => {
                    if (typeof roadWidthMeters === 'number' && typeof roadLengthMeters === 'number') {
                      setIsRecalibrating(true);
                      await onRecalibrate(
                        roadWidthMeters, 
                        roadLengthMeters, 
                        typeof lengthSkewFactor === 'number' ? lengthSkewFactor : undefined, 
                        typeof widthSkewFactor === 'number' ? widthSkewFactor : undefined
                      );
                      setIsRecalibrating(false);
                    }
                  }}
                  disabled={isRecalibrating || typeof roadWidthMeters !== 'number' || typeof roadLengthMeters !== 'number'}
                  className="col-span-2 p-2 bg-gradient-to-r from-orange-600 to-orange-700 text-white rounded-lg hover:from-orange-500 hover:to-orange-600 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-orange-500/25 font-semibold flex items-center justify-center space-x-2 text-xs"
                >
                  {isRecalibrating ? (
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                    </svg>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      <span>Recalibrate Speed</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Report Configuration */}
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-purple-500/20 rounded-md flex items-center justify-center">
                <span className="text-purple-400 font-bold text-xs">3</span>
              </div>
              <p className="font-semibold text-gray-200 text-sm">Configure Reports</p>
            </div>

            <div className="space-y-2">
              {/* Primary Direction */}
              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2">
                <label htmlFor="primary-direction-select" className="block text-xs font-medium text-gray-400 mb-1">
                  Primary Direction
                </label>
                <select
                  id="primary-direction-select"
                  value={primaryDirection || ''}
                  onChange={e => onPrimaryDirectionChange(e.target.value)}
                  className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-1.5 text-white text-xs focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={drawnZones.length === 0}
                >
                  <option value="" disabled>Select from zones...</option>
                  {drawnZones
                    .filter(zone => zone !== "road")
                    .map(zone => (
                      <option key={zone} value={zone}>
                        {zone.charAt(0).toUpperCase() + zone.slice(1)}
                      </option>
                    ))
                  }
                </select>
              </div>

              {/* Report Type Auto-inferred */}
            </div>
          </div>
        </div>
      </div>

      {/* Premium Launch Section - Fixed at Bottom */}
      <div className="bg-gradient-to-r from-gray-800/50 via-gray-700/50 to-gray-800/50 border-t border-gray-700/50 p-4 flex-shrink-0">
        <button
          onClick={onLaunch}
          disabled={!isLaunchable}
          className={`
            w-full p-3 rounded-xl font-bold text-base transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-blue-500/25
            ${isLaunchable
              ? 'bg-gradient-to-r from-blue-600 via-blue-700 to-blue-600 text-white shadow-2xl shadow-blue-500/25 hover:shadow-blue-500/40 hover:from-blue-500 hover:to-blue-500'
              : 'bg-gray-700 text-gray-400 cursor-not-allowed shadow-none'
            }
          `}
        >
          <div className="flex items-center justify-center space-x-2">
            {isLaunchable ? (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>LAUNCH ANALYSIS</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.314 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span>COMPLETE SETUP</span>
              </>
            )}
          </div>
        </button>

        {!isLaunchable && (
          <div className="mt-2 text-center">
            <p className="text-xs text-gray-500">
              {drawnZones.length === 0 && "Select files and draw zones"}
              {drawnZones.length > 0 && !primaryDirection && "Set primary direction"}
            </p>
          </div>
        )}

        {hasPreviousResults && onReopenResults && (
          <button
            onClick={onReopenResults}
            className="w-full mt-3 p-3 rounded-xl font-bold text-sm transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-green-500/25 bg-gray-800 border border-green-500/30 text-green-400 hover:bg-gray-700 shadow-lg shadow-green-500/10 flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>VIEW PREVIOUS RESULTS</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default ControlsPanel;
