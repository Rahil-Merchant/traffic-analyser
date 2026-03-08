import React from 'react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import type { DragEndEvent } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import DraggableFileListItem from './DraggableFileListItem';
import CustomTimeInput from './CustomTimeInput';

interface FileItem {
  id: string;
  name: string;
}

interface SetupPanelProps {
  files: FileItem[];
  onReorder: (oldIndex: number, newIndex: number) => void;
  onClearAll: () => void;
  selectedTimestamp: Date;
  onTimestampChange: (date: Date) => void;
  stationNumber: string;
  onStationNumberChange: (stationNumber: string) => void;
  analysisResults?: any;
  analysisStatistics?: any;
}

const SetupPanel: React.FC<SetupPanelProps> = ({
  files,
  onReorder,
  onClearAll,
  selectedTimestamp,
  onTimestampChange,
  stationNumber,
  onStationNumberChange,
  analysisResults,
  analysisStatistics
}) => {
  const sensors = useSensors(useSensor(PointerSensor));

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (active && over && active.id !== over.id) {
      const oldIndex = files.findIndex(f => f.id === active.id);
      const newIndex = files.findIndex(f => f.id === over.id);
      onReorder(oldIndex, newIndex);
    }
  };

  return (
    <div className="h-full w-full bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-sm border border-gray-700/50 rounded-2xl shadow-2xl overflow-hidden">
      {/* Force grid to respect container width */}
      <div
        className="h-full grid grid-rows-[auto_1fr_auto] gap-0"
        style={{
          width: '100%',
          maxWidth: '100%',
          minWidth: 0,
          overflow: 'hidden'
        }}
      >

        {/* Header - Responsive */}
        <div className="bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20 border-b border-gray-700/50 flex-shrink-0 p-2 sm:p-3 lg:p-4">
          <div className="flex items-center space-x-2 lg:space-x-3">
            <div className="w-6 h-6 sm:w-7 sm:h-7 lg:w-8 lg:h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg flex-shrink-0">
              <svg className="w-3 h-3 sm:w-4 sm:h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5a2 2 0 012-2h4a2 2 0 012 2v2H8V5z" />
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <h2 className="text-sm sm:text-base lg:text-lg font-bold text-white truncate">Project Setup</h2>
              <p className="text-gray-400 text-xs sm:text-xs lg:text-sm hidden sm:block truncate">Configure analysis parameters</p>
            </div>
          </div>
        </div>

        {/* Main Content - Scrollable with Responsive Grid */}
        <div
          className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden"
          style={{
            width: '100%',
            maxWidth: '100%',
            minWidth: 0
          }}
        >
          <div className="p-2 sm:p-3 lg:p-4 space-y-3 sm:space-y-4">

            {/* Processing Order Section - Responsive */}
            <div className="space-y-2">
              <div className="flex justify-between items-center gap-2">
                <div className="flex items-center space-x-2 min-w-0 flex-1">
                  <div className="w-4 h-4 lg:w-5 lg:h-5 bg-blue-500/20 rounded-md flex items-center justify-center flex-shrink-0">
                    <span className="text-blue-400 font-bold text-xs">1</span>
                  </div>
                  <p className="font-semibold text-gray-200 text-xs sm:text-sm lg:text-base truncate">Processing Order</p>
                </div>
                <button
                  onClick={onClearAll}
                  className="text-xs text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:cursor-not-allowed bg-red-500/10 hover:bg-red-500/20 px-2 py-1 rounded-full border border-red-500/20 transition-all duration-200 flex-shrink-0"
                  disabled={files.length === 0}
                >
                  Clear
                </button>
              </div>

              {/* Dynamic File List Height based on screen size */}
              <div
                className="bg-gray-900/50 rounded-lg border border-gray-700/30 overflow-hidden"
                style={{ height: 'clamp(80px, 15vh, 140px)' }}
              >
                <div className="h-full overflow-y-auto overflow-x-hidden p-1.5 sm:p-2 space-y-1">
                  <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                    <SortableContext items={files} strategy={verticalListSortingStrategy}>
                      {files.map(item => <DraggableFileListItem key={item.id} item={item} />)}
                    </SortableContext>
                  </DndContext>

                  {files.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center text-gray-500 space-y-1 p-2">
                      <div className="w-6 h-6 sm:w-8 sm:h-8 lg:w-10 lg:h-10 bg-gray-800/50 rounded-lg flex items-center justify-center">
                        <svg className="w-3 h-3 sm:w-4 sm:h-4 lg:w-5 lg:h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                        </svg>
                      </div>
                      <div>
                        <p className="font-medium text-xs sm:text-sm">No files selected</p>
                        <p className="text-xs hidden sm:block">Click canvas to add files</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Analysis Parameters Section - Responsive */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 lg:w-5 lg:h-5 bg-purple-500/20 rounded-md flex items-center justify-center flex-shrink-0">
                  <span className="text-purple-400 font-bold text-xs">2</span>
                </div>
                <p className="font-semibold text-gray-200 text-xs sm:text-sm lg:text-base truncate">Analysis Parameters</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2 sm:p-3 lg:p-4">
                {/* Responsive Form Grid */}
                <div className="grid gap-3 sm:gap-4">

                  {/* Start Date */}
                  <div className="space-y-1">
                    <label className="block text-xs sm:text-sm font-medium text-gray-400">Start Date</label>
                    <DatePicker
                      selected={selectedTimestamp}
                      onChange={(date: Date | null) => date && onTimestampChange(date)}
                      dateFormat="MMMM d, yyyy"
                      className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-2 text-xs sm:text-sm text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-200"
                    />
                  </div>

                  {/* Start Time */}
                  <div className="space-y-1">
                    <label className="block text-xs sm:text-sm font-medium text-gray-400">Start Time</label>
                    <CustomTimeInput date={selectedTimestamp} setDate={onTimestampChange} />
                  </div>

                  {/* Station Number */}
                  <div className="space-y-1">
                    <label className="block text-xs sm:text-sm font-medium text-gray-400">
                      Station Number <span className="text-gray-500">(Optional)</span>
                    </label>
                    <input
                      type="text"
                      value={stationNumber}
                      onChange={(e) => onStationNumberChange(e.target.value)}
                      placeholder="Enter station number..."
                      className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-2 text-xs sm:text-sm text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-200 outline-none placeholder:text-gray-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section - Constrained to parent width */}
        <div
          className="bg-gradient-to-r from-gray-800/50 via-gray-700/50 to-gray-800/50 border-t border-gray-700/50 flex-shrink-0"
          style={{
            width: '100%',
            maxWidth: '100%',
            minWidth: 0,
            overflow: 'hidden'
          }}
        >
          <div
            className="p-2 sm:p-3 lg:p-4"
            style={{
              maxHeight: 'min(30vh, 200px)',
              width: '100%',
              maxWidth: '100%',
              overflow: 'hidden'
            }}
          >
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-4 h-4 lg:w-5 lg:h-5 bg-green-500/20 rounded-md flex items-center justify-center flex-shrink-0">
                <svg className="w-2.5 h-2.5 lg:w-3 lg:h-3 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-200 text-xs sm:text-sm lg:text-base truncate min-w-0 flex-1">Analysis Results</h3>
            </div>

            <div
              className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-2 sm:p-3"
              style={{
                width: '100%',
                maxWidth: '100%',
                overflow: 'hidden',
                boxSizing: 'border-box'
              }}
            >
              {analysisStatistics ? (
                <div
                  className="space-y-2 sm:space-y-3"
                  style={{
                    width: '100%',
                    maxWidth: '100%',
                    overflow: 'hidden'
                  }}
                >

                  {/* Responsive Statistics Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <div className="bg-green-500/20 border border-green-500/30 rounded-md p-1.5 sm:p-2 overflow-hidden">
                      <div className="text-green-400 text-xs font-medium truncate">Valid Paths</div>
                      <div className="text-white font-bold text-sm sm:text-base truncate">
                        {analysisStatistics.total_valid_paths?.toLocaleString() || '0'}
                      </div>
                    </div>

                    <div className="bg-blue-500/20 border border-blue-500/30 rounded-md p-1.5 sm:p-2 overflow-hidden">
                      <div className="text-blue-400 text-xs font-medium truncate">Processing FPS</div>
                      <div className="text-white font-bold text-sm sm:text-base truncate">
                        {analysisStatistics.processing_fps || '0'}
                      </div>
                    </div>

                    <div className="bg-purple-500/20 border border-purple-500/30 rounded-md p-1.5 sm:p-2 sm:col-span-2 lg:col-span-1 overflow-hidden">
                      <div className="text-purple-400 text-xs font-medium truncate">Time Taken</div>
                      <div className="text-white font-bold text-sm sm:text-base truncate">
                        {analysisStatistics.processing_time_seconds
                          ? `${Math.floor(analysisStatistics.processing_time_seconds / 60)}m ${Math.floor(analysisStatistics.processing_time_seconds % 60)}s`
                          : '0s'
                        }
                      </div>
                    </div>

                    <div className="bg-orange-500/20 border border-orange-500/30 rounded-md p-1.5 sm:p-2 hidden lg:block overflow-hidden">
                      <div className="text-orange-400 text-xs font-medium truncate">Frames</div>
                      <div className="text-white font-bold text-sm sm:text-base truncate">
                        {analysisStatistics.total_frames_processed?.toLocaleString() || '0'}
                      </div>
                    </div>
                  </div>

                  {/* Generated Files with EXTREME truncation */}
                  {analysisResults && (
                    <div
                      className="space-y-1 sm:space-y-2"
                      style={{
                        width: '100%',
                        maxWidth: '100%',
                        overflow: 'hidden'
                      }}
                    >
                      <div className="text-gray-400 text-xs font-medium">Generated Files:</div>

                      <div
                        className="space-y-1"
                        style={{
                          width: '100%',
                          maxWidth: '100%',
                          overflow: 'hidden'
                        }}
                      >
                        {analysisResults.excel_path && (
                          <div
                            className="flex items-center space-x-2 bg-gray-800/50 rounded-md p-1.5"
                            style={{
                              width: '100%',
                              maxWidth: '100%',
                              overflow: 'hidden'
                            }}
                          >
                            <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0"></div>
                            <div
                              className="text-green-400 text-xs font-medium truncate"
                              style={{
                                maxWidth: 'calc(100% - 12px)',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}
                              title={analysisResults.excel_path.split(/[\\/]/).pop()}
                            >
                              {analysisResults.excel_path.split(/[\\/]/).pop()}
                            </div>
                          </div>
                        )}

                        {analysisResults.bidirectional_excel_path && (
                          <div
                            className="flex items-center space-x-2 bg-gray-800/50 rounded-md p-1.5"
                            style={{
                              width: '100%',
                              maxWidth: '100%',
                              overflow: 'hidden'
                            }}
                          >
                            <div className="w-2 h-2 bg-blue-500 rounded-full flex-shrink-0"></div>
                            <div
                              className="text-blue-400 text-xs font-medium truncate"
                              style={{
                                maxWidth: 'calc(100% - 12px)',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}
                              title={analysisResults.bidirectional_excel_path.split(/[\\/]/).pop()}
                            >
                              {analysisResults.bidirectional_excel_path.split(/[\\/]/).pop()}
                            </div>
                          </div>
                        )}

                        {analysisResults.video_path &&
                          analysisResults.video_path !== "None" &&
                          !analysisResults.video_path.includes("ERROR") && (
                            <div
                              className="flex items-center space-x-2 bg-gray-800/50 rounded-md p-1.5"
                              style={{
                                width: '100%',
                                maxWidth: '100%',
                                overflow: 'hidden'
                              }}
                            >
                              <div className="w-2 h-2 bg-purple-500 rounded-full flex-shrink-0"></div>
                              <div
                                className="text-purple-400 text-xs font-medium truncate"
                                style={{
                                  maxWidth: 'calc(100% - 12px)',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap'
                                }}
                                title={analysisResults.video_path.split(/[\\/]/).pop()}
                              >
                                {analysisResults.video_path.split(/[\\/]/).pop()}
                              </div>
                            </div>
                          )}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                // Responsive placeholder
                <div className="flex flex-col items-center justify-center text-center text-gray-500 space-y-1 py-2 sm:py-4">
                  <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-700/50 rounded-lg flex items-center justify-center">
                    <svg className="w-3 h-3 sm:w-4 sm:h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium text-xs sm:text-sm">No Results Yet</p>
                    <p className="text-xs hidden sm:block">Analysis outputs will appear here</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

};

export default SetupPanel;
