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
  analysisResults?: any;
  analysisStatistics?: any;
}

const SetupPanel: React.FC<SetupPanelProps> = ({
  files,
  onReorder,
  onClearAll,
  selectedTimestamp,
  onTimestampChange,
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
    <div className="h-full bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-sm border border-gray-700/50 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
      {/* Compact Header */}
      <div className="bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-blue-600/20 p-4 border-b border-gray-700/50 flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5a2 2 0 012-2h4a2 2 0 012 2v2H8V5z" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Project Setup</h2>
            <p className="text-gray-400 text-xs">Configure analysis parameters</p>
          </div>
        </div>
      </div>

      {/* Main Content - No Scrollbar */}
      <div className="flex-1 p-4 space-y-4 overflow-hidden">
        {/* File Processing Order */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <div className="w-5 h-5 bg-blue-500/20 rounded-md flex items-center justify-center">
                <span className="text-blue-400 font-bold text-xs">1</span>
              </div>
              <p className="font-semibold text-gray-200 text-sm">Processing Order</p>
            </div>
            <button 
              onClick={onClearAll} 
              className="text-xs text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:cursor-not-allowed bg-red-500/10 hover:bg-red-500/20 px-2 py-1 rounded-full border border-red-500/20 transition-all duration-200" 
              disabled={files.length === 0}
            >
              Clear
            </button>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 h-32 overflow-hidden">
            <div className="h-full overflow-y-auto p-2 space-y-1">
              <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                <SortableContext items={files} strategy={verticalListSortingStrategy}>
                  {files.map(item => <DraggableFileListItem key={item.id} item={item} />)}
                </SortableContext>
              </DndContext>
              
              {files.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-center text-gray-500 space-y-2">
                  <div className="w-12 h-12 bg-gray-800/50 rounded-lg flex items-center justify-center">
                    <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="font-medium text-sm">No files selected</p>
                    <p className="text-xs">Click canvas to add files</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Compact Timestamp Configuration */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 bg-purple-500/20 rounded-md flex items-center justify-center">
              <span className="text-purple-400 font-bold text-xs">2</span>
            </div>
            <p className="font-semibold text-gray-200 text-sm">Timestamp Configuration</p>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-3 space-y-3">
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Start Date</label>
              <DatePicker 
                selected={selectedTimestamp} 
                onChange={(date: Date | null) => date && onTimestampChange(date)} 
                dateFormat="MMMM d, yyyy" 
                className="w-full bg-gray-800/70 border border-gray-600/50 rounded-md p-2 text-sm text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-200" 
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Start Time</label>
              <CustomTimeInput date={selectedTimestamp} setDate={onTimestampChange} />
            </div>
          </div>
        </div>
      </div>
      
      {/* Premium Results Section */}
      <div className="bg-gradient-to-r from-gray-800/50 via-gray-700/50 to-gray-800/50 border-t border-gray-700/50 p-4 flex-shrink-0">
        <div className="flex items-center space-x-2 mb-3">
          <div className="w-5 h-5 bg-green-500/20 rounded-md flex items-center justify-center">
            <svg className="w-3 h-3 text-green-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          <h3 className="font-semibold text-gray-200 text-sm">Analysis Results</h3>
        </div>
        
        <div className="bg-gray-900/50 rounded-lg border border-gray-700/30 p-3">
          {analysisStatistics ? (
            // Show results when available
            <div className="space-y-3">
              {/* Statistics Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-2">
                  <div className="text-green-400 text-xs font-medium">Valid Paths</div>
                  <div className="text-white font-bold text-lg">{analysisStatistics.total_valid_paths?.toLocaleString() || '0'}</div>
                </div>
                
                <div className="bg-blue-500/20 border border-blue-500/30 rounded-lg p-2">
                  <div className="text-blue-400 text-xs font-medium">Processing FPS</div>
                  <div className="text-white font-bold text-lg">{analysisStatistics.processing_fps || '0'}</div>
                </div>
                
                <div className="bg-purple-500/20 border border-purple-500/30 rounded-lg p-2">
                  <div className="text-purple-400 text-xs font-medium">Time Taken</div>
                  <div className="text-white font-bold text-sm">
                    {analysisStatistics.processing_time_seconds 
                      ? `${Math.floor(analysisStatistics.processing_time_seconds / 60)}m ${Math.floor(analysisStatistics.processing_time_seconds % 60)}s`
                      : '0s'
                    }
                  </div>
                </div>
                
                <div className="bg-orange-500/20 border border-orange-500/30 rounded-lg p-2">
                  <div className="text-orange-400 text-xs font-medium">Frames</div>
                  <div className="text-white font-bold text-sm">{analysisStatistics.total_frames_processed?.toLocaleString() || '0'}</div>
                </div>
              </div>
              
              {/* Generated Files */}
              {analysisResults && (
                <div className="space-y-2">
                  <div className="text-gray-400 text-xs font-medium">Generated Files:</div>
                  
                  {analysisResults.excel_path && (
                    <div className="flex items-center space-x-2 bg-gray-800/50 rounded-md p-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-green-400 text-xs font-medium flex-1 truncate">
                        {analysisResults.excel_path.split(/[\\/]/).pop()}
                      </span>
                    </div>
                  )}
                  
                  {analysisResults.bidirectional_excel_path && (
                    <div className="flex items-center space-x-2 bg-gray-800/50 rounded-md p-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span className="text-blue-400 text-xs font-medium flex-1 truncate">
                        {analysisResults.bidirectional_excel_path.split(/[\\/]/).pop()}
                      </span>
                    </div>
                  )}
                  
                  {analysisResults.video_path && 
                   analysisResults.video_path !== "None" && 
                   !analysisResults.video_path.includes("ERROR") && (
                    <div className="flex items-center space-x-2 bg-gray-800/50 rounded-md p-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                      <span className="text-purple-400 text-xs font-medium flex-1 truncate">
                        {analysisResults.video_path.split(/[\\/]/).pop()}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            // Show placeholder when no results
            <div className="flex flex-col items-center justify-center text-center text-gray-500 space-y-2 py-4">
              <div className="w-8 h-8 bg-gray-700/50 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <p className="font-medium text-xs">No Results Yet</p>
                <p className="text-xs">Analysis outputs will appear here</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SetupPanel;