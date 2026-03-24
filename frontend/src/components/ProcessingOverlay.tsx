import React from 'react';

interface ProcessingOverlayProps {
  progress: number;
  status: string;
  eta: string;
  onAbort?: () => void;
}

const ProcessingOverlay: React.FC<ProcessingOverlayProps> = ({ progress, status, eta, onAbort }) => {
  return (
    <div className="fixed inset-0 bg-gray-950/95 backdrop-blur-lg flex items-center justify-center z-50">
      <div className="bg-gradient-to-br from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-sm border border-gray-700/50 rounded-3xl shadow-2xl p-8 max-w-lg w-full mx-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
            <svg className="w-10 h-10 text-white animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Analysis in Progress</h2>
          <p className="text-gray-400">Processing video data and generating reports</p>
        </div>

        {/* Progress Section */}
        <div className="space-y-6">
          {/* Progress Bar */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-300">Progress</span>
              <span className="text-sm font-bold text-blue-400">{Math.round(progress)}%</span>
            </div>
            
            <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden border border-gray-700/50">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-blue-600 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
                style={{ width: `${Math.max(progress, 0)}%` }}
              >
                {/* Animated shimmer effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
              </div>
            </div>
          </div>

          {/* Status Information */}
          <div className="bg-gray-900/50 rounded-xl border border-gray-700/30 p-4 space-y-3">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-gray-300">Current Status</span>
            </div>
            <p className="text-white font-medium pl-6">{status || "Initializing..."}</p>
          </div>

          {/* ETA Information */}
          <div className="bg-gray-900/50 rounded-xl border border-gray-700/30 p-4 space-y-3">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-gray-300">Estimated Time</span>
            </div>
            <p className="text-white font-medium pl-6">{eta || "Calculating..."}</p>
          </div>

          {/* Processing Steps Indicator */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { label: "Load", active: progress >= 0 },
              { label: "Detect", active: progress >= 25 },
              { label: "Track", active: progress >= 50 },
              { label: "Report", active: progress >= 75 }
            ].map((step, index) => (
              <div key={index} className="text-center">
                <div className={`w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                  step.active 
                    ? 'bg-gradient-to-br from-green-500 to-green-600 text-white shadow-lg' 
                    : 'bg-gray-700 text-gray-400'
                }`}>
                  {index + 1}
                </div>
                <div className={`text-xs font-medium transition-colors duration-300 ${
                  step.active ? 'text-green-400' : 'text-gray-500'
                }`}>
                  {step.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Warning Message */}
        <div className="mt-6 bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <svg className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.314 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <div>
              <p className="text-yellow-400 font-medium text-sm">Processing Active</p>
              <p className="text-yellow-300/80 text-xs mt-1">
                Please do not close the application while analysis is running. 
                This may take several minutes depending on video size.
              </p>
            </div>
          </div>
        </div>

        {/* Abort Button */}
        {onAbort && (
          <div className="mt-4 flex justify-center">
            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to abort the analysis? This will stop all processing.')) {
                  onAbort();
                }
              }}
              className="px-6 py-2.5 bg-gradient-to-r from-red-700 to-red-800 text-white font-semibold rounded-xl hover:from-red-600 hover:to-red-700 transition-all duration-200 shadow-lg shadow-red-500/20 hover:shadow-red-500/40 border border-red-600/30 flex items-center space-x-2 text-sm"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              <span>Abort Analysis</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProcessingOverlay;
