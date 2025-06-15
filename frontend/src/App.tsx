import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { arrayMove } from '@dnd-kit/sortable';
import { open } from '@tauri-apps/plugin-dialog';
import SetupPanel from './components/SetupPanel';
import ControlsPanel from './components/ControlsPanel';
import InteractiveCanvas from './components/InteractiveCanvas';
import ProcessingOverlay from './components/ProcessingOverlay';
import ResultsModal from './components/ResultsModal';

// --- Type Definitions ---
interface FileItem {
  id: string;
  name: string;
  path: string;
}
type Point = [number, number];
interface Polygons { [key: string]: Point[]; }

// --- Helper Functions ---
const getInitialTimestamp = (): Date => new Date();
const formatTimestampForBackend = (date: Date) => {
  const year = date.getFullYear().toString().slice(-2);
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  const milliseconds = date.getMilliseconds().toString().padStart(3, '0');
  return { date: `${year}${month}${day}`, time: `${hours}${minutes}${seconds}${milliseconds}` };
};

const API_BASE_URL = 'http://127.0.0.1:7862';

function App() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<Date>(getInitialTimestamp());
  const [statusText, setStatusText] = useState<string>("[READY] Select video files to begin analysis");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState<boolean>(false);
  const [polygons, setPolygons] = useState<Polygons>({});
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [activeDirection, setActiveDirection] = useState<string | null>(null);
  const [primaryDirection, setPrimaryDirection] = useState<string | null>(null);
  const [reportType, setReportType] = useState('Standard Report');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState("");
  const [processingEta, setProcessingEta] = useState("...");
  const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number }>({ width: 640, height: 480 });
  const [stationNumber, setStationNumber] = useState<string>('');

  // Results Modal State
  const [showResults, setShowResults] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [analysisStatistics, setAnalysisStatistics] = useState<any>(null);

  const webSocketRef = useRef<WebSocket | null>(null);
  const isLaunchable = files.length > 0 && Object.keys(polygons).length > 0 && !!primaryDirection && !isProcessing;

  const fetchPreview = async (path: string) => {
    setIsLoadingPreview(true);
    setStatusText(`[PROCESSING] Generating preview from first frame...`);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/video-preview`, { path });
      if (response.data.preview_image_url) {
        setPreviewUrl(response.data.preview_image_url);
        setStatusText(`[ACTIVE] Previewing '${path.split(/[\\/]/).pop()}' • Select zones to analyze`);

        // Get EXACT image dimensions when preview loads
        const img = new Image();
        img.onload = () => {
          const exactDimensions = {
            width: img.naturalWidth,  // Use naturalWidth for exact pixel size
            height: img.naturalHeight // Use naturalHeight for exact pixel size
          };
          setImageDimensions(exactDimensions);
          console.log(`Exact Image Dimensions: ${exactDimensions.width}x${exactDimensions.height} pixels`);
        };
        img.onerror = () => {
          console.error('Failed to load image for dimension calculation');
          // Fallback to default dimensions
          setImageDimensions({ width: 640, height: 480 });
        };
        img.src = response.data.preview_image_url;
      } else {
        setStatusText(`[ERROR] ${response.data.error || 'Failed to generate preview.'}`);
      }
    } catch (error: any) {
      setStatusText(`[ERROR] Preview generation failed: ${error.response?.data?.detail || error.message}`);
    }
    finally { setIsLoadingPreview(false); }
  };


  const connectToWebSocket = (jobId: string) => {
    const wsUrl = `ws://127.0.0.1:7862/ws/progress/${jobId}`;
    webSocketRef.current = new WebSocket(wsUrl);

    webSocketRef.current.onopen = () => { setProcessingStatus("Connected to analysis engine..."); };
    webSocketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.progress || progress);
      setProcessingStatus(data.status || processingStatus);
      setProcessingEta(data.eta || processingEta);

      // Update intermediate statistics if available
      if (data.intermediate_stats) {
        setAnalysisStatistics((prev: any) => ({
          ...prev,
          ...data.intermediate_stats
        }));
      }

      if (data.status === 'complete') {
        // Store final results and statistics
        setAnalysisResults(data.results || null);
        setAnalysisStatistics(data.statistics || null);
        setShowResults(true);

        setIsProcessing(false);
        setStatusText("[COMPLETE] Analysis finished • Click 'Continue' to proceed");
        webSocketRef.current?.close();
      } else if (data.status === 'error') {
        alert(`❌ Analysis Error\n\n${data.message}\n\nPlease check the logs and try again.`);
        setIsProcessing(false);
        setStatusText(`[ERROR] Analysis failed: ${data.message}`);
        webSocketRef.current?.close();
      }
    };
    webSocketRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      alert("⚠️ Connection Error\n\nLost connection to analysis engine.");
      setIsProcessing(false);
    };
    webSocketRef.current.onclose = () => { console.log("Analysis connection closed."); };
  };

  const handleLaunch = async () => {
    if (!isLaunchable) return;
    // Clear previous results
    setAnalysisResults(null);
    setAnalysisStatistics(null);
    setIsProcessing(true);
    setProgress(0);
    setProcessingEta("Calculating...");
    setProcessingStatus("Initializing analysis pipeline...");

    const { date, time } = formatTimestampForBackend(selectedTimestamp);

    const configPayload = {
      video_paths: files.map(f => f.path),
      start_date: date,
      start_time: time,
      station_number: stationNumber,
      polygons,
      primary_direction: primaryDirection || '',
      report_type: reportType,
      // Exact image dimensions used for polygon drawing
      preview_image_width: imageDimensions.width,
      preview_image_height: imageDimensions.height,
      // Additional metadata about the image
      image_metadata: {
        exact_width: imageDimensions.width,
        exact_height: imageDimensions.height,
        aspect_ratio: imageDimensions.width / imageDimensions.height,
        total_pixels: imageDimensions.width * imageDimensions.height
      }
    };

    console.log('Sending exact image dimensions:', {
      width: imageDimensions.width,
      height: imageDimensions.height,
      payload: configPayload
    });

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/process`, configPayload);
      if (response.data.status === 'processing_started') {
        const jobId = response.data.job_id;
        setProcessingStatus(`Analysis job '${jobId}' initiated • Connecting to progress stream...`);
        connectToWebSocket(jobId);
      } else {
        setStatusText(`[ERROR] Analysis initiation failed: ${response.data.error || 'Unknown error'}`);
        setIsProcessing(false);
      }
    } catch (error: any) {
      console.error("Error launching analysis:", error);
      setStatusText(`[ERROR] Failed to start analysis: ${error.response?.data?.detail || 'Check server connection'}`);
      setIsProcessing(false);
    }
  };


  const handleSelectFiles = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [{
          name: 'Video Files',
          extensions: ['mp4', 'avi', 'mkv', 'mov', 'dav', 'asf', 'wmv', 'flv', 'webm']
        }]
      });

      if (selected) {
        const paths = Array.isArray(selected) ? selected : [selected];
        const newFileItems: FileItem[] = paths.map(path => ({
          id: path + '-' + Math.random(),
          name: path.split(/[\\/]/).pop() || path,
          path: path
        }));
        setFiles(prev => [...prev, ...newFileItems]);
        setStatusText(`[SUCCESS] Added ${newFileItems.length} file(s) to analysis queue`);
      }
    } catch (error) {
      console.error('File selection cancelled', error);
    }
  };

  const handleReorder = (oldIndex: number, newIndex: number) => { setFiles(prev => arrayMove(prev, oldIndex, newIndex)); };
  const handleClearAllFiles = () => { setFiles([]); handleClearAllZones(); };
  const handleSetDirection = (direction: string) => { if (activeDirection) { handleFinishZone(); } setCurrentPoints([]); setActiveDirection(direction); setStatusText(`[DRAWING] Active zone: ${direction.toUpperCase()} • Click to place points`); };
  const handleCanvasClick = (x: number, y: number) => { if (activeDirection) { setCurrentPoints(prev => [...prev, [x, y]]); } };
  const handleFinishZone = () => { if (!activeDirection || currentPoints.length < 3) { setStatusText(`[WARNING] Zone must have at least 3 points to be valid`); return; } setPolygons(prev => ({ ...prev, [activeDirection]: currentPoints })); setCurrentPoints([]); setActiveDirection(null); setStatusText(`[SAVED] Zone ${activeDirection.toUpperCase()} defined successfully`); };
  const handleUndoPoint = () => { if (currentPoints.length > 0) { setCurrentPoints(prev => prev.slice(0, -1)); } };
  const handleClearAllZones = () => { setPolygons({}); setCurrentPoints([]); setActiveDirection(null); setPrimaryDirection(null); setReportType('Standard Report'); setStatusText("[CLEARED] All zones removed • Ready for new configuration"); };

  const handleCloseResults = () => {
    setShowResults(false);
    // setAnalysisResults(null);
    // setAnalysisStatistics(null);
  };

  useEffect(() => { if (files.length > 0 && files[0].path) { fetchPreview(files[0].path); } else { setPreviewUrl(null); setStatusText("[READY] Select video files to begin analysis"); } }, [files[0]?.path]);
  useEffect(() => { const zones = Object.keys(polygons); if (primaryDirection && !zones.includes(primaryDirection)) { setPrimaryDirection(null); } if (zones.length === 2) { setReportType('Standard + Bi-directional Report'); } else { setReportType('Standard Report'); } }, [polygons, primaryDirection]);
  useEffect(() => {
    axios.get(`${API_BASE_URL}/api/v1/health`)
      .then(() => console.log("✅ Backend Connection Established"))
      .catch(e => console.error("❌ Backend Connection Failed:", e.message));
  }, []);

  useEffect(() => {
    // Cleanup WebSocket on component unmount or page refresh
    const handleBeforeUnload = () => {
      if (webSocketRef.current) {
        webSocketRef.current.close();
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      if (webSocketRef.current) {
        webSocketRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-gray-100 font-inter overflow-hidden">
      {/* Premium Global Styles */}
      <style>{`
        .font-inter { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
        .react-datepicker-wrapper { width: 100%; }
        .react-datepicker { 
          background-color: #111827; 
          border: 1px solid #374151; 
          border-radius: 12px;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        .react-datepicker__header { 
          background-color: #1f2937; 
          border-bottom: 1px solid #374151;
          border-top-left-radius: 12px;
          border-top-right-radius: 12px;
        }
        .react-datepicker__current-month, 
        .react-datepicker-time__header, 
        .react-datepicker-day-name { color: #e5e7eb; font-weight: 600; }
        .react-datepicker__day { color: #d1d5db; border-radius: 6px; }
        .react-datepicker__day--disabled { color: #6b7280; }
        .react-datepicker__day:hover { background-color: #374151; }
        .react-datepicker__day--selected { background-color: #3b82f6; color: white; }
        .react-datepicker__time-container, 
        .react-datepicker__time-box { background-color: #111827; }
        .react-datepicker__time-list-item { color: #d1d5db !important; }
        .react-datepicker__time-list-item:hover { background-color: #374151 !important; }
      `}</style>

      {/* Premium Header */}
      <header className="relative bg-gradient-to-r from-gray-900/95 via-gray-800/95 to-gray-900/95 backdrop-blur-sm border-b border-gray-700/50 py-6 px-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text text-transparent">
                EMAAN Urban Reconnaissance
              </h1>
              <p className="text-gray-400 text-sm font-medium">Advanced Traffic Analysis Platform</p>
            </div>
          </div>
        </div>
      </header>

      {/* Premium Main Layout */}
      <main className="h-[calc(100vh-100px)] flex p-6 gap-6 relative">
        {isProcessing && <ProcessingOverlay progress={progress} status={processingStatus} eta={processingEta} />}

        {/* Left Panel - Enhanced */}
        <div className={`w-80 transition-all duration-300 ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}>
          <SetupPanel
            files={files}
            onReorder={handleReorder}
            onClearAll={handleClearAllFiles}
            selectedTimestamp={selectedTimestamp}
            onTimestampChange={setSelectedTimestamp}
            stationNumber={stationNumber}
            onStationNumberChange={setStationNumber}
            analysisResults={analysisResults}
            analysisStatistics={analysisStatistics}
          />
        </div>

        {/* Center Canvas - Full Size */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 relative">
            <InteractiveCanvas
              onEmptyCanvasClick={handleSelectFiles}
              imageUrl={previewUrl}
              isLoading={isLoadingPreview}
              polygons={polygons}
              currentPoints={currentPoints}
              activeDirection={activeDirection}
              onCanvasClick={handleCanvasClick}
            />
          </div>

          {/* Premium Status Bar */}
          <div className="mt-4 bg-gradient-to-r from-gray-900/90 via-gray-800/90 to-gray-900/90 backdrop-blur-sm border border-gray-700/50 rounded-xl p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-cyan-500 rounded-full animate-pulse"></div>
                <p className="font-mono text-sm text-cyan-300 font-medium">{statusText}</p>
              </div>
              <div className="flex items-center space-x-2 text-xs text-gray-400">
                <span>Files: {files.length}</span>
                <span>•</span>
                <span>Zones: {Object.keys(polygons).length}</span>
                <span>•</span>
                <span className={isLaunchable ? 'text-green-400' : 'text-gray-500'}>
                  {isLaunchable ? 'Ready' : 'Setup Required'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Enhanced */}
        <div className={`w-80 transition-all duration-300 ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}>
          <ControlsPanel
            activeDirection={activeDirection}
            drawnZones={Object.keys(polygons)}
            primaryDirection={primaryDirection}
            reportType={reportType}
            isLaunchable={isLaunchable}
            onSetDirection={handleSetDirection}
            onFinishZone={handleFinishZone}
            onUndoPoint={handleUndoPoint}
            onClearAllZones={handleClearAllZones}
            onPrimaryDirectionChange={setPrimaryDirection}
            onReportTypeChange={setReportType}
            onLaunch={handleLaunch}
          />
        </div>
      </main>

      {/* Results Modal */}
      <ResultsModal
        isOpen={showResults}
        onClose={handleCloseResults}
        statistics={analysisStatistics}
        results={analysisResults}
      />
    </div>
  );
}

export default App;