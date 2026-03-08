import React, { useState, useEffect } from 'react';
import { Stage, Layer, Image, Circle, Line, Text } from 'react-konva';
import Konva from 'konva';
import useImage from 'use-image';

// --- Type Definitions ---
type Point = [number, number];
interface Polygons { [key: string]: Point[]; }

// --- Helper Function ---
const calculatePolygonCenter = (points: Point[]): Point => {
  if (!points || points.length === 0) return [0, 0];
  let sumX = 0, sumY = 0;
  for (const point of points) {
    sumX += point[0];
    sumY += point[1];
  }
  return [sumX / points.length, sumY / points.length];
};

interface InteractiveCanvasProps {
  imageUrl: string | null;
  isLoading: boolean;
  polygons: Polygons;
  currentPoints: Point[];
  activeDirection: string | null;
  onCanvasClick: (x: number, y: number) => void;
  onEmptyCanvasClick: () => void;
}

const InteractiveCanvas: React.FC<InteractiveCanvasProps> = (props) => {
  const { imageUrl, isLoading, polygons, currentPoints, activeDirection, onCanvasClick, onEmptyCanvasClick } = props;
  const [image] = useImage(imageUrl || '', 'anonymous');
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateSize = () => {
      const container = document.getElementById('canvas-container');
      if (container) {
        setContainerSize({
          width: container.clientWidth,
          height: container.clientHeight
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // NO additional useEffect for image dimensions!

  const getImageDimensions = () => {
    if (!image || !containerSize.width || !containerSize.height) {
      return { width: containerSize.width || 800, height: containerSize.height || 600, scale: 1, offsetX: 0, offsetY: 0 };
    }

    const containerAspect = containerSize.width / containerSize.height;
    const imageAspect = image.width / image.height;

    let scale, width, height, offsetX = 0, offsetY = 0;

    if (imageAspect > containerAspect) {
      scale = containerSize.width / image.width;
      width = containerSize.width;
      height = image.height * scale;
      offsetY = (containerSize.height - height) / 2;
    } else {
      scale = containerSize.height / image.height;
      width = image.width * scale;
      height = containerSize.height;
      offsetX = (containerSize.width - width) / 2;
    }

    return { width, height, scale, offsetX, offsetY };
  };

  const { width: canvasWidth, height: canvasHeight, scale, offsetX, offsetY } = getImageDimensions();

  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (activeDirection && image) {
      const stage = e.target.getStage();
      if (!stage) return;
      const pos = stage.getPointerPosition();
      if (!pos) return;

      const originalX = (pos.x - offsetX) / scale;
      const originalY = (pos.y - offsetY) / scale;

      if (originalX >= 0 && originalX <= image.width && originalY >= 0 && originalY <= image.height) {
        onCanvasClick(originalX, originalY);
      }
    }
  };

  const directionColors = {
    north: '#10b981',
    south: '#f59e0b',
    east: '#3b82f6',
    west: '#ef4444',
    road: '#f97316'
  };

  return (
    <div
      id="canvas-container"
      className="relative w-full h-full bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-gray-700/50"
    >
      {/* Premium Loading State */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/95 backdrop-blur-sm z-10">
          <div className="text-center">
            <div className="relative w-16 h-16 mx-auto mb-4">
              <div className="absolute inset-0 border-4 border-blue-500/30 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-blue-500 rounded-full border-t-transparent animate-spin"></div>
            </div>
            <p className="text-xl font-medium text-blue-300 animate-pulse">Generating Preview...</p>
            <p className="text-sm text-gray-400 mt-2">Processing first frame</p>
          </div>
        </div>
      )}

      {/* Premium Empty State */}
      {!isLoading && !imageUrl && (
        <div
          onClick={onEmptyCanvasClick}
          className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer group transition-all duration-500 hover:bg-gray-800/30"
        >
          <div className="text-center p-12 rounded-3xl bg-gray-800/30 backdrop-blur-sm border border-gray-600/30 group-hover:border-blue-500/50 transition-all duration-300 group-hover:scale-105">
            <div className="relative mb-6">
              <div className="w-24 h-24 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-blue-500/25 transition-all duration-300">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
            </div>
            <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-blue-300 transition-colors duration-300">
              Select Video Files
            </h3>
            <p className="text-gray-400 mb-4 group-hover:text-gray-300 transition-colors duration-300">
              Click here to browse and select video files for analysis
            </p>
            <div className="flex items-center justify-center space-x-2 text-sm text-gray-500">
              <span className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>MP4</span>
              </span>
              <span className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>AVI</span>
              </span>
              <span className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span>DAV</span>
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Premium Canvas with Perfect Aspect Ratio */}
      {!isLoading && imageUrl && (
        <Stage
          width={containerSize.width}
          height={containerSize.height}
          onClick={handleStageClick}
          className={activeDirection ? 'cursor-crosshair' : 'cursor-default'}
        >
          <Layer>
            {image && (
              <Image
                image={image}
                width={canvasWidth}
                height={canvasHeight}
                x={offsetX}
                y={offsetY}
                shadowColor="rgba(0,0,0,0.3)"
                shadowBlur={10}
                shadowOffset={{ x: 0, y: 4 }}
              />
            )}

            {/* Drawn Polygons with Premium Styling */}
            {Object.entries(polygons).map(([direction, points]) => {
              const center = calculatePolygonCenter(points);
              const color = directionColors[direction.toLowerCase() as keyof typeof directionColors] || '#3b82f6';
              return (
                <React.Fragment key={direction}>
                  <Line
                    points={points.flat().map((coord, i) => i % 2 === 0 ? coord * scale + offsetX : coord * scale + offsetY)}
                    stroke={color}
                    strokeWidth={3}
                    closed={true}
                    fill={`${color}20`}
                    shadowColor={color}
                    shadowBlur={8}
                    shadowOpacity={0.3}
                    listening={false}
                  />
                  <Text
                    text={direction.toUpperCase()}
                    x={center[0] * scale + offsetX}
                    y={center[1] * scale + offsetY}
                    fontSize={16}
                    fontFamily="Inter, system-ui, sans-serif"
                    fill="white"
                    fontStyle="bold"
                    shadowColor="rgba(0,0,0,0.8)"
                    shadowBlur={4}
                    offsetX={direction.length * 4}
                    offsetY={8}
                    listening={false}
                  />
                </React.Fragment>
              );
            })}

            {/* Current Drawing Line with Premium Effect */}
            {currentPoints.length > 0 && (
              <>
                <Line
                  points={currentPoints.flat().map((coord, i) => i % 2 === 0 ? coord * scale + offsetX : coord * scale + offsetY)}
                  stroke="#fbbf24"
                  strokeWidth={4}
                  lineCap="round"
                  lineJoin="round"
                  shadowColor="#fbbf24"
                  shadowBlur={8}
                  shadowOpacity={0.6}
                />
                {currentPoints.map((point, i) => (
                  <Circle
                    key={i}
                    x={point[0] * scale + offsetX}
                    y={point[1] * scale + offsetY}
                    radius={6}
                    fill="#fbbf24"
                    stroke="white"
                    strokeWidth={2}
                    shadowColor="#fbbf24"
                    shadowBlur={6}
                    shadowOpacity={0.8}
                  />
                ))}
              </>
            )}
          </Layer>
        </Stage>
      )}

      {activeDirection && (
        <div className="absolute top-4 right-4 bg-yellow-500/90 backdrop-blur-sm text-black px-4 py-2 rounded-full font-bold text-sm border border-yellow-400 shadow-lg">
          Drawing: {activeDirection.toUpperCase()}
        </div>
      )}
    </div>
  );
};

export default InteractiveCanvas;