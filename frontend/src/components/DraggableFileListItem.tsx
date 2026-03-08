import React from 'react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

interface DraggableFileListItemProps {
  item: {
    id: string;
    name: string;
  };
}

const DraggableFileListItem: React.FC<DraggableFileListItemProps> = ({ item }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: item.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  // Extract file extension for icon
  const getFileExtension = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    return ext;
  };

  const getFileIcon = (extension: string): { icon: string; color: string } => {
    const iconMap: { [key: string]: { icon: string; color: string } } = {
      'mp4': { icon: '🎬', color: 'from-blue-500 to-blue-600' },
      'avi': { icon: '📹', color: 'from-green-500 to-green-600' },
      'mkv': { icon: '🎭', color: 'from-purple-500 to-purple-600' },
      'mov': { icon: '🎪', color: 'from-pink-500 to-pink-600' },
      'dav': { icon: '📷', color: 'from-orange-500 to-orange-600' },
      'wmv': { icon: '🎨', color: 'from-red-500 to-red-600' },
      'flv': { icon: '⚡', color: 'from-yellow-500 to-yellow-600' },
      'webm': { icon: '🌐', color: 'from-cyan-500 to-cyan-600' },
    };
    return iconMap[extension] || { icon: '📄', color: 'from-gray-500 to-gray-600' };
  };

  const extension = getFileExtension(item.name);
  const { icon, color } = getFileIcon(extension);

  return (
    <div 
      ref={setNodeRef} 
      style={style} 
      {...attributes} 
      {...listeners} 
      className={`
        group relative bg-gradient-to-r from-gray-800/60 via-gray-700/60 to-gray-800/60 
        backdrop-blur-sm border border-gray-600/30 rounded-xl p-3 
        hover:from-gray-700/80 hover:via-gray-600/80 hover:to-gray-700/80 
        hover:border-gray-500/50 hover:shadow-lg hover:shadow-black/20
        transition-all duration-300 transform hover:scale-[1.02]
        cursor-grab active:cursor-grabbing touch-none select-none
        ${isDragging ? 'opacity-50 scale-105 shadow-2xl ring-2 ring-blue-500/50' : ''}
      `}
    >
      {/* File Content */}
      <div className="flex items-center space-x-3">
        {/* File Type Icon */}
        <div className={`w-10 h-10 bg-gradient-to-br ${color} rounded-lg flex items-center justify-center text-lg shadow-lg group-hover:scale-110 transition-transform duration-200`}>
          {icon}
        </div>
        
        {/* File Info */}
        <div className="flex-1 min-w-0">
          <p className="text-white font-medium text-sm truncate group-hover:text-blue-200 transition-colors duration-200" title={item.name}>
            {item.name}
          </p>
          <div className="flex items-center space-x-2 mt-1">
            <span className="text-xs text-gray-400 uppercase font-semibold bg-gray-600/30 px-2 py-0.5 rounded-full">
              {extension || 'file'}
            </span>
            <div className="w-1 h-1 bg-gray-500 rounded-full"></div>
            <span className="text-xs text-gray-500">Video</span>
          </div>
        </div>
        
        {/* Drag Handle */}
        <div className="flex-shrink-0 p-1 rounded-lg bg-gray-600/20 group-hover:bg-gray-500/30 transition-colors duration-200">
          <svg 
            className="w-4 h-4 text-gray-400 group-hover:text-gray-300 transition-colors duration-200" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16M4 16h16" />
          </svg>
        </div>
      </div>

      {/* Drag Indicator Dots */}
      <div className="absolute left-2 top-1/2 transform -translate-y-1/2 flex flex-col space-y-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
        <div className="w-1 h-1 bg-gray-500 rounded-full"></div>
        <div className="w-1 h-1 bg-gray-500 rounded-full"></div>
        <div className="w-1 h-1 bg-gray-500 rounded-full"></div>
      </div>

      {/* Hover Glow Effect */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/0 via-blue-500/5 to-purple-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
    </div>
  );
};

export default DraggableFileListItem;
