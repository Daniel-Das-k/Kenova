'use client';

import { useAIAssistant } from '../../contexts/AIAssistantContext';
import FileUploadRed from './FileUploadRed';

export default function Sources() {
  const { sources, addSource, removeSource } = useAIAssistant();

  const handleFilesSelected = async (files: File[]) => {
    if (sources.length > 0) {
      return; // Don't add more files if we already have one
    }
    
    const file = files[0]; // Only take the first file
    const source = {
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      file: file
    };
    addSource(source);
  };

  return (
    <div className="border-2 border-black h-[calc(90vh-80px)]">
      <div className="p-4">
        <h2 className="text-lg font-bold mb-4">Sources</h2>
        
        {/* Display uploaded files */}
        <div className="space-y-4 mb-6">
          <FileUploadRed
            onFilesSelected={handleFilesSelected}
            onFileRemove={removeSource}
            files={sources}
            buttonText={sources.length === 0 ? "Add source" : "Source added"}
            acceptedFileTypes={['.pdf']}
            disabled={sources.length > 0}
          />
        </div>
      </div>
    </div>
  );
}
