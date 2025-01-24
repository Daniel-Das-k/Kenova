'use client';

import { useState } from 'react';
import FileUploadBlue from './FileUploadBlue';

interface Source {
  id: string;
  name: string;
  size: number;
  type: string;
}

export default function Sources() {
  const [sources, setSources] = useState<Source[]>([]);

  const handleFilesSelected = (files: File[]) => {
    const newSources = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type
    }));
    setSources(prev => [...prev, ...newSources]);
  };

  const handleRemoveSource = (id: string) => {
    setSources(prev => prev.filter(source => source.id !== id));
  };

  return (
    <div className="border-2 border-black h-[calc(90vh-80px)]">
      <div className="p-4">
        <h2 className="text-lg font-bold mb-4">Sources</h2>
        
        {/* Display uploaded files */}
        <div className="space-y-4 mb-6">
          <FileUploadBlue
            onFilesSelected={handleFilesSelected}
            onFileRemove={handleRemoveSource}
            files={sources}
            buttonText="Add source"
            acceptedFileTypes={['.pdf', '.doc', '.docx', '.txt']}
          />
        </div>
      </div>
    </div>
  );
}
