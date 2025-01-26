'use client';

import { useState, useRef } from 'react';
import { FileText, X } from 'lucide-react';
import Navbar from './Navbar';

interface UploadedFile {
  name: string;
  size: number;
  type: string;
  lastModified: number;
}

export default function FileUpload() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    setIsUploading(true);
    const files = event.target.files;
    if (files) {
      const newFiles = Array.from(files).map(file => ({
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified
      }));
      setUploadedFiles(prev => [...prev, ...newFiles]);
      // Simulate upload completion after a short delay
      setTimeout(() => setIsUploading(false), 1000);
    }
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };

  return (
    <div className="min-h-screen bg-white">
      <Navbar isBlurred={isUploading} />
      <div className="space-y-4">
        {uploadedFiles.map((file, index) => (
          <div
            key={file.name + index}
            className="relative bg-white border-2 border-black p-4 flex items-center gap-3 group hover:bg-gray-50 transition-colors"
          >
            <FileText className="w-8 h-8 text-[#4CAF50]" />
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{file.name}</p>
              <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
            </div>
            <button
              onClick={() => removeFile(index)}
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
              aria-label="Remove file"
            >
              <X className="w-5 h-5 text-gray-500 hover:text-black" />
            </button>
          </div>
        ))}

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
          multiple
          accept=".pdf,.doc,.docx,.txt"
        />

        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full py-2 px-4 border-2 border-black bg-white hover:bg-gray-50 transition-colors flex items-center justify-center gap-2"
        >
          <span>+</span> Add source
        </button>
      </div>
    </div>
  );
}
