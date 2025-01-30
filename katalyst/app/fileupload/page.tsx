'use client';

import React, { useState, useCallback } from 'react';
import Navbar from '../components/Navbar';
import PDFViewer from '../components/PDFViewer';
import { Upload, Loader2 } from 'lucide-react';

export default function FileUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);

  const validateFile = (file: File): boolean => {
    // Check file type
    if (file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return false;
    }
    
    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      setError('File size must be less than 10MB');
      return false;
    }

    setError(null);
    return true;
  };

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (!droppedFile) return;

    if (validateFile(droppedFile)) {
      setFile(droppedFile);
      setShowPreview(false); // Reset preview when new file is dropped
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    if (validateFile(selectedFile)) {
      setFile(selectedFile);
      setShowPreview(false); // Reset preview when new file is selected
    }
  };

  const simulateUploadProgress = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setUploadProgress(progress);
      if (progress >= 100) {
        clearInterval(interval);
        setIsUploading(false);
        setUploadProgress(0);
        setShowPreview(true); // Show preview when upload is complete
      }
    }, 200);
  };

  const handleUpload = async () => {
    if (!file) return;

    try {
      setIsUploading(true);
      setError(null);

      // Simulate file upload with progress
      simulateUploadProgress();

      // Here you would typically send the file to your server
      // const formData = new FormData();
      // formData.append('file', file);
      // const response = await fetch('/api/upload', {
      //   method: 'POST',
      //   body: formData,
      // });
      
      // if (!response.ok) throw new Error('Upload failed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setIsUploading(false);
      setShowPreview(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <Navbar isBlurred={!showPreview} />
      {!showPreview ? (
        <div className="h-[calc(100vh-5rem)] flex items-center justify-center px-4 md:px-8 pt-24">
          <div className="w-full max-w-2xl">
            <div className="border-2 border-black p-6 rounded-lg text-center">
              <h1 className="text-3xl font-bold mb-8 tracking-wider">FILE UPLOAD</h1>
              <div className="flex flex-col items-center gap-6">
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragEnter={handleDragEnter}
                  onDragLeave={handleDragLeave}
                  onClick={() => document.getElementById('fileInput')?.click()}
                  className={`
                    w-full aspect-[2/1] border-2 border-black border-dashed 
                    ${isDragging ? 'bg-[#ffc333]' : 'bg-[#FFB800]'}
                    flex flex-col items-center justify-center cursor-pointer 
                    hover:bg-[#ffc333] transition-colors p-8 relative
                    ${error ? 'border-red-500' : 'border-black'}
                  `}
                >
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="fileInput"
                  />
                  <Upload className="w-12 h-12 mb-4" />
                  <h2 className="text-2xl font-bold mb-2">Drop your Notes here!</h2>
                  <p className="text-sm">Drag & drop or tap to level up</p>
                  {file && (
                    <p className="mt-4 font-medium">Selected: {file.name}</p>
                  )}
                  {error && (
                    <p className="mt-2 text-red-500 text-sm">{error}</p>
                  )}
                  {isDragging && (
                    <div className="absolute inset-0 border-2 border-black border-dashed animate-pulse" />
                  )}
                </div>
                
                <button
                  onClick={handleUpload}
                  disabled={!file || isUploading}
                  className={`
                    w-full py-3 px-6 bg-white border-2 border-black text-lg font-medium
                    hover:translate-x-[2px] hover:translate-y-[2px] 
                    hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] 
                    active:translate-x-[4px] active:translate-y-[4px] 
                    active:shadow-none 
                    transition-all
                    disabled:opacity-50 disabled:cursor-not-allowed
                    relative
                  `}
                >
                  {isUploading ? (
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Uploading... {uploadProgress}%</span>
                    </div>
                  ) : (
                    'Upload'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <main className="pt-24 px-4 md:px-8">
          <div className="max-w-[1400px] mx-auto">
            <div className="border-2 border-black p-6 rounded-lg">
              {file && <PDFViewer pdfUrl={URL.createObjectURL(file)} />}
            </div>
          </div>
        </main>
      )}
    </div>
  );
}
