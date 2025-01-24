'use client';

import { FileText, X } from 'lucide-react';
import { useRef } from 'react';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  onFileRemove: (id: string) => void;
  files: Array<{
    id: string;
    name: string;
    size: number;
  }>;
  buttonText?: string;
  acceptedFileTypes?: string[];
  disabled?: boolean;
}

export default function FileUploadRed({
  onFilesSelected,
  onFileRemove,
  files,
  buttonText = "Upload Files",
  acceptedFileTypes = ['.pdf', '.doc', '.docx', '.txt'],
  disabled = false
}: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    onFilesSelected(selectedFiles);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  return (
    <div className="w-full space-y-4">
      {/* Display uploaded files */}
      <div className="space-y-4">
        {files.map((file) => (
          <div
            key={file.id}
            className="relative group border-2 border-black p-3 flex items-center gap-3
                     transition-all duration-200 hover:bg-gray-50
                     hover:translate-x-[-2px] hover:translate-y-[-2px]
                     shadow-[2px_2px_0_0_#000] 
                     hover:shadow-[4px_4px_0_0_#000]"
          >
            <FileText className="w-8 h-8 text-red-500" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{file.name}</p>
              <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
            </div>
            <button
              onClick={() => onFileRemove(file.id)}
              className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5
                       hover:bg-red-50 rounded-full"
              aria-label="Remove file"
            >
              <X className="w-4 h-4 text-red-500" />
            </button>
          </div>
        ))}
      </div>

      {/* Upload button */}
      <div>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
          multiple
          accept={acceptedFileTypes.join(',')}
          disabled={disabled}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className={`w-full py-2 px-4 border-2 border-black bg-white 
                   transition-all duration-200 
                   ${!disabled ? 'hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0_0_#000] hover:bg-red-50' : ''}
                   shadow-[4px_4px_0_0_#000] 
                   flex items-center justify-center gap-2
                   ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <FileText className={`w-5 h-5 ${disabled ? 'text-gray-500' : 'text-red-500'}`} />
          {buttonText}
        </button>
      </div>
    </div>
  );
}
