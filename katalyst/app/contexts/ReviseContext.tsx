'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

interface Source {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
}

interface ReviseContextType {
  sources: Source[];
  keypoints: string;
  isProcessing: boolean;
  addSource: (source: Source) => void;
  removeSource: (id: string) => void;
  setKeypoints: (keypoints: string) => void;
  setIsProcessing: (isProcessing: boolean) => void;
}

const ReviseContext = createContext<ReviseContextType | undefined>(undefined);

export function ReviseProvider({ children }: { children: ReactNode }) {
  const [sources, setSources] = useState<Source[]>([]);
  const [keypoints, setKeypoints] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);

  const addSource = (source: Source) => {
    setSources(prev => [...prev, source]);
  };

  const removeSource = (id: string) => {
    setSources(prev => prev.filter(source => source.id !== id));
  };

  return (
    <ReviseContext.Provider
      value={{
        sources,
        keypoints,
        isProcessing,
        addSource,
        removeSource,
        setKeypoints,
        setIsProcessing,
      }}
    >
      {children}
    </ReviseContext.Provider>
  );
}

export function useRevise() {
  const context = useContext(ReviseContext);
  if (context === undefined) {
    throw new Error('useRevise must be used within a ReviseProvider');
  }
  return context;
}
