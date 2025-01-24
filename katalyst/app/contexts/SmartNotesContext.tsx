'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

interface Source {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
}

interface SmartNotesContextType {
  sources: Source[];
  notes: string;
  isProcessing: boolean;
  addSource: (source: Source) => void;
  removeSource: (id: string) => void;
  setNotes: (notes: string) => void;
  setIsProcessing: (isProcessing: boolean) => void;
}

const SmartNotesContext = createContext<SmartNotesContextType | undefined>(undefined);

export function SmartNotesProvider({ children }: { children: ReactNode }) {
  const [sources, setSources] = useState<Source[]>([]);
  const [notes, setNotes] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);

  const addSource = (source: Source) => {
    setSources(prev => [...prev, source]);
  };

  const removeSource = (id: string) => {
    setSources(prev => prev.filter(source => source.id !== id));
  };

  return (
    <SmartNotesContext.Provider
      value={{
        sources,
        notes,
        isProcessing,
        addSource,
        removeSource,
        setNotes,
        setIsProcessing,
      }}
    >
      {children}
    </SmartNotesContext.Provider>
  );
}

export function useSmartNotes() {
  const context = useContext(SmartNotesContext);
  if (context === undefined) {
    throw new Error('useSmartNotes must be used within a SmartNotesProvider');
  }
  return context;
}
