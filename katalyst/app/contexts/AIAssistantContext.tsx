'use client';

import React, { createContext, useContext, useState, ReactNode } from 'react';

interface Source {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
}

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  isLoading?: boolean;
}

interface AIAssistantContextType {
  sources: Source[];
  messages: Message[];
  addSource: (source: Source) => void;
  removeSource: (id: string) => void;
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  isProcessing: boolean;
  setIsProcessing: (value: boolean) => void;
}

const AIAssistantContext = createContext<AIAssistantContextType | undefined>(undefined);

export function AIAssistantProvider({ children }: { children: ReactNode }) {
  const [sources, setSources] = useState<Source[]>([]);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: "Hello! How can I help you today?",
      isUser: false
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);

  const addSource = (source: Source) => {
    setSources(prev => [...prev, source]);
  };

  const removeSource = (id: string) => {
    setSources(prev => prev.filter(source => source.id !== id));
  };

  const addMessage = (message: Message) => {
    setMessages(prev => {
      // Remove any existing loading messages
      const filteredMessages = prev.filter(msg => !msg.isLoading);
      return [...filteredMessages, message];
    });
  };

  return (
    <AIAssistantContext.Provider 
      value={{ 
        sources, 
        messages, 
        addSource, 
        removeSource, 
        addMessage,
        setMessages,
        isProcessing,
        setIsProcessing
      }}
    >
      {children}
    </AIAssistantContext.Provider>
  );
}

export function useAIAssistant() {
  const context = useContext(AIAssistantContext);
  if (context === undefined) {
    throw new Error('useAIAssistant must be used within an AIAssistantProvider');
  }
  return context;
}
