'use client';

import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import ChatMessage from './ChatMessage';
import { useAIAssistant } from '../../contexts/AIAssistantContext';

export default function ChatSection() {
  const { messages, setMessages, sources, isProcessing, setIsProcessing } = useAIAssistant();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isProcessing) return;

    const userMessage = {
      id: Date.now(),
      content: inputValue,
      isUser: true
    };
    setMessages(prev => [...prev, userMessage]);

    // Add loading message
    const loadingMessageId = Date.now() + 1;
    const loadingMessage = {
      id: loadingMessageId,
      content: "Thinking...",
      isUser: false,
      isLoading: true
    };
    setMessages(prev => [...prev, loadingMessage]);
    setIsProcessing(true);
    setInputValue('');

    try {
      // Create FormData and append the message
      const formData = new FormData();
      formData.append('message', inputValue);

      // Append all source files
      sources.forEach(source => {
        formData.append('file', source.file);
      });

      // Send request to backend
      const response = await fetch('http://localhost:5002/process/qa', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process request');
      }

      const data = await response.json();

      // Update messages with response
      setMessages(prev => {
        const filteredMessages = prev.filter(msg => msg.id !== loadingMessageId);
        return [
          ...filteredMessages,
          {
            id: Date.now() + 2,
            content: data.response || "I apologize, but I couldn't process that request.",
            isUser: false
          }
        ];
      });

    } catch (error) {
      console.error('Error processing request:', error);
      setMessages(prev => {
        const filteredMessages = prev.filter(msg => msg.id !== loadingMessageId);
        return [
          ...filteredMessages,
          {
            id: Date.now() + 2,
            content: "I apologize, but there was an error processing your request. Please try again.",
            isUser: false
          }
        ];
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-2 border-black h-[calc(90vh-80px)] flex flex-col">
      <div className="p-6">
        <h1 className="text-xl font-bold mb-6">Chat</h1>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 space-y-4">
        <div className="flex flex-col space-y-4">
          {messages.map(message => (
            <ChatMessage
              key={message.id}
              content={message.content}
              isUser={message.isUser}
              isLoading={message.isLoading}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="p-6 border-t-2 border-black">
        <div className="flex gap-4">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 resize-none border-2 border-black p-3 h-[50px] focus:outline-none"
            disabled={isProcessing}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || isProcessing}
            className="px-6 border-2 border-black bg-white
                     transition-all duration-200 
                     hover:translate-x-[-2px] hover:translate-y-[-2px]
                     shadow-[4px_4px_0_0_#000] 
                     hover:shadow-[6px_6px_0_0_#000]
                     disabled:opacity-50 disabled:cursor-not-allowed
                     disabled:hover:translate-x-0 disabled:hover:translate-y-0
                     disabled:hover:shadow-[4px_4px_0_0_#000]"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
