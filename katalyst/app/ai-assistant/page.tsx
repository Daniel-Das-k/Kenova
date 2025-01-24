'use client';

import Sources from '../components/ai-assistant/Sources';
import ChatSection from '../components/ai-assistant/ChatSection';
import PageTemplate from '../components/PageTemplate';
import { AIAssistantProvider } from '../contexts/AIAssistantContext';

export default function AIAssistantPage() {
  return (
    <PageTemplate>
      <AIAssistantProvider>
        <div className="px-4 pt-6">
          <div className="flex gap-6">
            {/* Left Column - Sources */}
            <div className="w-72">
              <Sources />
            </div>

            {/* Middle Column - Chat */}
            <div className="flex-1">
              <ChatSection />
            </div>
          </div>
        </div>
      </AIAssistantProvider>
    </PageTemplate>
  );
}
