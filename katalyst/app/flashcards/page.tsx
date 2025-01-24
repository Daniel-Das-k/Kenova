'use client';

import Sources from '../components/flashcards/Sources';
import FlashcardsContent from '../components/flashcards/FlashcardsContent';
import PageTemplate from '../components/PageTemplate';

export default function FlashcardsPage() {
  return (
    <PageTemplate>
      <div className="px-4 pt-6">
        <div className="flex gap-6">
          {/* Left Column - Sources */}
          <div className="w-72">
            <Sources />
          </div>

          {/* Main Content - Flashcards */}
          <div className="flex-1">
            <FlashcardsContent />
          </div>
        </div>
      </div>
    </PageTemplate>
  );
}
