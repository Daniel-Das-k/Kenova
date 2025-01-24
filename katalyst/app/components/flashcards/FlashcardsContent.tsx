'use client';

export default function FlashcardsContent() {
  return (
    <div className="border-2 border-black h-[calc(90vh-80px)]">
      <div className="p-6">
        <h1 className="text-xl font-bold mb-6">FLASHCARDS</h1>

        {/* Purple content blocks */}
        <div className="space-y-6">
          <div className="w-full h-48 bg-[#B980FF] rounded-sm"></div>
          <div className="w-full h-48 bg-[#B980FF] rounded-sm"></div>
        </div>
      </div>
    </div>
  );
}
