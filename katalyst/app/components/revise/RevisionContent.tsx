'use client';

export default function RevisionContent() {
  return (
    <div className="border-2 border-black h-[calc(90vh-80px)]">
      <div className="p-6">
        <h1 className="text-xl font-bold mb-6">REVISION</h1>

        {/* Blue rectangles */}
        <div className="space-y-6">
          <div className="w-full h-48 bg-[#01B0C7] rounded-sm"></div>
          <div className="w-full h-48 bg-[#01B0C7] rounded-sm"></div>
        </div>
      </div>
    </div>
  );
}
