import React from 'react';

export default function Home() {
  return (
    <main className="min-h-screen p-16 md:p-24" style={{ fontFamily: 'var(--font-courier-prime)' }}>
      <h1 className="text-4xl font-bold mb-8">
        Welcome To <span className="text-[#FFB800]">Kenova</span>
      </h1>
      
      <div className="relative">
        <div className="flex items-center justify-between mb-6 border-b-2 border-black">
          <h2 className="text-xl font-medium">My Notebooks</h2>
          <button className="text-3xl font-bold pb-1">+</button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Notebook Cards */}
          <div className="w-full aspect-[4/3] bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] cursor-pointer hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all"></div>
          <div className="w-full aspect-[4/3] bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] cursor-pointer hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all"></div>
        </div>
      </div>
    </main>
  );
}
