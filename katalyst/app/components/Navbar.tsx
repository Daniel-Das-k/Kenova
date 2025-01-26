'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface NavbarProps {
  isBlurred?: boolean;
}

export default function Navbar({ isBlurred = false }: NavbarProps) {
  const pathname = usePathname();

  const navItems = [
    { label: 'Home', href: '/' },
    { label: 'Question Paper', href: '/question-paper' },
    { label: 'Quiz', href: '/quiz' },
    { label: 'Smart Notes', href: '/smart-notes' },
    { label: 'AI Assistant', href: '/ai-assistant' },
    { label: 'Revise', href: '/revise' },
  ];

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 bg-white border-b-2 border-t-2 border-black ${
      isBlurred ? 'blur-sm pointer-events-none cursor-not-allowed' : ''
    }`}>
      <div className="flex justify-between items-center">
        <Link href="/" className={`border-r-2 border-black px-6 py-4 ${isBlurred ? 'cursor-not-allowed' : ''}`}>
          <h1 className="text-xl font-bold" style={{ fontFamily: 'var(--font-courier-prime)' }}>
            EduSage
          </h1>
        </Link>
        <div className="flex flex-wrap">
          {navItems.map((item, index) => (
            <Link 
              key={item.href}
              href={item.href}
              className={`px-6 py-4 border-x-2 border-black ${index === 0 ? 'border-l-2' : ''} ${pathname === item.href ? 'bg-[#FFB800]' : 'hover:bg-[#FFB800]'} transition-colors`}
              style={{ fontFamily: 'var(--font-courier-prime)' }}
            >
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
