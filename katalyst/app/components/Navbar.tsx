'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'AI Assistant', path: '/ai-assistant' },
    { name: 'Smart Notes', path: '/smart-notes' },
    { name: 'Flashcards', path: '/flashcards' },
    { name: 'Revise', path: '/revise' },
    { name: 'Exam Mode', path: '/exam-mode' }
  ];

  const getNavItemStyle = (path: string) => {
    if (pathname === path) {
      if (path === '/ai-assistant') {
        return 'bg-red-500 text-white';
      } else if (path === '/revise') {
        return 'bg-[#01B0C7] text-white font-medium';
      } else if (path === '/flashcards') {
        return 'bg-[#B980FF] text-white font-medium';
      }
      return 'bg-[#4CAF50] text-white';
    }
    return 'bg-white hover:bg-gray-50';
  };

  return (
    <div className="p-4">
      <div className="border-2 border-black">
        <nav className="flex justify-end">
          {navItems.map((item) => (
            <Link
              key={item.path}
              href={item.path}
              className={`px-8 py-3 text-sm border-l-2 border-black ${getNavItemStyle(item.path)}`}
            >
              {item.name}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}
