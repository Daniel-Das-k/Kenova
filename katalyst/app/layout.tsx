import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Katalyst - Smart Learning Platform',
  description: 'AI-powered learning platform for students',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
