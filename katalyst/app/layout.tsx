import type { Metadata } from 'next';
import { Courier_Prime } from 'next/font/google';
import './globals.css';

const courierPrime = Courier_Prime({
  weight: ['400', '700'],
  subsets: ['latin'],
  variable: '--font-courier-prime',
});

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
    <html lang="en" className={courierPrime.variable}>
      <body className={`antialiased ${courierPrime.className}`}>
        {children}
      </body>
    </html>
  );
}
