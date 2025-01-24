import type { Metadata } from "next";
import localFont from 'next/font/local';
import "./globals.css";

const courierPrime = localFont({
  src: './fonts/CourierPrime-Regular.ttf',
  variable: '--font-courier-prime',
  display: 'swap',
});

export const metadata: Metadata = {
  title: "EduSage",
  description: "Your Digital Notebook",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${courierPrime.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
