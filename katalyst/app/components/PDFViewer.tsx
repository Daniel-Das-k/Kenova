'use client';

import React from 'react';

interface PDFViewerProps {
  pdfUrl: string;
}

export default function PDFViewer({ pdfUrl }: PDFViewerProps) {
  return (
    <div className="w-full h-[80vh] border-2 border-black">
      <iframe
        src={`${pdfUrl}#toolbar=0`}
        className="w-full h-full"
        style={{ backgroundColor: 'white' }}
      />
    </div>
  );
}
