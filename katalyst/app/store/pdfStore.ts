import { create } from 'zustand';

interface PdfStore {
  pdfUrl: string | null;
  pdfSources: string[];
  isGenerating: boolean;
  generatedPdf: any;
  examAnswers: { [key: string]: string } | null;
  setPdfUrl: (url: string | null) => void;
  addPdfSource: (url: string) => void;
  removePdfSource: (url: string) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  setGeneratedPdf: (pdf: any) => void;
  setExamAnswers: (answers: { [key: string]: string }) => void;
  clearGeneratedPdf: () => void;
  reset: () => void;
}

export const usePdfStore = create<PdfStore>((set) => ({
  pdfUrl: null,
  pdfSources: [],
  isGenerating: false,
  generatedPdf: null,
  examAnswers: null,
  setPdfUrl: (url) => set({ pdfUrl: url }),
  addPdfSource: (url) => set((state) => ({ pdfSources: [...state.pdfSources, url] })),
  removePdfSource: (url) => set((state) => ({ pdfSources: state.pdfSources.filter(source => source !== url) })),
  setIsGenerating: (isGenerating) => set({ isGenerating }),
  setGeneratedPdf: (pdf) => set({ generatedPdf: pdf }),
  setExamAnswers: (answers) => set({ examAnswers: answers }),
  clearGeneratedPdf: () => set({ generatedPdf: null, examAnswers: null }),
  reset: () => set({ pdfUrl: null, pdfSources: [], isGenerating: false, generatedPdf: null, examAnswers: null }),
}));
