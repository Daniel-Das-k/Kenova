'use client';

import React from 'react';
import Navbar from '../components/Navbar';
import { usePdfStore } from '../store/pdfStore';
import PDFViewer from '../components/PDFViewer';
import { FileText, X, Loader2 } from "lucide-react";
import { useRef } from "react";
import { useRouter } from "next/navigation";

export default function QuestionPaper() {
  // const [isNavbarBlurred, setIsNavbarBlurred] = React.useState(false);
  const router = useRouter();
  const { 
    pdfUrl, 
    pdfSources, 
    setPdfUrl, 
    addPdfSource, 
    removePdfSource,
    isGenerating,
    generatedPdf,
    setIsGenerating,
    setGeneratedPdf,
    reset 
  } = usePdfStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    // setIsNavbarBlurred(true);
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      const url = URL.createObjectURL(file);
      addPdfSource(url);
      if (!pdfUrl) {
        setPdfUrl(url);
      }
    } else {
      alert('Please upload a PDF file');
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleGenerate = async () => {
    if (!pdfUrl) return;

    setIsGenerating(true);
    try {
      const formData = new FormData();
      // Convert URL to File object
      const response = await fetch(pdfUrl);
      const blob = await response.blob();
      const file = new File([blob], 'document.pdf', { type: 'application/pdf' });
      formData.append('file', file);

      const serverResponse = await fetch('http://localhost:5002/generate', {
        method: 'POST',
        body: formData,
      });

      if (!serverResponse.ok) {
        throw new Error(`HTTP error! status: ${serverResponse.status}`);
      }

      const data = await serverResponse.json();
      let datas = data.final_output;
      let parsedData: any;

      console.log('Raw response:', datas);
      
      if (datas && typeof datas === 'string') {
        // Remove the markdown code block markers
        datas = datas.trim().slice(7, -3);
        console.log('Cleaned data:', datas);
      }

      try {
        parsedData = JSON.parse(datas);
        console.log('Successfully parsed JSON:', parsedData);
      } catch (parseError) {
        console.error('JSON Parsing Error:', parseError);
        throw new Error('Failed to parse the server response.');
      }

      // Validate the parsed data structure
      if (parsedData && 
          typeof parsedData === 'object' &&
          'title' in parsedData &&
          'sections' in parsedData &&
          Array.isArray(parsedData.sections) && 
          parsedData.sections.length > 0) {
        console.log('Valid question paper format:', parsedData);
        setGeneratedPdf(parsedData);
      } else {
        console.error('Invalid question paper format:', {
          hasData: !!parsedData,
          isObject: typeof parsedData === 'object',
          hasTitle: parsedData && 'title' in parsedData,
          hasSections: parsedData && 'sections' in parsedData,
          isSectionsArray: parsedData && 'sections' in parsedData && Array.isArray(parsedData.sections),
          sectionsLength: parsedData?.sections?.length
        });
        throw new Error('Invalid question paper format');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to generate question paper. Please try again.');
    } finally {
      setIsGenerating(false);
      // setIsNavbarBlurred(false);
    }
  };

  const handleStartAnswering = () => {
    router.push('/exam-mode');
  };

  return (
    <div className="min-h-screen bg-white">
      <Navbar/>
      <main className="pt-24 px-4 md:px-8">
        <div className="flex gap-6">
          {/* Left Section - Sources */}
          {!generatedPdf && (
            <div className="w-72">
              <div className="border-2 border-black h-[calc(90vh-80px)]">
                <div className="p-4">
                  <h2 className="text-lg font-bold mb-4">Sources</h2>
                  <div className="space-y-3">
                    {pdfSources.map((source, index) => (
                      <div 
                        key={index} 
                        className="flex items-center justify-between p-3 border-2 border-black cursor-pointer hover:bg-gray-50"
                        onClick={() => setPdfUrl(source)}
                      >
                        <span>PDF Source {index + 1}</span>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            removePdfSource(source);
                            if (pdfUrl === source) {
                              setPdfUrl(pdfSources[0] || null);
                            }
                          }}
                          className="hover:text-red-500"
                        >
                          <X size={18} />
                        </button>
                      </div>
                    ))}
                  </div>
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={handleFileSelect}
                    ref={fileInputRef}
                    className="hidden"
                  />
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="mt-4 w-full flex items-center justify-center gap-2 py-2 px-4 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
                  >
                    <FileText className="w-5 h-5" />
                    Add Source
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Right Section - Content */}
          <div className="flex-1">
            <div className="border-2 border-black h-[calc(90vh-80px)] bg-white">
              {generatedPdf ? (
                <div className="p-6 h-full overflow-auto">
                  <div className="prose max-w-none">
                    <h1 className="text-2xl font-bold text-center mb-6">{generatedPdf.title}</h1>
                    <div className="mb-8">
                      <h2 className="text-lg font-semibold mb-2">Instructions:</h2>
                      <ul className="list-disc pl-5">
                        {generatedPdf.instructions.map((instruction: string, index: number) => (
                          <li key={index} className="text-gray-700">{instruction}</li>
                        ))}
                      </ul>
                    </div>
                    {generatedPdf.sections.map((section: any, sectionIndex: number) => (
                      <div key={sectionIndex} className="mb-8">
                        <div className="flex justify-between items-center mb-4">
                          <h2 className="text-xl font-semibold">{section.section_title}</h2>
                          <span className="text-gray-600">Total Marks: {section.total_marks}</span>
                        </div>
                        <div className="space-y-6">
                          {section.questions.map((question: any, questionIndex: number) => (
                            <div key={questionIndex} className="border-2 border-black p-4 rounded">
                              <div className="flex">
                                <div className="mr-4 font-medium min-w-[2rem]">
                                  {question.question_number}.
                                </div>
                                <div className="flex-grow">
                                  <div className="flex justify-between items-start">
                                    <div className="flex-grow">
                                      <p className="text-gray-800 whitespace-pre-wrap">{question.question_text}</p>
                                      {question.options && (
                                        <div className="mt-4 space-y-2">
                                          {Object.entries(question.options).map(([key, value]) => (
                                            <div key={key} className="flex items-start">
                                              <span className="mr-2 font-medium">{key}.</span>
                                              <span>{value as string}</span>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                    <span className="text-gray-500 ml-4 whitespace-nowrap">[{question.marks} marks]</span>
                                  </div>
                                  {section.background && questionIndex === 0 && (
                                    <div className="mt-4 bg-gray-50 p-4 rounded-lg">
                                      <h3 className="font-medium mb-2">Case Study Background:</h3>
                                      <p className="text-gray-700">{section.background}</p>
                                      {section.problem_statement && (
                                        <div className="mt-4">
                                          <h4 className="font-medium mb-1">Problem Statement:</h4>
                                          <p className="text-gray-700">{section.problem_statement}</p>
                                        </div>
                                      )}
                                      {section.supporting_data && (
                                        <div className="mt-4">
                                          <h4 className="font-medium mb-1">Supporting Data:</h4>
                                          <ul className="list-disc pl-5 space-y-1">
                                            {section.supporting_data.map((data: string, index: number) => (
                                              <li key={index} className="text-gray-600">{data}</li>
                                            ))}
                                          </ul>
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-8">
                    <button
                      onClick={handleStartAnswering}
                      className="w-full py-2.5 px-4 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
                    >
                      Start Answering
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {pdfUrl ? (
                    <>
                      <PDFViewer pdfUrl={pdfUrl} />
                      <div className="p-4 border-t-2 border-black">
                        <button 
                          onClick={handleGenerate}
                          disabled={isGenerating}
                          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 className="h-5 w-5 animate-spin" />
                              Generating...
                            </>
                          ) : (
                            'Generate Question Paper'
                          )}
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-[590px]">
                      <p className="text-lg">Please upload a PDF file first</p>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Reset Button - Only show when question paper is generated */}
          {generatedPdf && (
            <div className="mt-4">
              <button
                onClick={reset}
                className="w-full py-2.5 px-4 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
              >
                Generate Another Question Paper
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
