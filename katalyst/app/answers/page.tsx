'use client';

import { usePdfStore } from '../store/pdfStore';

export default function Answers() {
  const { generatedPdf } = usePdfStore();

  if (!generatedPdf) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl">No question paper found. Please generate a question paper first.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white" style={{ fontFamily: 'var(--font-courier-prime)' }}>
      <main className="py-8">
        <div className="max-w-[1400px] mx-auto px-6">
          <div className="flex flex-col">
            {/* Header */}
            <div className="mb-12 text-center">
              <h1 className="text-3xl font-bold mb-4">{generatedPdf.title}</h1>
              <h2 className="text-xl font-semibold text-gray-700">Answer Key</h2>
            </div>

            {/* Sections */}
            <div className="space-y-12">
              {generatedPdf.sections.map((section: any, sectionIndex: number) => (
                <div key={sectionIndex} className="border-2 border-black p-6 bg-white">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-bold">{section.section_title}</h2>
                    <span className="text-gray-600">Total Marks: {section.total_marks}</span>
                  </div>

                  {section.background && (
                    <div className="mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200">
                      <h3 className="font-semibold mb-2">Background:</h3>
                      <p className="text-gray-700">{section.background}</p>
                      {section.supporting_data && (
                        <div className="mt-4">
                          <h4 className="font-medium mb-2">Supporting Data:</h4>
                          <ul className="list-disc pl-5">
                            {section.supporting_data.map((data: string, index: number) => (
                              <li key={index} className="text-gray-600">{data}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="space-y-8">
                    {section.questions.map((question: any, questionIndex: number) => (
                      <div key={questionIndex} className="border-b-2 border-gray-100 pb-8 last:border-b-0 last:pb-0">
                        <div className="flex gap-4">
                          <span className="font-medium">{question.question_number}.</span>
                          <div className="flex-grow">
                            <div className="flex justify-between items-start">
                              <p className="text-gray-800 mb-4 flex-grow">{question.question_text}</p>
                              <span className="text-gray-500 ml-4 whitespace-nowrap">[{question.marks} marks]</span>
                            </div>
                            <div className="mt-4">
                              <h3 className="font-semibold mb-2">Answer:</h3>
                              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                <p className="text-gray-700 whitespace-pre-wrap">{question.answer}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Back to Question Paper Button */}
            <div className="mt-8">
              <button
                onClick={() => window.location.href = '/question-paper'}
                className="w-full py-2.5 px-4 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
              >
                Back to Question Paper
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
