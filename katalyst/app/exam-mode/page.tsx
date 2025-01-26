'use client';

import { useState } from 'react';
import { usePdfStore } from '../store/pdfStore';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function ExamMode() {
  const router = useRouter();
  const { generatedPdf, setExamAnswers } = usePdfStore();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<{ [key: string]: string }>({});

  if (!generatedPdf) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl">No question paper generated. Please generate a question paper first.</p>
      </div>
    );
  }

  // Flatten all questions from all sections into a single array
  const allQuestions = generatedPdf.sections.flatMap((section: any) =>
    section.questions.map((question: any) => ({
      ...question,
      sectionTitle: section.section_title,
      totalMarks: section.total_marks
    }))
  );

  const currentQuestion = allQuestions[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === allQuestions.length - 1;

  const handleNext = () => {
    if (currentQuestionIndex < allQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleAnswerChange = (answer: string) => {
    setAnswers({
      ...answers,
      [currentQuestionIndex]: answer
    });
  };

  const handleSubmit = () => {
    // Store answers in the PDF store
    setExamAnswers(answers);
    // Navigate to the evaluation page
    router.push('/evaluation');
  };

  return (
    <div className="min-h-screen bg-white" style={{ fontFamily: 'var(--font-courier-prime)' }}>
      <main className="h-screen">
        <div className="max-w-[1400px] mx-auto h-full p-6">
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="mb-8">
              <h1 className="text-2xl font-bold text-center">{generatedPdf.title}</h1>
              <div className="mt-4 flex justify-between items-center">
                <div>
                  <h2 className="font-semibold">{currentQuestion.sectionTitle}</h2>
                  <p className="text-sm text-gray-600">Question {currentQuestionIndex + 1} of {allQuestions.length}</p>
                </div>
                <div className="text-right">
                  <p className="font-medium">[{currentQuestion.marks} marks]</p>
                </div>
              </div>
            </div>

            {/* Question and Answer Section */}
            <div className="flex-grow flex flex-col">
              {/* Question */}
              <div className="mb-6">
                <div className="prose max-w-none">
                  <div className="flex gap-4">
                    <span className="font-medium">{currentQuestion.question_number}.</span>
                    <p className="text-gray-800">{currentQuestion.question_text}</p>
                  </div>
                </div>
              </div>

              {/* Answer Box */}
              <div className="flex-grow">
                <textarea
                  className="w-full h-full p-4 border-2 border-black resize-none focus:outline-none font-courier-prime"
                  placeholder="Write your answer here..."
                  value={answers[currentQuestionIndex] || ''}
                  onChange={(e) => handleAnswerChange(e.target.value)}
                />
              </div>

              {/* Navigation Buttons */}
              <div className="mt-6 flex justify-between gap-4">
                <button
                  onClick={handlePrevious}
                  disabled={currentQuestionIndex === 0}
                  className="flex items-center gap-2 py-2.5 px-6 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft size={20} />
                  Previous
                </button>
                {isLastQuestion ? (
                  <button
                    onClick={handleSubmit}
                    className="flex items-center gap-2 py-2.5 px-6 bg-black text-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
                  >
                    Submit
                  </button>
                ) : (
                  <button
                    onClick={handleNext}
                    className="flex items-center gap-2 py-2.5 px-6 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
                  >
                    Next
                    <ChevronRight size={20} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
