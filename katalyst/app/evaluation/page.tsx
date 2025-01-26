'use client';

import { useState, useEffect } from 'react';
import { usePdfStore } from '../store/pdfStore';
import { useRouter } from 'next/navigation';

interface EvaluatedAnswer {
  question: string;
  userAnswer: string;
  feedback: string;
  marks?: number;
  totalMarks?: number;
}

export default function EvaluationPage() {
  const router = useRouter();
  const { generatedPdf, examAnswers } = usePdfStore();
  const [evaluatedAnswers, setEvaluatedAnswers] = useState<EvaluatedAnswer[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!generatedPdf || !examAnswers) {
      router.push('/');
      return;
    }

    const evaluateAnswers = async () => {
      try {
        const allQuestions = generatedPdf.sections.flatMap((section: any) =>
          section.questions.map((question: any) => ({
            ...question,
            sectionTitle: section.section_title,
          }))
        );

        const evaluationPromises = Object.entries(examAnswers).map(async ([index, answer]) => {
          const questionData = allQuestions[parseInt(index)];
          const response = await fetch('http://127.0.0.1:5002/submit_answer', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              question: questionData.question_text,
              answer: answer,
              context: generatedPdf.context || ''
            }),
          });

          if (!response.ok) {
            throw new Error('Failed to evaluate answer');
          }

          const data = await response.json();
          return {
            question: questionData.question_text,
            userAnswer: answer,
            feedback: data.response,
            marks: questionData.marks,
            totalMarks: questionData.total_marks
          };
        });

        const evaluatedResults = await Promise.all(evaluationPromises);
        setEvaluatedAnswers(evaluatedResults);
      } catch (error) {
        console.error('Error evaluating answers:', error);
      } finally {
        setIsLoading(false);
      }
    };

    evaluateAnswers();
  }, [generatedPdf, examAnswers, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl">Evaluating your answers...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#D7B6FF]" style={{ fontFamily: 'var(--font-courier-prime)' }}>
      <main className="py-8 px-4 md:px-8">
        <div className="max-w-[1400px] mx-auto">
          <h1 className="text-3xl font-bold text-center mb-8">Evaluation Results</h1>
          
          <div className="space-y-8 bg-white">
            {evaluatedAnswers.map((item, index) => (
              <div key={index} className="border-2 border-black p-6 space-y-4">
                <div className="flex justify-between items-start">
                  <h2 className="font-semibold">Question {index + 1}</h2>
                  <span className="text-sm font-medium">[{item.marks} marks]</span>
                </div>
                
                <div className="space-y-2">
                  
                  <p className="text-gray-800">{item.question}</p>
                </div>

                <div className="space-y-2">
                  <p className="font-medium font-semibold ">Your Answer:</p>
                  <p className="text-gray-800 whitespace-pre-wrap">{item.userAnswer}</p>
                </div>

                <div className="space-y-2 bg-gray-200 p-4 border-2 border-black">
                  <p className="font-medium font-semibold ">Feedback:</p>
                  <p className="text-gray-800 whitespace-pre-wrap">{item.feedback}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 flex justify-center">
            <button
              onClick={() => router.push('/')}
              className="py-2.5 px-6 bg-white border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all font-medium"
            >
              Back to Home
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
