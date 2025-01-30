'use client';
import React, { useState } from 'react';
import PageTemplate from '../components/PageTemplate';
import FileUploadGreen from '../components/smart-notes/FileUploadGreen';

interface Question {
  question: string;
  options: Record<string, string>;
  correct_answer?: string;
  explanation?: string;
  marks?: number;
}

interface Answer {
  questionId: number;
  answer: string;
  feedback?: string;
}

const QuizPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [selectedAnswer, setSelectedAnswer] = useState<string>('');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [feedback, setFeedback] = useState<string>('');
  const [sources, setSources] = useState<{ id: string; file: File }[]>([]);

  const handleFilesSelected = async (files: File[]) => {
    if (sources.length > 0) {
      return; // Don't add more files if we already have one
    }
    
    const file = files[0]; // Only take the first file
    if (!file) return;

    const source = {
      id: Math.random().toString(36).substr(2, 9),
      file: file
    };
    setSources([source]);
    setFile(file);
  };

  const handleRemoveSource = (id: string) => {
    setSources([]);
    setFile(null);
    setQuestions([]);
  };

  const handleGenerate = async () => {
    if (!file) {
      alert('Please select a PDF file first');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://127.0.0.1:5002/process/quiz', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      let datas = data.final_output;
      
      if (datas && typeof datas === 'string') {
        datas = datas.trim().slice(7, -3);
      }

      let parsedData;
      try {
        parsedData = JSON.parse(datas);
      } catch (parseError) {
        console.error('JSON Parsing Error:', parseError);
        throw new Error('Failed to parse the server response.');
      }

      if (Array.isArray(parsedData) && 
          parsedData.length > 0 && 
          parsedData.every((q: any) => 
            typeof q === 'object' &&
            'question' in q &&
            'options' in q &&
            typeof q.options === 'object'
          )) {
        setQuestions(parsedData);
      } else {
        throw new Error('Invalid quiz format');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to generate quiz. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnswerSelect = (answer: string) => {
    setSelectedAnswer(answer);
    setAnswers(prev => [
      ...prev.filter(a => a.questionId !== currentQuestionIndex),
      { questionId: currentQuestionIndex, answer }
    ]);
  };

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <PageTemplate>
      <div className="pt-24 px-4 md:px-8">
        <div className="flex gap-6">
          {/* Left Section - Sources */}
          <div className="w-72">
            <div className="border-2 border-black h-[calc(90vh-80px)]">
              <div className="p-4">
                <h2 className="text-lg font-bold mb-4">Sources</h2>
                <FileUploadGreen
                  onFilesSelected={handleFilesSelected}
                  onFileRemove={handleRemoveSource}
                  files={sources.map(source => ({
                    id: source.id,
                    name: source.file.name,
                    size: source.file.size,
                    type: source.file.type,
                    file: source.file
                  }))}
                  buttonText={sources.length === 0 ? "Add source" : "Source added"}
                  acceptedFileTypes={['.pdf']}
                  disabled={sources.length > 0}
                />
                {file && (
                  <button
                    onClick={handleGenerate}
                    className="mt-4 w-full bg-green-500 text-white py-2 px-4 rounded hover:bg-green-600 disabled:bg-gray-300"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Generating...' : 'Generate'}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Right Section - Quiz */}
          <div className="flex-1">
            <div className="border-2 border-black h-[calc(90vh-80px)] bg-white">
              <div className="p-6 h-full flex flex-col">
                <h2 className="text-lg font-bold mb-4">Quiz</h2>
                <div className="flex-1 overflow-y-auto">
                  {isLoading ? (
                    <div className="flex items-center justify-center h-full">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                    </div>
                  ) : questions.length > 0 ? (
                    <div className="space-y-6">
                      <div className="font-medium">
                        Question {currentQuestionIndex + 1} of {questions.length}
                      </div>
                      <div className="text-lg">{currentQuestion.question}</div>
                      <div className="space-y-3">
                        {Object.entries(currentQuestion.options).map(([key, value]) => (
                          <button
                            key={key}
                            onClick={() => handleAnswerSelect(key)}
                            className={`w-full text-left p-3 rounded border ${selectedAnswer === key ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400'}`}
                          >
                            {value}
                          </button>
                        ))}
                      </div>
                      <div className="flex justify-between pt-4">
                        <button
                          onClick={() => setCurrentQuestionIndex(prev => Math.max(0, prev - 1))}
                          disabled={currentQuestionIndex === 0}
                          className="px-4 py-2 rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                        >
                          Previous
                        </button>
                        <button
                          onClick={() => setCurrentQuestionIndex(prev => Math.min(questions.length - 1, prev + 1))}
                          disabled={currentQuestionIndex === questions.length - 1}
                          className="px-4 py-2 rounded bg-green-500 text-white hover:bg-green-600 disabled:opacity-50"
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      Upload a PDF file to generate quiz questions
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </PageTemplate>
  );
};

export default QuizPage;
