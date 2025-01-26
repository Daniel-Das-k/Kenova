'use client';

import FileUploadGreen from '../components/smart-notes/FileUploadGreen';
import KeypointsDisplay from '../components/revise/KeypointsDisplay';
import PageTemplate from '../components/PageTemplate';
import { ReviseProvider, useRevise } from '../contexts/ReviseContext';

function ReviseContent() {
  const { sources, keypoints, isProcessing, addSource, removeSource, setKeypoints, setIsProcessing } = useRevise();

  const handleFilesSelected = async (files: File[]) => {
    if (sources.length > 0) {
      return; // Don't add more files if we already have one
    }
    
    const file = files[0]; // Only take the first file
    const source = {
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      file: file
    };
    addSource(source);

    // Process the file
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:5002/process/keypoints', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process file');
      }

      const data = await response.json();
      console.log('Raw backend response:', data);

      let datas = data.response;
      // Clean the string if needed
      if (datas && typeof datas === 'string') {
        // Remove the ```json prefix and ``` suffix if present
        if (datas.startsWith('```json')) {
          datas = datas.trim().slice(7, -3);
        }
      }

      console.log('Cleaned response:', datas);

      // The response is already in the correct format, just pass it through
      setKeypoints(datas);
    } catch (error) {
      console.error('Error processing file:', error);
      setKeypoints(JSON.stringify([{
        Concept: 'Error',
        response: ['Error processing file. Please try uploading the file again.']
      }]));
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="pt-20 px-4 md:px-8">
      <div className="flex gap-6">
        {/* Left Section - Sources */}
        <div className="w-72">
          <div className="border-2 border-black h-[calc(90vh-80px)]">
            <div className="p-4">
              <h2 className="text-lg font-bold mb-4">Sources</h2>
              <FileUploadGreen
                onFilesSelected={handleFilesSelected}
                onFileRemove={removeSource}
                files={sources}
                buttonText={sources.length === 0 ? "Add source" : "Source added"}
                acceptedFileTypes={['.pdf']}
                disabled={sources.length > 0}
              />
            </div>
          </div>
        </div>

        {/* Right Section - Revision */}
        <div className="flex-1">
          <div className="border-2 border-black h-[calc(90vh-80px)] bg-white">
            <div className="p-6 h-full flex flex-col">
              <h2 className="text-lg font-bold mb-4">Revision</h2>
              <div className="flex-1 overflow-y-auto">
                {isProcessing ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                  </div>
                ) : keypoints ? (
                  <div className="bg-white rounded-lg h-full">
                    <KeypointsDisplay content={keypoints} />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    Upload a file to generate revision keypoints
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RevisePage() {
  return (
    <PageTemplate>
      <ReviseProvider>
        <ReviseContent />
      </ReviseProvider>
    </PageTemplate>
  );
}
