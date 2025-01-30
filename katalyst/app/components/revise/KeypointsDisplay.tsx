'use client';

interface KeyPoint {
  Concept: string;
  response: string[];
}

type KeyPointsContent = KeyPoint[];

interface KeypointsDisplayProps {
  content: string;
}

export default function KeypointsDisplay({ content }: KeypointsDisplayProps) {
  console.log('KeypointsDisplay received content:', content);
  
  let parsedContent: KeyPointsContent;
  
  try {
    // If content is already an object, use it directly; otherwise, parse it
    parsedContent = typeof content === 'string' ? JSON.parse(content) : content;
    console.log('Successfully parsed content:', parsedContent);
    
    if (!Array.isArray(parsedContent) || !parsedContent.every(item => 
      item && typeof item === 'object' && 'Concept' in item && 'response' in item
    )) {
      throw new Error('Invalid keypoints format');
    }
  } catch (error) {
    console.error('Error parsing question paper content:', error);
    console.log('Raw content that caused error:', content);
    
    return (
      <div className="p-4 border-2 border-red-200 rounded-lg bg-red-50">
        <h3 className="text-red-600 font-bold mb-2">Error displaying question paper</h3>
        <p className="text-red-500">There was an error processing the content. Please try uploading the file again.</p>
        <p className="text-red-400 text-sm mt-2">Error details: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {parsedContent.map((keypoint, index) => (
        <div key={index} className="mb-8 p-6 bg-white rounded-lg border-2 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
          <h2 className="text-xl font-bold mb-4 text-green-600">{keypoint.Concept}</h2>
          <ul className="space-y-3">
            {keypoint.response.map((point, pointIndex) => (
              <li key={pointIndex} className="flex items-start gap-3">
                <span className="text-green-500 mt-1">•</span>
                <p className="text-gray-700">{point}</p>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
