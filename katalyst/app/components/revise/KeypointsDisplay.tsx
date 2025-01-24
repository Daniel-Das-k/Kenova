'use client';

interface ConceptItem {
  Concept: string;
  response: string[];
}

interface KeypointsDisplayProps {
  content: string;
}

export default function KeypointsDisplay({ content }: KeypointsDisplayProps) {
  console.log('KeypointsDisplay received content:', content);
  
  let parsedContent: ConceptItem[];
  
  try {
    // Add square brackets to make it a valid JSON array if needed
    const jsonString = content.trim().startsWith('[') ? content : `[${content}]`;
    console.log('Processing JSON string:', jsonString);
    
    parsedContent = JSON.parse(jsonString);
    console.log('Successfully parsed content:', parsedContent);
    
    // Validate the structure
    if (!Array.isArray(parsedContent)) {
      console.error('Content is not an array');
      throw new Error('Invalid content structure - not an array');
    }

    // Validate each concept item
    parsedContent = parsedContent.map(item => ({
      Concept: item?.Concept || 'Unknown Concept',
      response: Array.isArray(item?.response) ? item.response : []
    }));
    
  } catch (error) {
    console.error('Error parsing keypoints content:', error);
    console.log('Raw content that caused error:', content);
    
    return (
      <div className="p-4 border-2 border-red-200 rounded-lg bg-red-50">
        <h3 className="text-red-600 font-bold mb-2">Error displaying keypoints</h3>
        <p className="text-red-500">There was an error processing the keypoints. Please try uploading the file again.</p>
        <p className="text-red-400 text-sm mt-2">Error details: {error.message}</p>
      </div>
    );
  }

  if (!parsedContent.length) {
    return (
      <div className="p-4 border-2 border-yellow-200 rounded-lg bg-yellow-50">
        <h3 className="text-yellow-600 font-bold mb-2">No concepts found</h3>
        <p className="text-yellow-700">No revision concepts were found in the uploaded file.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 px-4">
      {parsedContent.map((concept, index) => (
        <section key={index} className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
          <h2 className="text-xl font-bold text-green-700 mb-4">
            {concept.Concept}
          </h2>
          {Array.isArray(concept.response) && concept.response.length > 0 ? (
            <ul className="list-disc pl-6 space-y-3">
              {concept.response.map((point, pointIndex) => (
                <li key={pointIndex} className="text-gray-700">
                  {point}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-500 italic">No details available for this concept.</p>
          )}
        </section>
      ))}
    </div>
  );
}
