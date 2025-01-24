'use client';

interface NotesContent {
  Overview: string;
  "Key Concepts": string[];
  "Critical Points": string[];
  Application: string[];
  "Additional Insights": string[];
}

interface NotesDisplayProps {
  content: string;
}

export default function NotesDisplay({ content }: NotesDisplayProps) {
  console.log('NotesDisplay received content:', content);
  
  let parsedContent: NotesContent;
  
  try {
    // Try to parse the content
    parsedContent = JSON.parse(content);
    console.log('Successfully parsed content:', parsedContent);
    
    // Validate the structure
    if (!parsedContent.Overview) {
      console.error('Missing Overview in parsed content');
      throw new Error('Missing Overview');
    }

    // Ensure all arrays exist
    parsedContent["Key Concepts"] = parsedContent["Key Concepts"] || [];
    parsedContent["Critical Points"] = parsedContent["Critical Points"] || [];
    parsedContent.Application = parsedContent.Application || [];
    parsedContent["Additional Insights"] = parsedContent["Additional Insights"] || [];
    
  } catch (error) {
    console.error('Error parsing notes content:', error);
    console.log('Raw content that caused error:', content);
    
    // Return a more informative error display
    return (
      <div className="p-4 border-2 border-red-200 rounded-lg bg-red-50">
        <h3 className="text-red-600 font-bold mb-2">Error displaying notes</h3>
        <p className="text-red-500">There was an error processing the notes. Please try uploading the file again.</p>
        <p className="text-red-400 text-sm mt-2">Error details: {error.message}</p>
      </div>
    );
  }

  // Function to render a section if it has content
  const renderSection = (title: string, content: string | string[]) => {
    if (!content || (Array.isArray(content) && content.length === 0)) return null;
    
    return (
      <section>
        <h2 className="text-xl font-bold text-green-700 mb-3">{title}</h2>
        {typeof content === 'string' ? (
          <p className="text-gray-700">{content}</p>
        ) : (
          <ul className="list-disc pl-6 space-y-2">
            {content.map((item, index) => (
              <li key={index} className="text-gray-700">{item}</li>
            ))}
          </ul>
        )}
      </section>
    );
  };

  return (
    <div className="space-y-8 px-4">
      {renderSection("Overview", parsedContent.Overview)}
      {renderSection("Key Concepts", parsedContent["Key Concepts"])}
      {renderSection("Critical Points", parsedContent["Critical Points"])}
      {renderSection("Application", parsedContent.Application)}
      {renderSection("Additional Insights", parsedContent["Additional Insights"])}
    </div>
  );
}
