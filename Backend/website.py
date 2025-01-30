import os
import requests
from bs4 import BeautifulSoup
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WebsiteAnalyzer:
    def __init__(self, model_name='gemini-2.0-flash-exp'):
        """Initialize the analyzer with GenAI client."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in environment variables.")
        os.environ['GOOGLE_API_KEY'] = self.api_key
        self.client = genai.Client(http_options={'api_version': 'v1alpha'})
        self.model = model_name

    def fetch_content(self, url):
        """Fetch and extract readable content from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = ' '.join([p.text for p in soup.find_all('p')])
            if not content:
                raise ValueError("No readable content found on the page.")
            return content[:2000]  # Truncate to 2000 characters
        except Exception as e:
            return f"Error fetching the website content: {e}"

    def analyze_content(self, question, content):
        """Send the content and question to GenAI and get the response."""
        try:
            chat = self.client.chats.create(model=self.model)
            response = chat.send_message(f"{question}\n\n{content}")
            if hasattr(response, 'candidates') and response.candidates:
                return "\n".join(
                    part.text for part in response.candidates[0].content.parts if part.text
                )
            elif hasattr(response, 'error'):
                print("DI")
                return f"Error: {response.error.message}"
            return "No response available."
        except Exception as e:
            return f"Error analyzing the content: {e}"

    def process_url(self, url,prompt):
        """Handle the end-to-end process of fetching content and analyzing it."""
        content = self.fetch_content(url)
        if "Error" in content:
            print(content)
        else:
            result = self.analyze_content(prompt, content)
            return (result)

# if __name__ == "__main__":
#     analyzer = WebsiteAnalyzer()
#     user_url = input("Enter the URL of the website: ")
#     analyzer.process_url(user_url)
