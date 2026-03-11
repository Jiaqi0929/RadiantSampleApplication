# RAG Chatbot System (LangChain + Node.js)

This project is a Retrieval Augmented Generation (RAG) Chatbot System built using Node.js and LangChain. The system allows users to interact with an AI chatbot that can process documents, summarize text, and maintain conversation memory. The chatbot retrieves relevant information from uploaded documents and combines it with AI responses to provide more accurate answers.

## 🔗 Live Demo
[Try the Chatbot](https://radiantsampleapplication.onrender.com/)

## ✨ Features
- AI chatbot powered by LangChain
- Retrieval Augmented Generation (RAG) for better responses
- PDF document processing
- Text summarization
- Chat memory management
- File upload for document analysis
- Clean, responsive chat interface

## 🛠️ Technologies Used
- **Backend:** Node.js, Express.js
- **AI/ML:** LangChain, OpenRouter API
- **File Processing:** Multer, PDF-Parse
- **Frontend:** HTML, JavaScript

## 📁 Project Structure
| File / Folder | Description |
|---------------|-------------|
| server.js | Main backend server handling chatbot logic |
| index.html | Frontend user interface |
| package.json | Project configuration and dependencies |
| package-lock.json | Dependency version control |
| .env | Stores API keys and environment variables |

## 🚀 Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-chatbot-system.git
   cd rag-chatbot-system
   
2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Add your OpenAI API key**  
   Create a `.env` file in the project root:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   npm start
   ```

5. **Access locally**  
   Open `http://localhost:3000` in your browser.

## 💡 What I Learned
- Implementing RAG architecture for better AI responses
- Integrating LangChain with Node.js
- Processing and extracting text from PDFs
- Managing conversation state and memory
- Working with OpenAI APIs

## 📝 Note
- A valid OPENROUTER API key is required to run the project locally.
- The application runs on port 3000 by default.
- PDF files can be uploaded for document-based question answering.
