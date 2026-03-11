# RAG Chatbot System (LangChain + Node.js)

This project is a **Retrieval Augmented Generation (RAG) Chatbot System** built using **Node.js and LangChain**.
The system allows users to interact with an AI chatbot that can process documents, summarize text, and maintain conversation memory.

The chatbot retrieves relevant information from uploaded documents and combines it with AI responses to provide more accurate answers.

---

# Live Application

You can access the deployed application here:

**Main Website:**
[https://radiantsampleapplication.onrender.com/](https://radiantsampleapplication.onrender.com/)

---

# Features

* AI chatbot powered by **LangChain**
* **Retrieval Augmented Generation (RAG)** for better responses
* **PDF document processing**
* **Text summarization**
* **Chat memory management**
* **File upload for document analysis**
* Simple browser-based interface

---

# Technologies Used

* **Node.js**
* **Express.js**
* **LangChain**
* **OpenAI API**
* **PDF-Parse**
* **Multer (File Upload)**
* **JavaScript**
* **HTML**

---

# Project Structure

| File / Folder       | Description                                |
| ------------------- | ------------------------------------------ |
| `server.js`         | Main backend server handling chatbot logic |
| `index.html`        | Frontend user interface                    |
| `package.json`      | Project configuration and dependencies     |
| `package-lock.json` | Dependency version control                 |
| `.env`              | Stores API keys and environment variables  |

---

# Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-chatbot-system.git
```

---

### 2. Navigate to the Project Folder

```bash
cd rag-chatbot-system
```

---

### 3. Install Dependencies

```bash
npm install
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your API key:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 5. Run the Application

```bash
npm start
```

---

# Access the Application

After running the server, open your browser and go to:

```
http://localhost:3000
```

---

# How the System Works

1. The user sends a question through the chatbot interface.
2. The system processes the request using **LangChain**.
3. Relevant information is retrieved from uploaded documents.
4. The AI generates a response using **Retrieval Augmented Generation (RAG)**.
5. The chatbot returns an accurate answer to the user.

---

# Dependencies

Main libraries used in this project:

* Express
* LangChain
* OpenAI
* Multer
* PDF-Parse
* Dotenv
* Cors
* UUID

---

# Purpose of the Project

This project demonstrates how to build an **AI-powered RAG chatbot system** that can:

* Process documents
* Retrieve relevant information
* Generate AI responses
* Maintain conversation memory

It serves as a **practical example of integrating LangChain with a Node.js backend**.

---

# Notes

* Requires a valid **OpenAI API key**.
* The application runs on **port 3000 by default**.
* PDF files can be uploaded for document-based question answering.
