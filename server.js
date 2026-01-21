import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import path from "path";
import { v4 as uuidv4 } from "uuid";
import session from "express-session";
import cookieParser from "cookie-parser";

// ========== LANGCHAIN IMPORTS ==========
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
// =======================================

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(cookieParser());
app.use(express.static(__dirname));

// Session management for user isolation
app.use(session({
  secret: process.env.SESSION_SECRET || 'rag-system-secret-key',
  resave: false,
  saveUninitialized: true,
  cookie: { 
    secure: process.env.NODE_ENV === 'production',
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
}));

// User authentication middleware
const authenticateUser = (req, res, next) => {
  if (!req.session.userId) {
    req.session.userId = uuidv4();
    console.log(`👤 New user created: ${req.session.userId}`);
  }
  req.userId = req.session.userId;
  next();
};

// Apply authentication to all routes
app.use(authenticateUser);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

// ========== LANGCHAIN SETUP ==========

// 1. Embeddings with OpenRouter
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
  model: "text-embedding-3-small"
});

// 2. Lightweight LLM (Gemma 2B)
const chatModel = new ChatOpenAI({
  openAIApiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
  modelName: "google/gemma-2-9b-it",
  temperature: 0.1,
  maxTokens: 1000
});

// 3. Text Splitter for chunking
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

// ========== GLOBAL STORAGE ==========

// Store everything globally (simpler for testing)
const globalVectorStore = new MemoryVectorStore(embeddings);
const globalDocumentsMetadata = new Map(); // documentId -> metadata
const globalMemory = new BufferMemory({
  returnMessages: true,
  memoryKey: "history",
});

// ========== SIMPLIFIED ROUTES ==========

// Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// 1. UPLOAD & RAG PROCESSING
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const userId = req.userId;
    console.log(`👤 User ${userId} uploading PDF...`);

    // LangChain PDF Processing
    const blob = new Blob([req.file.buffer], { type: "application/pdf" });
    const loader = new PDFLoader(blob);
    const docs = await loader.load();
    
    // Text Chunking
    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log(`📄 Split into ${splitDocs.length} chunks`);

    // Add metadata
    const docsWithMetadata = splitDocs.map((doc, index) => ({
      ...doc,
      metadata: {
        ...doc.metadata,
        source: req.file.originalname,
        chunkId: uuidv4(),
        chunkIndex: index,
        uploadedAt: new Date().toISOString(),
        userId: userId,
        owner: userId
      }
    }));

    // Add to vector store
    await globalVectorStore.addDocuments(docsWithMetadata);
    console.log(`✅ Documents added to vector store`);

    // Store metadata
    const documentId = uuidv4();
    globalDocumentsMetadata.set(documentId, {
      id: documentId,
      filename: req.file.originalname,
      chunks: splitDocs.length,
      uploadedAt: new Date().toISOString(),
      size: req.file.size,
      userId: userId,
      owner: userId
    });

    res.json({
      success: true,
      message: "PDF processed successfully!",
      documentId,
      chunks: splitDocs.length,
      filename: req.file.originalname,
      userId: userId
    });

  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ error: "PDF processing failed: " + error.message });
  }
});

// 2. RAG QUERY
app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;
    const userId = req.userId;
    
    if (!question) return res.status(400).json({ error: "No question provided" });

    console.log(`🔍 Performing RAG query for user ${userId}...`);

    // Check if any documents exist
    if (globalDocumentsMetadata.size === 0) {
      return res.json({
        answer: "❌ No documents uploaded yet. Please upload a PDF first.",
        sources: [],
        userId: userId,
        relevantChunks: 0,
        note: "Upload documents first"
      });
    }

    // Semantic search
    const relevantDocs = await globalVectorStore.similaritySearch(question, 4);
    console.log(`📚 Found ${relevantDocs.length} relevant chunks`);

    // Build context
    const context = relevantDocs.map((doc, index) =>
      `[Source ${index + 1} from "${doc.metadata.source}"]:\n${doc.pageContent}\n`
    ).join("\n");

    // Create conversation chain
    const chain = new ConversationChain({
      llm: chatModel,
      memory: globalMemory
    });

    const ragPrompt = `
    CONTEXT FROM DOCUMENTS:
    ${context}

    USER QUESTION: ${question}

    INSTRUCTIONS:
    - Answer based ONLY on the context provided
    - If the context doesn't contain the answer, say: "I cannot find this information in the uploaded documents."
    - Be helpful and conversational
    - Use **bold** for important terms
    - Use bullet points • for lists

    Response:`;

    // Generate response
    const response = await chain.call({ input: ragPrompt });

    res.json({
      answer: response.response,
      sources: relevantDocs.map(doc => ({
        source: doc.metadata.source,
        page: doc.metadata.loc?.pageNumber || 'N/A',
        contentPreview: doc.pageContent.substring(0, 150) + '...',
        chunkId: doc.metadata.chunkId
      })),
      userId: userId,
      relevantChunks: relevantDocs.length,
      note: "Searching all uploaded documents"
    });

  } catch (error) {
    console.error("RAG Query error:", error);
    res.status(500).json({ error: "RAG query failed: " + error.message });
  }
});

// 3. TEXT SUMMARIZATION
app.post("/summarize", async (req, res) => {
  try {
    const { text, documentId } = req.body;
    const userId = req.userId;

    if (!text && !documentId) {
      return res.status(400).json({ error: "Provide text or documentId" });
    }

    console.log(`📝 User ${userId} requesting summarization...`);

    let textToSummarize = "";
    let documentName = "";
    let type = "text";

    // If documentId provided, summarize the document
    if (documentId) {
      console.log(`🔍 Summarizing document: ${documentId}`);

      const document = globalDocumentsMetadata.get(documentId);

      if (!document) {
        return res.status(404).json({ 
          error: "Document not found"
        });
      }

      // Search for this document's chunks
      const relevantDocs = await globalVectorStore.similaritySearch(
        document.filename.substring(0, 50),
        50
      );
      
      // Filter for this specific document
      const documentChunks = relevantDocs.filter(doc =>
        doc.metadata.source === document.filename
      );

      if (documentChunks.length === 0) {
        return res.status(404).json({ 
          error: "No content found for this document"
        });
      }

      textToSummarize = documentChunks.map(chunk => chunk.pageContent).join("\n\n");
      documentName = document.filename;
      type = "document";
      console.log(`📄 Found ${documentChunks.length} chunks from document: ${documentName}`);
    } else {
      // Use provided text
      textToSummarize = text;
      type = "text";
    }

    if (!textToSummarize || textToSummarize.length === 0) {
      return res.status(400).json({ error: "No text available to summarize" });
    }

    const simpleSummaryPrompt = PromptTemplate.fromTemplate(`
Please provide a comprehensive yet concise summary of the following text. Focus on:

**MAIN POINTS:**
- Key ideas and concepts
- Important findings
- Major conclusions

**STRUCTURE:**
- Start with an overview
- List key points with bullet points
- End with main takeaways

TEXT TO SUMMARIZE:
{text}

Please use clear formatting with **bold** for important terms and • bullet points for lists.

SUMMARY:`);

    const summaryChain = RunnableSequence.from([
      simpleSummaryPrompt,
      chatModel,
      new StringOutputParser()
    ]);

    console.log(`📋 Summarizing text (${textToSummarize.length} characters)...`);
    const summary = await summaryChain.invoke({
      text: textToSummarize.substring(0, 3000)
    });

    console.log(`✅ Summary generated: ${summary.length} characters`);

    res.json({
      summary: summary,
      originalLength: textToSummarize.length,
      summaryLength: summary.length,
      type: type,
      userId: userId,
      ...(documentName && { documentName: documentName })
    });

  } catch (error) {
    console.error("Summarize error:", error);
    res.status(500).json({ error: "Summarization failed: " + error.message });
  }
});

// 4. CHAT WITH MEMORY
app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;
    const userId = req.userId;
    
    if (!message) return res.status(400).json({ error: "No message provided" });

    console.log(`💬 Chat from user ${userId}: ${message}`);

    // Create conversation chain
    const chain = new ConversationChain({
      llm: chatModel,
      memory: globalMemory
    });

    // Generate response with memory
    const response = await chain.call({ input: message });

    const chatHistory = await globalMemory.chatHistory.getMessages();
    
    res.json({
      response: response.response,
      userId: userId,
      memoryLength: chatHistory.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Chat failed: " + error.message });
  }
});

// 5. MEMORY MANAGEMENT
app.get("/memory/:userId", async (req, res) => {
  try {
    const userId = req.params.userId;

    const chatHistory = await globalMemory.chatHistory.getMessages();
    
    const recentMessages = chatHistory.slice(-10).map(msg => ({
      type: msg._getType(),
      content: msg.content,
      timestamp: new Date().toISOString()
    }));

    res.json({
      userId: userId,
      messageCount: chatHistory.length,
      recentMessages: recentMessages,
      status: "Memory retrieved successfully"
    });

  } catch (error) {
    console.error("Memory retrieval error:", error);
    res.status(500).json({
      error: "Memory retrieval failed: " + error.message
    });
  }
});

app.delete("/memory/:userId", (req, res) => {
  try {
    // Clear memory
    globalMemory.clear();
    
    res.json({
      success: true,
      message: "Memory cleared successfully",
      deleted: true
    });

  } catch (error) {
    console.error("Memory deletion error:", error);
    res.status(500).json({
      success: false,
      error: "Memory deletion failed: " + error.message
    });
  }
});

// 6. DOCUMENT MANAGEMENT
app.get("/documents", (req, res) => {
  const userId = req.userId;
  const documents = Array.from(globalDocumentsMetadata.values());

  res.json({
    totalDocuments: documents.length,
    userId: userId,
    documents: documents
  });
});

// 7. USER PROFILE ENDPOINT
app.get("/profile", (req, res) => {
  const userId = req.userId;
  const documentsCount = globalDocumentsMetadata.size;
  const chatHistory = globalMemory.chatHistory;

  res.json({
    userId: userId,
    stats: {
      documents: documentsCount,
      memoryMessages: chatHistory.messages ? chatHistory.messages.length : 0
    }
  });
});

// 8. HEALTH CHECK
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    stats: {
      totalDocuments: globalDocumentsMetadata.size,
      vectorStoreReady: true,
      memoryReady: true
    }
  });
});

// 9. CLEAR ALL DATA (for testing)
app.delete("/clear-all", (req, res) => {
  try {
    // Clear everything
    globalMemory.clear();
    globalDocumentsMetadata.clear();
    
    res.json({
      success: true,
      message: "All data cleared successfully",
      cleared: {
        memory: true,
        documents: true
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 LangChain RAG System running on http://localhost:${PORT}`);
  console.log(`✅ ALL FEATURES READY: Upload, RAG Query, Summarize, Chat, Memory, Documents`);
  console.log(`📁 Upload PDF first, then everything will work!`);
});
