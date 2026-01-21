// 'import' syntax bound
import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import path from "path";
// 'uuidv4' implicitly declared
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

// Compile-time binding (determined during parsing)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Binding 'app' to an Express instance
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

// User authentication middleware (simple version)
const authenticateUser = (req, res, next) => {
  // Generate a user ID if not exists
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

// ========== USER-ISOLATED STORAGE ==========

// Store vector stores per user
const userVectorStores = new Map(); // userId -> vectorStore

// Store document metadata per user
const userDocumentsMetadata = new Map(); // userId -> Map(documentId -> metadata)

// Store memories per user
const userMemories = new Map(); // userId -> memory

// Helper function to get or create user's vector store
const getUserVectorStore = (userId) => {
  if (!userVectorStores.has(userId)) {
    userVectorStores.set(userId, new MemoryVectorStore(embeddings));
  }
  return userVectorStores.get(userId);
};

// Helper function to get or create user's document metadata
const getUserDocumentsMetadata = (userId) => {
  if (!userDocumentsMetadata.has(userId)) {
    userDocumentsMetadata.set(userId, new Map());
  }
  return userDocumentsMetadata.get(userId);
};

// Helper function to get or create user's memory
const getUserMemory = (userId) => {
  if (!userMemories.has(userId)) {
    userMemories.set(userId, new BufferMemory({
      returnMessages: true,
      memoryKey: "history",
    }));
  }
  return userMemories.get(userId);
};

// ========== ROUTES WITH USER ISOLATION ==========

// Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// 1. UPLOAD & RAG PROCESSING (User-specific)
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const userId = req.userId;
    console.log(`👤 User ${userId} uploading PDF...`);

    // Get user's vector store and metadata
    const userVectorStore = getUserVectorStore(userId);
    const userDocsMetadata = getUserDocumentsMetadata(userId);

    // LangChain PDF Processing
    const blob = new Blob([req.file.buffer], { type: "application/pdf" });
    const loader = new PDFLoader(blob);
    const docs = await loader.load();
    
    // Text Chunking
    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log(`📄 User ${userId}: Split into ${splitDocs.length} chunks`);

    // Add metadata with user ownership
    const docsWithMetadata = splitDocs.map((doc, index) => ({
      ...doc,
      metadata: {
        ...doc.metadata,
        source: req.file.originalname,
        chunkId: uuidv4(),
        chunkIndex: index,
        uploadedAt: new Date().toISOString(),
        userId: userId, // Mark ownership
        owner: userId
      }
    }));

    // Add to user's vector store
    await userVectorStore.addDocuments(docsWithMetadata);
    console.log(`✅ Documents added to user ${userId}'s vector store`);

    // Store metadata
    const documentId = uuidv4();
    userDocsMetadata.set(documentId, {
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
      message: "PDF processed and stored in your private space",
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

// 2. RAG QUERY (Only searches user's documents)
app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;
    const userId = req.userId;
    
    if (!question) return res.status(400).json({ error: "No question provided" });

    console.log(`🔍 User ${userId} performing RAG query...`);

    // Get user's vector store
    const userVectorStore = getUserVectorStore(userId);
    const userDocsMetadata = getUserDocumentsMetadata(userId);

    // Check if user has any documents
    if (userDocsMetadata.size === 0) {
      return res.status(404).json({ 
        error: "You haven't uploaded any documents yet. Please upload a PDF first.",
        userId: userId
      });
    }

    // LANGCHAIN: Semantic search in user's vector store
    const relevantDocs = await userVectorStore.similaritySearch(question, 4);
    console.log(`📚 User ${userId}: Found ${relevantDocs.length} relevant chunks`);

    // Filter to ensure we only return this user's documents
    const userRelevantDocs = relevantDocs.filter(doc => 
      doc.metadata.userId === userId
    );

    // Build context from user's documents only
    const context = userRelevantDocs.map((doc, index) =>
      `[Source ${index + 1} from "${doc.metadata.source}"]:\n${doc.pageContent}\n`
    ).join("\n");

    // Get user's memory
    const memory = getUserMemory(userId);

    // LANGCHAIN: Create conversation chain with memory
    const chain = new ConversationChain({
      llm: chatModel,
      memory: memory
    });

    const ragPrompt = `
    CONTEXT FROM YOUR DOCUMENTS:
    ${context}

    USER QUESTION: ${question}

    INSTRUCTIONS:
    - Answer based ONLY on the context provided from your documents
    - If the context doesn't contain the answer, say: "I cannot find this information in your uploaded documents."
    - Be helpful and conversational
    - Use **bold** for important terms
    - Use bullet points • for lists

    Response based on your documents:`;

    // LANGCHAIN: Generate response
    const response = await chain.call({ input: ragPrompt });

    res.json({
      answer: response.response,
      sources: userRelevantDocs.map(doc => ({
        source: doc.metadata.source,
        page: doc.metadata.loc?.pageNumber || 'N/A',
        contentPreview: doc.pageContent.substring(0, 150) + '...',
        chunkId: doc.metadata.chunkId
      })),
      userId: userId,
      relevantChunks: userRelevantDocs.length,
      note: "Searching only in your uploaded documents"
    });

  } catch (error) {
    console.error("RAG Query error:", error);
    res.status(500).json({ error: "RAG query failed: " + error.message });
  }
});

// 3. TEXT SUMMARIZATION (User-specific documents only)
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

    // If documentId provided, summarize the user's document
    if (documentId) {
      console.log(`🔍 User ${userId} summarizing document: ${documentId}`);

      const userDocsMetadata = getUserDocumentsMetadata(userId);
      const document = userDocsMetadata.get(documentId);

      if (!document) {
        return res.status(404).json({ 
          error: "Document not found in your library",
          userId: userId
        });
      }

      // Verify ownership
      if (document.userId !== userId) {
        return res.status(403).json({ 
          error: "You don't have permission to access this document",
          userId: userId
        });
      }

      // Get user's vector store
      const userVectorStore = getUserVectorStore(userId);
      
      // Search for this document's chunks
      const relevantDocs = await userVectorStore.similaritySearch(
        document.filename.substring(0, 50),
        50
      );
      
      // Filter for this specific document AND user ownership
      const documentChunks = relevantDocs.filter(doc =>
        doc.metadata.source === document.filename && doc.metadata.userId === userId
      );

      if (documentChunks.length === 0) {
        return res.status(404).json({ 
          error: "No content found for this document in your library",
          userId: userId
        });
      }

      textToSummarize = documentChunks.map(chunk => chunk.pageContent).join("\n\n");
      documentName = document.filename;
      type = "document";
      console.log(`📄 User ${userId}: Found ${documentChunks.length} chunks from document: ${documentName}`);
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

    console.log(`📋 User ${userId}: Summarizing text (${textToSummarize.length} characters)...`);
    const summary = await summaryChain.invoke({
      text: textToSummarize.substring(0, 3000)
    });

    console.log(`✅ User ${userId}: Summary generated: ${summary.length} characters`);

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

// 4. CHAT WITH MEMORY MANAGEMENT (User-specific)
app.post("/chat", async (req, res) => {
  try {
    const { message, clearMemory = false } = req.body;
    const userId = req.userId;
    
    if (!message) return res.status(400).json({ error: "No message provided" });

    // Get user's memory
    let memory = getUserMemory(userId);

    // Clear memory if requested
    if (clearMemory) {
      userMemories.delete(userId);
      memory = getUserMemory(userId); // Create fresh memory
    }

    // LANGCHAIN: Create conversation chain
    const chain = new ConversationChain({
      llm: chatModel,
      memory: memory
    });

    // LANGCHAIN: Generate response with memory
    const response = await chain.call({ input: message });

    res.json({
      response: response.response,
      userId: userId,
      memoryLength: (await memory.chatHistory.getMessages()).length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: "Chat failed: " + error.message });
  }
});

// 5. MEMORY MANAGEMENT (User-specific)
app.get("/memory/:userId", async (req, res) => {
  try {
    const requestedUserId = req.params.userId;
    const currentUserId = req.userId;

    // Users can only access their own memory
    if (requestedUserId !== currentUserId) {
      return res.status(403).json({
        error: "You can only access your own memory",
        userId: currentUserId
      });
    }

    if (!userMemories.has(currentUserId)) {
      return res.json({
        userId: currentUserId,
        messageCount: 0,
        recentMessages: [],
        status: "No memory found"
      });
    }

    const memory = userMemories.get(currentUserId);
    const chatHistory = await memory.chatHistory.getMessages();

    const recentMessages = chatHistory.slice(-10).map(msg => ({
      type: msg._getType(),
      content: msg.content,
      timestamp: new Date().toISOString()
    }));

    res.json({
      userId: currentUserId,
      messageCount: chatHistory.length,
      recentMessages: recentMessages,
      status: "Memory retrieved successfully"
    });

  } catch (error) {
    console.error("Memory retrieval error:", error);
    res.status(500).json({
      error: "Memory retrieval failed: " + error.message,
      userId: req.userId
    });
  }
});

app.delete("/memory/:userId", (req, res) => {
  try {
    const requestedUserId = req.params.userId;
    const currentUserId = req.userId;

    // Users can only delete their own memory
    if (requestedUserId !== currentUserId) {
      return res.status(403).json({
        success: false,
        message: "You can only clear your own memory",
        userId: currentUserId
      });
    }

    if (!userMemories.has(currentUserId)) {
      return res.json({
        success: false,
        message: "No memory found for this user",
        userId: currentUserId
      });
    }

    const deleted = userMemories.delete(currentUserId);

    res.json({
      success: true,
      message: deleted ? "Your memory cleared successfully" : "No memory found",
      userId: currentUserId,
      deleted: deleted
    });

  } catch (error) {
    console.error("Memory deletion error:", error);
    res.status(500).json({
      success: false,
      error: "Memory deletion failed: " + error.message
    });
  }
});

// 6. DOCUMENT MANAGEMENT (User-specific)
app.get("/documents", (req, res) => {
  const userId = req.userId;
  const userDocsMetadata = getUserDocumentsMetadata(userId);
  const documents = Array.from(userDocsMetadata.values());

  res.json({
    totalDocuments: documents.length,
    userId: userId,
    documents: documents
  });
});

// 7. USER PROFILE ENDPOINT
app.get("/profile", (req, res) => {
  const userId = req.userId;
  const userDocsMetadata = getUserDocumentsMetadata(userId);
  const documentsCount = userDocsMetadata.size;
  const hasMemory = userMemories.has(userId);
  const hasVectorStore = userVectorStores.has(userId);

  res.json({
    userId: userId,
    stats: {
      documents: documentsCount,
      hasMemory: hasMemory,
      hasVectorStore: hasVectorStore
    },
    session: req.session
  });
});

// 8. HEALTH CHECK
app.get("/health", (req, res) => {
  const totalUsers = new Set([
    ...userVectorStores.keys(),
    ...userDocumentsMetadata.keys(),
    ...userMemories.keys()
  ]).size;

  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    system: {
      usingLangChain: true,
      framework: "Node.js + LangChain",
      userIsolation: true,
      features: [
        "User-specific document storage",
        "Private RAG queries",
        "Isolated memory management",
        "Multi-tenant vector stores"
      ]
    },
    stats: {
      totalUsers: totalUsers,
      totalDocuments: Array.from(userDocumentsMetadata.values()).reduce((sum, map) => sum + map.size, 0),
      activeVectorStores: userVectorStores.size,
      activeMemories: userMemories.size
    }
  });
});

// 9. CLEANUP OLD SESSIONS (optional, for production)
setInterval(() => {
  const now = Date.now();
  const oneDay = 24 * 60 * 60 * 1000;
  
  // In production, you'd want to clean up old user data
  // This is a simplified version
  console.log(`🧹 Cleanup: ${userVectorStores.size} vector stores, ${userMemories.size} memories`);
}, 60 * 60 * 1000); // Every hour

app.listen(PORT, () => {
  console.log(`🚀 LangChain RAG System with User Isolation running on http://localhost:${PORT}`);
  console.log(`🔐 Features: User-specific document storage, Private RAG queries`);
  console.log(`🤖 LLM: Google Gemma 2B via OpenRouter`);
  console.log(`🔒 User Isolation: Enabled`);
});
