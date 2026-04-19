// 'import' syntax bound
import express from "express"; // Web server framework
import multer from "multer"; // Handles file uploads
import dotenv from "dotenv"; // Loads environment variables (.env file)
import { fileURLToPath } from "url"; // File path utilities
import path from "path";
// 'uuidv4' implicitly declared 
import { v4 as uuidv4 } from "uuid"; //Generates unique IDs

// ========== LANGCHAIN IMPORTS ==========
// Static Binding
import { MemoryVectorStore } from "langchain/vectorstores/memory"; // Stores document embeddings in memory
import { OpenAIEmbeddings } from "@langchain/openai"; // Converts text to vectors
import { ChatOpenAI } from "@langchain/openai";
// Link-time bindings for PDF loading
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { BufferMemory } from "langchain/memory"; // Stores conversation history
import { ConversationChain } from "langchain/chains"; // Sequences of LLM operations
import { PromptTemplate } from "langchain/prompts"; // Templates for LLM instructions
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
// =======================================

dotenv.config();

// Compile-time binding (determined during parsing)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Binding 'app' to an Express instance
// Explicit with 'const'
const app = express();
// Binding 'PORT' to a numeric value
// Load-time binding (happens when module is loaded)
const PORT = process.env.PORT || 3000;


// Middleware
app.use(express.json());
app.use(express.static(__dirname));

const upload = multer({ 
  storage: multer.memoryStorage(),
  // Bound when server starts
  limits: { fileSize: 10 * 1024 * 1024 }
});

// ========== LANGCHAIN SETUP ==========

// 1. Embeddings with OpenRouter
// Converts text to vector embeddings for semantic search
// Bound at module load
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
  modelName: "nousresearch/hermes-3-llama-3.1-8b:free",
  temperature: 0.1, //  Low randomness for consistent answers
  maxTokens: 1000 // Response length limit
});

// 3. Vector Store for RAG
// Explicit with 'let'
// In-memory storage for document embeddings
// Stores chunks as vectors for similarity search
let vectorStore = new MemoryVectorStore(embeddings);

// 4. Text Splitter for chunking
const textSplitter = new RecursiveCharacterTextSplitter({
  // Static numeric literal - bound at load time
  chunkSize: 1000, // Each chunk ~1000 characters
  chunkOverlap: 200, // // 200 chars overlap between chunks
});

// 5. Memory Management
// STATIC (Application Lifetime)
// GLOBAL VARIABLE
const userMemories = new Map(); // Accessible throughout file
// Allocated once, lives entire app runtime
// 6. Document Metadata Storage
// MODULE SCOPE
const documentsMetadata = new Map(); // Visible to all functions 
// Never deallocated until server stops

// ========== ROUTES WITH LANGCHAIN ==========

// Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// 1. UPLOAD & RAG PROCESSING
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    console.log("🔄 Processing PDF with LangChain...");

    // BLOCK-SCOPED VARIABLES (blob, loader ONLY exist in this try block)
    // LANGCHAIN: Load PDF
    // HEAP-DYNAMIC (Explicit Management)

    // LangChain PDF Processing
    // 1. Get uploaded file
    const blob = new Blob([req.file.buffer], { type: "application/pdf" }); 
    // 2. Load PDF with LangChain
    const loader = new PDFLoader(blob);                                    
    const docs = await loader.load();    
    // 3. Split into chunks
    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log(`📄 Split into ${splitDocs.length} chunks`);
    // 4. Add metadata to each chunk
    const docsWithMetadata = splitDocs.map((doc, index) => ({
      ...doc,
      metadata: {
        ...doc.metadata,
        source: req.file.originalname,
        chunkId: uuidv4(),
        chunkIndex: index,
        uploadedAt: new Date().toISOString()
      }
    }));
    // 5. Store in vector database
    await vectorStore.addDocuments(docsWithMetadata);
    console.log("✅ Documents added to vector store");
    // 6. Save document metadata
    const documentId = uuidv4();                     
    documentsMetadata.set(documentId, {       
      id: documentId,                         
      filename: req.file.originalname,        
      chunks: splitDocs.length,               
      uploadedAt: new Date().toISOString(),
      size: req.file.size
    });

    // 7. Return success
    res.json({
      success: true,
      message: "PDF processed with LangChain RAG",
      documentId,
      chunks: splitDocs.length,
      filename: req.file.originalname
    });

  } catch (error) {
    // No declaration - implicitly creates global property
    console.error("Upload error:", error); 
    res.status(500).json({ error: "PDF processing failed: " + error.message });
  }
});

// 2. RAG QUERY (Retrieval Augmented Generation)
app.post("/ask", async (req, res) => {
  try {
    // Bound at runtime
    // LOCAL VARIABLES
    // FUNCTION SCOPE VARIABLES
    const { question, userId = "default" } = req.body; // Only accessible in this function 
    if (!question) return res.status(400).json({ error: "No question provided" });

    console.log("🔍 Performing RAG query...");

    // 1. Semantic search in vector store
    const relevantDocs = await vectorStore.similaritySearch(question, 4);
    console.log(`📚 Found ${relevantDocs.length} relevant chunks`);

    // 2. Build context from relevant documents
    const context = relevantDocs.map((doc, index) => 
      `[Source ${index + 1} from "${doc.metadata.source}"]:\n${doc.pageContent}\n`
    ).join("\n");

    // 3. Get or create user memory
    // userMemories is NONLOCAL - not declared here but accessible
    if (!userMemories.has(userId)) {
      userMemories.set(userId, new BufferMemory({ // Can modify global
        returnMessages: true,
        memoryKey: "history",
      }));
    }
    const memory = userMemories.get(userId);       // Can read global

    // 4. Create conversation chain with memory
    const chain = new ConversationChain({ 
      llm: chatModel,
      memory: memory
    });

     // 5. Build RAG prompt with context
    const ragPrompt = `
    CONTEXT FROM DOCUMENTS:
    ${context}

    CONVERSATION HISTORY: [Available in memory]

    USER QUESTION: ${question}

    INSTRUCTIONS:
    - Answer conversationally like a helpful assistant
    - Use **bold** for important terms and key points
    - Use bullet points • for lists when helpful
    - Use numbered lists for steps or sequences
    - Break into clear paragraphs for readability
    - Be concise but thorough
    - If information comes from documents, mention it naturally
    - If context doesn't have the answer, say so politely and offer general help

    Please provide a helpful, well-formatted response:`;

    // 6. Generate response
    const response = await chain.call({ input: ragPrompt });

    // 7. Return answer with sources
    res.json({
      answer: response.response,
      sources: relevantDocs.map(doc => ({
        source: doc.metadata.source,
        page: doc.metadata.loc?.pageNumber || 'N/A',
        contentPreview: doc.pageContent.substring(0, 150) + '...',
        chunkId: doc.metadata.chunkId
      })),
      userId: userId,
      relevantChunks: relevantDocs.length
    });

  } catch (error) {
    console.error("RAG Query error:", error);
    res.status(500).json({ error: "RAG query failed: " + error.message });
  }
});

// 3. TEXT SUMMARIZATION
app.post("/summarize", async (req, res) => {
  try {
    // Declaration at point of use
    const { text, documentId } = req.body;

    if (!text && !documentId) {
      return res.status(400).json({ error: "Provide text or documentId" });
    }

    console.log("📝 Performing text summarization...");

    // Dynamic binding
    // Initially bound as string
    // Declared where needed, not at top
    let textToSummarize = "";
    let documentName = "";
    // Node.js infers: string
    let type = "text";

    // If documentId provided, summarize the document
    if (documentId) {
      console.log(`🔍 Summarizing document: ${documentId}`);
      
      // Get document metadata
      // Declaration inside conditional block
      const document = documentsMetadata.get(documentId);
      if (!document) {
        return res.status(404).json({ error: "Document not found" });
      }

      // Get all chunks for this document
      const relevantDocs = await vectorStore.similaritySearch(
        document.filename.substring(0, 50), // First 50 chars of filename
        50 // Limit to 50 results
      );
      const documentChunks = relevantDocs.filter(doc => 
        doc.metadata.source === document.filename
      );

      if (documentChunks.length === 0) {
        return res.status(404).json({ error: "No content found for this document" });
      }

      // Combine all chunks
      // Dynamically REBOUND to new string value
      textToSummarize = documentChunks.map(chunk => chunk.pageContent).join("\n\n");
      documentName = document.filename;
      type = "document";
      // Type stays string but VALUE changes dynamically
      console.log(`📄 Found ${documentChunks.length} chunks from document: ${documentName}`);
    } else {
      // Use provided text
      textToSummarize = text;
      type = "text";
    }

    if (!textToSummarize || textToSummarize.length === 0) {
      return res.status(400).json({ error: "No text available to summarize" });
    }

    // Use a simpler prompt that works better with the model
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

    // Create chain
    const summaryChain = RunnableSequence.from([
      simpleSummaryPrompt,
      chatModel,
      new StringOutputParser()
    ]);

    // Generate summary
    console.log(`📋 Summarizing text (${textToSummarize.length} characters)...`);
    const summary = await summaryChain.invoke({ 
      text: textToSummarize.substring(0, 3000)  // Limit text length for performance
    });

    console.log(`✅ Summary generated: ${summary.length} characters`);

    res.json({
      summary: summary,
      originalLength: textToSummarize.length,
      summaryLength: summary.length,
      type: type,
      ...(documentName && { documentName: documentName })
    });

  } catch (error) {
    console.error("Summarize error:", error);
    res.status(500).json({ error: "Summarization failed: " + error.message });
  }
});

// 4. CHAT WITH MEMORY MANAGEMENT
app.post("/chat", async (req, res) => {
  try {
    const { message, userId = "default", clearMemory = false } = req.body;
    if (!message) return res.status(400).json({ error: "No message provided" });

    // Clear memory if requested
    if (clearMemory && userMemories.has(userId)) {
      userMemories.delete(userId);
    }

    // Get or create user memory
    if (!userMemories.has(userId)) {
      userMemories.set(userId, new BufferMemory({
        returnMessages: true,
        memoryKey: "history",
      }));
    }
    const memory = userMemories.get(userId);

    // Create conversation chain
    const chain = new ConversationChain({ 
      llm: chatModel,
      memory: memory
    });

    // Generate response with memory
    const response = await chain.call({ input: message });

    // Return response with memory stats
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

// 5. MEMORY MANAGEMENT
app.get("/memory/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    
    if (!userMemories.has(userId)) {
      return res.json({ 
        userId: userId,
        messageCount: 0,
        recentMessages: [],
        status: "No memory found"
      });
    }
    // Access global Map
    const memory = userMemories.get(userId);
    const chatHistory = await memory.chatHistory.getMessages();
    // Only the last 10 messages are shown in the display
    const recentMessages = chatHistory.slice(-10).map(msg => ({
      type: msg._getType(),
      content: msg.content,
      timestamp: new Date().toISOString()
    }));
    
    // Server creates and talks back to the browser
    res.json({ 
      userId: userId,
      messageCount: chatHistory.length,
      recentMessages: recentMessages,
      status: "Memory retrieved successfully"
    });
    
  } catch (error) {
    console.error("Memory retrieval error:", error);
    res.status(500).json({ 
      error: "Memory retrieval failed: " + error.message,
      userId: req.params.userId
    });
  }
});

app.delete("/memory/:userId", (req, res) => {
  try {
    const { userId } = req.params;
    
    if (!userMemories.has(userId)) {
      return res.json({ 
        success: false, 
        message: "No memory found for this user",
        userId: userId
      });
    }
    
    const deleted = userMemories.delete(userId);
    
    res.json({ 
      success: true, 
      message: deleted ? "Memory cleared successfully" : "No memory found",
      userId: userId,
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


// 6. DOCUMENT MANAGEMENT
app.get("/documents", (req, res) => {
  const documents = Array.from(documentsMetadata.values());
  res.json({
    totalDocuments: documents.length,
    documents: documents
  });
});

// 7. HEALTH CHECK
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    system: {
      usingLangChain: true,
      framework: "Node.js + LangChain",
      features: [
        "Retrieval Augmented Generation (RAG)",
        "Lightweight LLM (Gemma 2B)", 
        "Text Summarization",
        "Memory Management",
        "Vector Store",
        "PDF Processing"
      ],
      components: {
        vectorStore: "MemoryVectorStore",
        embeddings: "OpenAIEmbeddings",
        llm: "ChatOpenAI (Gemma 2B)",
        memory: "BufferMemory",
        textSplitter: "RecursiveCharacterTextSplitter"
      }
    },
    stats: {
      documents: documentsMetadata.size,
      activeUsers: userMemories.size
    }
  });
});

app.get("/test-all-models", async (req, res) => {
  const modelsToTest = [
    "nousresearch/hermes-3-llama-3.1-8b:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "microsoft/phi-3-mini-4k-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "liquid/lfm-40b:free",
    "openai/gpt-3.5-turbo",  // Paid but cheap
    "mistralai/mistral-7b-instruct",  // Paid but cheap
  ];
  
  const results = {};
  
  for (const modelName of modelsToTest) {
    try {
      const testModel = new ChatOpenAI({
        openAIApiKey: process.env.OPENROUTER_API_KEY,
        configuration: { baseURL: "https://openrouter.ai/api/v1" },
        modelName: modelName,
        temperature: 0.1,
        maxTokens: 20
      });
      
      const response = await testModel.invoke("Say 'ok'");
      results[modelName] = { working: true, response: response.content };
    } catch (error) {
      results[modelName] = { working: false, error: error.message };
    }
    
    // Small delay to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  res.json(results);
});

app.listen(PORT, () => {
  console.log(`🚀 LangChain RAG System running on http://localhost:${PORT}`);
  console.log(`📚 Using Node.js with LangChain`);
  console.log(`🔗 Features: RAG, Lightweight LLM, Text Summarization, Memory Management`);
  console.log(`🤖 LLM: Google Gemma 2B via OpenRouter`);
  console.log(`💾 Vector Store: MemoryVectorStore`);
});
