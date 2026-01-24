process.env.SHARP_IGNORE_GLOBAL_LIBVIPS = "1";
process.env.DISABLE_SHARP = "true";

import express from "express";
import cors from "cors";
import multer from "multer";
import pdfParse from "pdf-parse";
import axios from "axios";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";

dotenv.config();



const app = express();
app.use(cors({
  origin: true,
    
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"],
}));
app.options("*", cors()); 

app.use(express.json());

const upload = multer();

const VERTEX_API_KEY = process.env.VERTEX_API_KEY;
console.log("API KEY FOUND?", !!VERTEX_API_KEY);
let embedderPromise = null;

async function loadEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    ).then(model => {
      console.log("✅ Local embedding model loaded");
      return model;
    });
  }
  return embedderPromise;
}



let embeddingQueue = Promise.resolve();

async function getLocalEmbedding(text) {
  embeddingQueue = embeddingQueue.then(async () => {
    const embedder = await loadEmbedder();

    const output = await embedder(text, {
      pooling: "mean",
      normalize: true,
    });

    return Array.from(output.data);
  });

  return embeddingQueue;
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}


app.post("/analyze", upload.single("resume"), async (req, res) => {
  try {
    const pdf = await pdfParse(req.file.buffer);

    const chunks = pdf.text
      .split("\n\n")
      .filter(c => c.length > 300)
      .slice(0, 5); // safe limit

    // 2️⃣ Create embeddings locally
    const embeddedChunks = [];
    for (const chunk of chunks) {
      const embedding = await getLocalEmbedding(chunk);
      embeddedChunks.push({ text: chunk, embedding });
    }

    // 3️⃣ Embed query
    const query = "Analyze resume for ATS score, skills, grammar and improvements";
    const queryEmbedding = await getLocalEmbedding(query);

    // 4️⃣ Retrieve top-K chunks (RAG CORE)
    const topChunks = embeddedChunks
      .map(c => ({
        text: c.text,                            //RAG model
        score: cosineSimilarity(queryEmbedding, c.embedding),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(c => c.text)
      .join("\n\n");

    // 3️⃣ Embed query (retrieval step)
    // const query = "Analyze resume for ATS score, skills, grammar and improvements";
   
    // 4️⃣ Retrieve top relevant chunks (RAG CORE) 

    // 5️⃣ Send ONLY retrieved content to Gemini
    const prompt = `
You are an ATS resume analyzer.

Analyze the resume content below and return:
- ATS Score /100
- Skill Match Score /100
- Grammar Score /100
- Summary
- Improvements

Resume:
${topChunks}
`;

    const response = await axios.post(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + VERTEX_API_KEY,
      {
        contents: [{ parts: [{ text: prompt }] }]
      }
    );
    const embedding = response.data.embedding.values;

    const result =
      response.data.candidates?.[0]?.content?.parts?.[0]?.text ||
      "No response from AI";

    res.json({ success: true, output: result });

  } catch (err) {
    console.error("FULL AI ERROR:", err.response?.data || err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get("/", (req, res) => {
  res.send("AI Resume Analyzer Backend Running");
});

app.listen(8080, () => console.log("Server running on 8080"));
