

 # Word Stats excels links---https://drive.google.com/drive/folders/1K1Dl3zIrqkCWjVhSCC6H0UMJueU3bMN-?usp=sharing

# RAG Deployment System in Scala
This project implements a scalable Retrieval-Augmented Generation (RAG) system using Scala, Apache Hadoop (MapReduce), Apache Lucene for vector search, and Ollama for embedding generation and LLM-based answer synthesis. It processes PDF documents stored in distributed file systems (e.g., HDFS or S3), chunks them, generates embeddings, builds sharded Lucene indexes, and provides an HTTP API for querying.

Additionally, the project includes components for word statistics and embedding post-processing (e.g., semantic analysis like nearest neighbors, analogies, and similarities), which can be used for vocabulary analysis or standalone embedding tasks.

The system is designed for distributed processing, making it suitable for large-scale document corpora.

## Features
- **Distributed Indexing**: Uses MapReduce to process PDFs in parallel, chunk text, generate embeddings via Ollama, and build sharded Lucene indexes.
- **Vector Search**: Supports cosine, Euclidean, or dot-product similarity for KNN searches in Lucene.
- **Query Pipeline**: Embeds queries, searches across shards (sequentially or via parallel MapReduce), retrieves context, and generates answers using an LLM (e.g., Llama3 via Ollama).
- **API Server**: HTTP endpoints for querying, searching, and health checks using Http4s and Cats Effect.
- **Embedding Post-Processing**: Analyzes word embeddings for semantic neighbors, analogies, and similarities.
- **Utilities**: Vector normalization, PDF text extraction (via PDFBox), text chunking.

## Architecture and Code Components

The codebase is organized into Scala objects and classes under the `ragdeployment` package. Below is a breakdown of key components:

### 1. **Indexing Pipeline (MapReduce-based)**
   - **RagMapper.scala**: 
     - Mapper for processing individual PDF paths.
     - Reads PDFs from HDFS/S3 using Hadoop FileSystem.
     - Extracts text using PDFBox (`Pdfs.readText`).
     - Splits text into overlapping chunks (`Chunker.split` with maxChars=1200, overlap=200).
     - Generates embeddings using Ollama (`Ollama.embed` with configurable model, e.g., "mxbai-embed-large").
     - Normalizes vectors (L2 norm via `Vectors.l2` if configured).
     - Assigns chunks to shards based on document ID hash.
     - Emits (shard_id, JSON chunk record) pairs.
     - Counters: Tracks failures (e.g., pdf_read_failures, empty_embeddings).
   - **RagShardReducer.scala**:
     - Reducer for building Lucene index shards.
     - Receives chunk records per shard.
     - Parses JSON, validates data.
     - Creates local Lucene index using StandardAnalyzer and configurable similarity (COSINE default).
     - Adds documents with fields: doc_id (StringField), chunk_id (StringField), text (TextField), vec (KnnFloatVectorField).
     - Commits index and uploads files to HDFS/S3.
     - Cleans up local temp files.
     - Emits summary records.
     - Counters: Tracks parse_errors, shard_docs.

### 2. **Word Embeddings and Analysis**
   - **WordStatsMapper.scala**:
     - Mapper for word frequency counting.
     - Reads PDFs, extracts text.
     - Cleans text (lowercase, remove punctuation).
     - Filters stopwords and short words.
     - Emits (word, 1) pairs.
   - **PostProcessEmbeddings.scala** (Standalone Utility):
     - Processes MapReduce output from word stats (assumes a WordStatsReducer exists, not in provided files).
     - Parses vocabulary with embeddings.
     - Computes:
       - Nearest neighbors (top-k by cosine similarity).
       - Word similarities for test pairs.
       - Analogies (e.g., king - man + woman ≈ queen).
     - Outputs CSV files: vocabulary.csv, nearest_neighbors.csv, word_similarities.csv, word_analogies.csv.
     - Optimized with pre-computed norms and while-loops for performance.
### 3. **Query Pipeline**
   - **RagQuery.scala**:
     - Core logic for RAG queries.
     - `searchShard`: Searches a single Lucene shard using KnnFloatVectorQuery.
     - `searchAllShards`: Searches all shards in parallel (fan-out/fan-in), merges top-k results.
     - `answer`: Full RAG flow:
       - Embed query with Ollama.
       - Search shards.
       - Pack context from results.
       - Generate answer via Ollama chat API (system prompt ensures context-only responses).
       - Appends source attributions.
   - **ParallelSearchJob.scala** (Optional Distributed Search):
     - Launches MapReduce job for parallel shard searches (useful for very large indexes).
     - Lists shards, creates temp input file with shard paths.
     - Configures job with query vector and topK.
     - Uses `SearchMapper` and `SearchReducer`.
     - Parses and returns results, cleans up temp files.
   - **SearchMapper.scala**:
     - Mapper for searching one shard in parallel mode.
     - Loads query vector from config.
     - Performs KNN search on shard.
     - Emits (negative_score, JSON result) for sorting.
   - **SearchReducer.scala**:
     - Reducer for merging results in parallel mode.
     - Takes top-k from sorted inputs (single reducer).

### 4. **API Server**
   - **RagApiServer.scala**:
     - HTTP server using Http4s, Ember, and Cats Effect.
     - Endpoints:
       - `POST /api/v1/query`: Full RAG (search + answer). Accepts JSON with query, topK, embedModel, chatModel.
       - `GET /api/v1/search?q=<query>&topK=<k>&model=<model>`: Search only, returns results.
       - `GET /api/v1/health`: Checks index status and shard count.
     - Uses `RagQuery` for core logic.
     - Configurable via env vars (RAG_INDEX_PATH, RAG_API_PORT).
     - JSON handling with Circe.



### 5. **Utilities**
   - **Vectors.scala**: Provides L2 normalization for vectors (`Vectors.l2`).
   - **Other Implied Utilities** (not fully detailed in files):
     - `Pdfs`: PDF text extraction.
     - `Chunker`: Text splitting with overlap.
     - `Ollama`: Client for embedding and chat APIs (assumed to handle connections to Ollama server).
     - Logging: Uses SLF4J throughout.

## Program Flow

### Indexing Flow (MapReduce Job)
1. **Input**: Text file with PDF paths (one per line, e.g., s3://bucket/path/to/pdf).
2. **Mappers (`RagMapper`)**: Each mapper processes one PDF:
   - Read PDF → Extract text → Chunk → Embed (Ollama) → Normalize → Assign shard → Emit to reducer.
3. **Reducers (`RagShardReducer`)**: One per shard:
   - Collect chunks → Build Lucene index locally → Upload to HDFS/S3 shard directory.
4. **Output**: Sharded Lucene indexes in `index_shard_X` directories.

To run: Configure a Hadoop Job with `RagMapper` and `RagShardReducer`, set configs (e.g., rag.embed.model, rag.normalize.l2).

### Word Analysis Flow (Standalone)
1. **Word Stats MapReduce**: Use `WordStatsMapper` (and implied reducer) to count words and generate embeddings.
2. **Post-Processing**: Run `PostProcessEmbeddings.run(inputPath)` on output → Generate CSV analyses.

### Query Flow (API or Standalone)
1. **Embed Query**: Use Ollama to generate query vector.
2. **Search**: Two options parallel or sequential
   - Sequential: `RagQuery.searchAllShards` (fan-out to shards).
   - Parallel: `ParallelSearchJob.search` (MapReduce for large-scale).
4. **Retrieve Context**: Pack top-k results into prompt.
5. **Generate Answer**: Use Ollama chat API with system/user messages.
6. **Response**: Answer + sources + metadata.

Via API: POST to `/query` with JSON payload.



## Installation and Setup
1. **Dependencies**:
   - Scala 2.13+ (or compatible).
   - Hadoop (for MapReduce).
   - Lucene (core, analyzers).
   - Ollama (running server for embeddings/LLM).
   - PDFBox (for PDF extraction).
   - Http4s, Cats Effect, Circe (for API).
   - SLF4J (logging).
   - Build with sbt or Maven.
NOTE---Currently PATHS in code follow aws s3 filesystem.
2. **Ollama Setup**: Install Ollama, pull models (e.g., `ollama pull mxbai-embed-large`, `ollama pull llama3`).

3. Build in sbt shell using assembly.

## Running on aws emr
1. **Setup Cluster**: Create emr cluster with appropriate resources.
2. **Load bootstrap file into s3/emr**: this file contains all intial config.
3. **Store all inputs files in s3**: Store all i/p files like jar/input ppaths/Pdfs into s3 for emr to load.
4. **Create cluster**: After creating cluster run the folllowing to start the emr jobs.

**Example Command:**
```bash
 hadoop jar /home/hadoop/RagDeployment-assembly-0.1.0.jar ragdeployment.Driver Driver rag s3://myhadoopemr/paths.txt s3://myhadoopemr/output mxbai-embed-large COSINE
 hadoop jar /home/hadoop/RagDeployment-assembly-0.1.0.jar ragdeployment.Driver Driver wordstats s3://myhadoopemr/paths.txt s3://myhadoopemr/output mxbai-embed-large COSINE

## Usage
- **Build Index**: Launch Hadoop job with input file of PDF paths.
- **Start API Server**: Run `RagApiServer` → Access at `http://localhost:8080/api/v1/`.
- **Query Example** (cURL): curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "seasonal decomposition time series analysis to TravisTorrent data",
    "topK": 5,
    "embedModel": "mxbai-embed-large",
    "chatModel": "llama3"
  }'

# NOTE
TO run in locally, first you have to make some filesystem changes in code, currently it is configured according to AWS s3 filesystem. Then  setup the local hadoop cluster and then run these 3 comands one after another for Rag indexing job, wordstat job and postprocessing respectively.
```bash
 hadoop jar /home/hadoop/RagDeployment-assembly-0.1.0.jar ragdeployment.Driver Driver rag s3://myhadoopemr/paths.txt s3://myhadoopemr/output mxbai-embed-large COSINE
 hadoop jar /home/hadoop/RagDeployment-assembly-0.1.0.jar ragdeployment.Driver Driver rag s3://myhadoopemr/paths.txt s3://myhadoopemr/output mxbai-embed-large COSINE
 java -cp /Users/home/RagDeplyment/RagDeployment/target/scala-3.5.1/RagDeployment-assembly-0.1.0.jar ragdeployment.Driver Driver postprocess /Users/home/RagDeplyment/RagDeployment/src/main/scala/ragdeployment dummy dummy
 ```
