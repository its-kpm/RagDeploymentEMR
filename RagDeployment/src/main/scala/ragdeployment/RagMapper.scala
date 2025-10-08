package ragdeployment

import io.circe.syntax.*
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.slf4j.LoggerFactory

import java.io.InputStream
import java.nio.file.Paths

/**
 * Mapper for RAG index building.
 *
 * Pipeline:
 * 1. Read PDF from path (HDFS or S3)
 * 2. Extract text using PDFBox
 * 3. Split into overlapping chunks
 * 4. Generate embeddings via Ollama
 * 5. Normalize vectors (L2 if needed)
 * 6. Emit (shard_id, json_chunk) pairs
 *
 * Input: LongWritable (line number), Text (PDF path)
 * Output: IntWritable (shard ID), Text (JSON with doc_id, chunk_id, text, embedding)
 */
class RagMapper extends Mapper[LongWritable, Text, IntWritable, Text] {

  private val log = LoggerFactory.getLogger(getClass)

  // Ollama client for embedding generation (reused across map calls)
  private val client = new Ollama()

  /**
   * Map function: Process a single PDF file.
   *
   * @param key Line number in input file (unused)
   * @param v PDF file path (HDFS or S3)
   * @param ctx Mapper context for output and configuration
   */
  override def map(key: LongWritable, v: Text, ctx: Mapper[LongWritable, Text, IntWritable, Text]#Context): Unit = {
    // Parse input path
    val pathStr = v.toString.trim
    log.info("=" * 80)
    log.info(s"Map task started for input: $pathStr")

    // Get Hadoop filesystem (works with HDFS, S3, local, etc.)
    val conf = ctx.getConfiguration
    val fs = FileSystem.get(conf)
    val s3Path = new Path(pathStr)
    val docId = s3Path.getName  // Extract filename as document ID

    log.info(s"Document ID: $docId")
    log.info(s"Filesystem: ${fs.getScheme}")

    // ========================================================================
    // Step 1: Read PDF from distributed filesystem
    // ========================================================================
    log.info(s"Step 1: Reading PDF from $pathStr")
    val text = try {
      // Open input stream from HDFS/S3
      val inputStream = fs.open(s3Path)
      try {
        log.debug(s"Opened input stream for $docId")
        // PDFBox can read directly from InputStream
        val extractedText = Pdfs.readText(inputStream)
        log.info(s"Successfully read PDF: ${extractedText.length} characters")
        extractedText
      } finally {
        inputStream.close()
        log.debug(s"Closed input stream for $docId")
      }
    } catch {
      case e: Exception =>
        log.error(s"Failed to read PDF $docId: ${e.getMessage}", e)
        log.error(s"Path attempted: $pathStr")
        ctx.getCounter("rag", "pdf_read_failures").increment(1)
        return  // Skip this document
    }

    // Validate extracted text
    if (text.trim.isEmpty) {
      log.warn(s"No text extracted from $docId (empty or image-only PDF)")
      ctx.getCounter("rag", "empty_pdfs").increment(1)
      return  // Skip this document
    }

    // ========================================================================
    // Step 2: Split text into chunks
    // ========================================================================
    log.info(s"Step 2: Chunking text (maxChars=1200, overlap=200)")
    val chunks = Chunker.split(text, maxChars = 1200, overlap = 200).toVector

    if (chunks.isEmpty) {
      log.warn(s"No text chunks extracted from $docId")
      ctx.getCounter("rag", "no_chunks").increment(1)
      return  // Skip this document
    }

    log.info(s"Created ${chunks.size} chunks")
    log.debug(s"First chunk preview: ${chunks.head.take(100)}...")

    // ========================================================================
    // Step 3: Generate embeddings via Ollama
    // ========================================================================
    log.info(s"Step 3: Generating embeddings for ${chunks.size} chunks")

    // Get embedding model from configuration
    val model = ctx.getConfiguration.get("rag.embed.model", "mxbai-embed-large")
    log.info(s"Using embedding model: $model")

    val rawVecs = try {
      // Call Ollama API to generate embeddings
      val vecs = client.embed(chunks, model)
      log.info(s"Generated ${vecs.length} embeddings")

      // Validate embeddings
      var emptyCount = 0
      vecs.zipWithIndex.foreach { case (vec, idx) =>
        if (vec.isEmpty) {
          log.warn(s"Chunk $idx: Empty embedding vector!")
          ctx.getCounter("rag", "empty_embeddings").increment(1)
          emptyCount += 1
        } else {
          log.debug(s"Chunk $idx: ${vec.length}-dimensional vector")
        }
      }

      if (emptyCount > 0) {
        log.warn(s"Warning: $emptyCount out of ${vecs.length} embeddings are empty")
      }

      vecs
    } catch {
      case e: Exception =>
        log.error(s"Embedding generation failed for $docId: ${e.getMessage}", e)
        ctx.getCounter("rag", "embed_failures").increment(1)
        return  // Skip this document
    }

    // ========================================================================
    // Step 4: Normalize vectors (L2 normalization for cosine similarity)
    // ========================================================================
    val useL2 = ctx.getConfiguration.getBoolean("rag.normalize.l2", true)
    log.info(s"Step 4: Vector normalization (L2=${useL2})")

    val vecs = if (useL2) {
      log.debug("Applying L2 normalization to vectors")
      rawVecs.map(Vectors.l2)
    } else {
      log.debug("Skipping normalization")
      rawVecs
    }

    // ========================================================================
    // Step 5: Assign to shard and emit records
    // ========================================================================
    log.info(s"Step 5: Emitting records to reducers")

    // Determine shard based on document ID hash
    val numReducers = ctx.getNumReduceTasks
    val shard = math.abs(docId.hashCode) % numReducers
    log.info(s"Assigned to shard $shard (out of $numReducers reducers)")

    var emittedCount = 0
    var skippedCount = 0

    // Emit each chunk with its embedding
    chunks.zip(vecs).zipWithIndex.foreach { case ((chunk, embedding), id) =>
      // Skip invalid chunks/embeddings
      if (chunk.trim.nonEmpty && embedding.nonEmpty) {
        // Create JSON record
        val rec = Map(
          "doc_id"   -> docId.asJson,
          "chunk_id" -> id.asJson,
          "text"     -> chunk.asJson,
          "vec"      -> embedding.toVector.asJson
        ).asJson.noSpaces

        // Emit to reducer
        ctx.write(new IntWritable(shard), new Text(rec))
        emittedCount += 1

        log.trace(s"Emitted chunk $id: ${chunk.take(50)}...")
      } else {
        log.warn(s"Skipping chunk $id: empty content or embedding")
        ctx.getCounter("rag", "skipped_chunks").increment(1)
        skippedCount += 1
      }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    log.info("=" * 80)
    log.info(s"Completed processing $docId")
    log.info(s"Summary:")
    log.info(s"  Total chunks: ${chunks.size}")
    log.info(s"  Emitted: $emittedCount")
    log.info(s"  Skipped: $skippedCount")
    log.info(s"  Assigned to shard: $shard")
    log.info("=" * 80)
  }

  /**
   * Cleanup method called once after all map tasks complete.
   * Closes Ollama client to release resources.
   *
   * @param ctx Mapper context
   */
  override def cleanup(ctx: Mapper[LongWritable, Text, IntWritable, Text]#Context): Unit = {
    log.info("Mapper cleanup: Closing Ollama client")
    client.close()
    log.info("Ollama client closed successfully")
  }
}