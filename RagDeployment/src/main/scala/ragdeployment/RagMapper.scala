package ragdeployment

import io.circe.syntax.*
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.slf4j.LoggerFactory

import java.io.InputStream
import java.nio.file.{Files, Paths}

/**
 * Mapper for RAG index building.
 *
 * Pipeline:
 * 1. Read PDF from path (HDFS, S3, or Local)
 * 2. Extract text using PDFBox
 * 3. Split into overlapping chunks
 * 4. Generate embeddings via Ollama
 * 5. Normalize vectors (L2 if needed)
 * 6. Emit (shard_id, json_chunk) pairs
 */
class RagMapper extends Mapper[LongWritable, Text, IntWritable, Text] {

  private val log = LoggerFactory.getLogger(getClass)
  private val client = new Ollama()

  override def map(key: LongWritable, v: Text, ctx: Mapper[LongWritable, Text, IntWritable, Text]#Context): Unit = {
    val pathStr = v.toString.trim
    log.info("=" * 80)
    log.info(s"Map task started for input: $pathStr")

    val conf = ctx.getConfiguration
    val fs = FileSystem.get(conf)
    val hadoopPath = new Path(pathStr)
    val docId = hadoopPath.getName

    log.info(s"Document ID: $docId")
    log.info(s"Filesystem detected: ${fs.getScheme}")

    // Step 1: Read PDF from either distributed or local filesystem
    val text = try {
      val extractedText =
        if (pathStr.startsWith("s3://") || pathStr.startsWith("s3a://") || pathStr.startsWith("hdfs://")) {
          log.info(s"Reading from distributed FS: $pathStr")
          val inputStream = fs.open(hadoopPath)
          try {
            Pdfs.readText(inputStream)
          } finally {
            inputStream.close()
          }
        } else {
          // Local mode for Mac/Windows development
          log.info(s"Reading from local file system: $pathStr")
          Pdfs.readText(Paths.get(pathStr))
        }

      log.info(s"Successfully read PDF (${extractedText.length} characters) for $docId")
      extractedText
    } catch {
      case e: Exception =>
        log.error(s"Failed to read PDF $docId: ${e.getMessage}", e)
        ctx.getCounter("rag", "pdf_read_failures").increment(1)
        return
    }

    if (text.trim.isEmpty) {
      log.warn(s"No text extracted from $docId (empty or image-only PDF)")
      ctx.getCounter("rag", "empty_pdfs").increment(1)
      return
    }

    // Step 2: Split text into chunks
    log.info(s"Step 2: Chunking text (maxChars=1200, overlap=200)")
    val chunks = Chunker.split(text, maxChars = 1200, overlap = 200).toVector

    if (chunks.isEmpty) {
      log.warn(s"No chunks extracted for $docId")
      ctx.getCounter("rag", "no_chunks").increment(1)
      return
    }

    log.info(s"Created ${chunks.size} chunks")

    // Step 3: Generate embeddings via Ollama
    val model = ctx.getConfiguration.get("rag.embed.model", "mxbai-embed-large")
    log.info(s"Step 3: Generating embeddings using model: $model")

    val rawVecs = try {
      val vecs = client.embed(chunks, model)
      log.info(s"Generated ${vecs.length} embeddings")
      vecs
    } catch {
      case e: Exception =>
        log.error(s"Embedding generation failed for $docId: ${e.getMessage}", e)
        ctx.getCounter("rag", "embed_failures").increment(1)
        return
    }

    // Step 4: Normalize vectors (L2)
    val useL2 = ctx.getConfiguration.getBoolean("rag.normalize.l2", true)
    log.info(s"Step 4: Vector normalization (L2=$useL2)")
    val vecs = if (useL2) rawVecs.map(Vectors.l2) else rawVecs

    // Step 5: Emit records to reducers
    val numReducers = ctx.getNumReduceTasks
    val shard = math.abs(docId.hashCode) % numReducers
    log.info(s"Assigned to shard $shard (out of $numReducers reducers)")

    var emittedCount = 0
    chunks.zip(vecs).zipWithIndex.foreach { case ((chunk, embedding), id) =>
      if (chunk.trim.nonEmpty && embedding.nonEmpty) {
        val rec = Map(
          "doc_id" -> docId.asJson,
          "chunk_id" -> id.asJson,
          "text" -> chunk.asJson,
          "vec" -> embedding.toVector.asJson
        ).asJson.noSpaces

        ctx.write(new IntWritable(shard), new Text(rec))
        emittedCount += 1
      } else {
        ctx.getCounter("rag", "skipped_chunks").increment(1)
      }
    }

    log.info("=" * 80)
    log.info(s"Completed processing $docId — emitted $emittedCount chunks to shard $shard")
    log.info("=" * 80)
  }

  override def cleanup(ctx: Mapper[LongWritable, Text, IntWritable, Text]#Context): Unit = {
    log.info("Mapper cleanup: Closing Ollama client")
    client.close()
    log.info("Ollama client closed successfully")
  }
}
