package ragdeployment

import io.circe.parser.parse
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document.*
import org.apache.lucene.index.{IndexWriter, IndexWriterConfig, VectorSimilarityFunction}
import org.apache.lucene.store.FSDirectory
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path as JPath}
import java.util.Comparator
import scala.jdk.CollectionConverters.*

/**
 * Reducer for building Lucene index shards.
 *
 * Each reducer receives chunks assigned to its shard and:
 * 1. Creates a local Lucene index
 * 2. Adds documents with text and vector embeddings
 * 3. Commits and closes the index
 * 4. Uploads index files to HDFS/S3
 * 5. Cleans up local temporary files
 *
 * Input: IntWritable (shard ID), Text (JSON chunk records)
 * Output: Text (summary), Text (statistics)
 */
class RagShardReducer extends Reducer[IntWritable, Text, Text, Text] {

  private val log = LoggerFactory.getLogger(getClass)

  /**
   * Copy all files from local directory to HDFS/S3.
   * Used to upload the built Lucene index to distributed storage.
   *
   * @param localDir Local directory containing Lucene index files
   * @param hdfsDir Target HDFS/S3 directory
   * @param fs Hadoop FileSystem instance
   */
  private def copyLocalDirToHdfs(localDir: JPath, hdfsDir: Path, fs: FileSystem): Unit = {
    log.info(s"Copying index from local $localDir to HDFS $hdfsDir")

    var fileCount = 0
    var totalBytes = 0L

    Files.list(localDir).iterator().asScala.foreach { p =>
      val fileName = p.getFileName.toString
      val fileSize = Files.size(p)

      log.debug(s"Copying file: $fileName (${fileSize} bytes)")

      // Create output stream in HDFS/S3
      val out = new Path(hdfsDir, fileName)
      val os = fs.create(out, true)  // true = overwrite if exists

      try {
        Files.copy(p, os)
        fileCount += 1
        totalBytes += fileSize
      } finally {
        os.close()
      }
    }

    log.info(s"Successfully copied $fileCount files (${totalBytes} bytes) to HDFS")
  }

  /**
   * Reduce function: Build Lucene index shard for assigned chunks.
   *
   * @param key Shard ID (IntWritable)
   * @param values Iterator of JSON chunk records
   * @param ctx Reducer context for output and configuration
   */
  override def reduce(key: IntWritable, values: java.lang.Iterable[Text], ctx: Reducer[IntWritable, Text, Text, Text]#Context): Unit = {
    val shard = key.get

    log.info("=" * 80)
    log.info(s"Reducer started for shard $shard")
    log.info("=" * 80)

    // ========================================================================
    // Step 1: Create local temporary directory for Lucene index
    // ========================================================================
    log.info("Step 1: Creating temporary Lucene index directory")
    val local: JPath = Files.createTempDirectory(s"lucene-shard-$shard")
    log.info(s"Temporary directory: $local")

    // ========================================================================
    // Step 2: Initialize Lucene IndexWriter
    // ========================================================================
    log.info("Step 2: Initializing Lucene IndexWriter")

    // StandardAnalyzer for text tokenization
    val analyzer = new StandardAnalyzer()
    log.debug("Created StandardAnalyzer")

    // Configure index writer
    val iwConf = new IndexWriterConfig(analyzer)
    log.debug("Created IndexWriterConfig")

    // Open index writer on local filesystem
    val iw = new IndexWriter(FSDirectory.open(local), iwConf)
    log.info("IndexWriter opened successfully")

    // ========================================================================
    // Step 3: Configure vector similarity function
    // ========================================================================
    log.info("Step 3: Configuring vector similarity function")
    val simName = ctx.getConfiguration.get("rag.similarity", "COSINE")

    val sim = simName match {
      case "EUCLIDEAN" =>
        log.info("Using EUCLIDEAN distance for vector similarity")
        VectorSimilarityFunction.EUCLIDEAN
      case "DOT_PRODUCT" =>
        log.info("Using DOT_PRODUCT for vector similarity")
        VectorSimilarityFunction.DOT_PRODUCT
      case _ =>
        log.info("Using COSINE similarity (default)")
        VectorSimilarityFunction.COSINE
    }

    // ========================================================================
    // Step 4: Process incoming chunk records and build index
    // ========================================================================
    log.info("Step 4: Processing chunk records and building index")

    var docCount = 0
    var parseErrors = 0

    values.forEach { t =>
      // Parse JSON record
      parse(t.toString) match {
        case Right(json) =>
          // Extract fields from JSON
          val c = json.hcursor
          val docId = c.get[String]("doc_id").getOrElse("unknown")
          val chunkId = c.get[Int]("chunk_id").getOrElse(-1)
          val text = c.get[String]("text").getOrElse("")
          val vec = c.get[Vector[Float]]("vec").getOrElse(Vector.empty).toArray

          // Validate extracted data
          if (docId == "unknown") {
            log.warn(s"Document with missing doc_id, chunk $chunkId")
          }
          if (text.isEmpty) {
            log.warn(s"Empty text for $docId chunk $chunkId")
          }
          if (vec.isEmpty) {
            log.warn(s"Empty vector for $docId chunk $chunkId")
          }

          // Create Lucene document
          val doc = new Document()

          // Add stored fields (retrievable)
          doc.add(new StringField("doc_id", docId, Field.Store.YES))
          doc.add(new StringField("chunk_id", chunkId.toString, Field.Store.YES))
          doc.add(new TextField("text", text, Field.Store.YES))

          // Add vector field for KNN search
          doc.add(new KnnFloatVectorField("vec", vec, sim))

          // Add document to index
          iw.addDocument(doc)
          docCount += 1

          // Log progress every 100 documents
          if (docCount % 100 == 0) {
            log.debug(s"Indexed $docCount documents so far...")
          }

        case Left(err) =>
          // JSON parsing failed
          log.error(s"Failed to parse JSON: ${err.getMessage}")
          log.debug(s"Problematic JSON: ${t.toString.take(200)}...")
          ctx.getCounter("rag", "parse_errors").increment(1)
          parseErrors += 1
      }
    }

    // ========================================================================
    // Step 5: Commit and close Lucene index
    // ========================================================================
    log.info("Step 5: Committing and closing Lucene index")

    try {
      iw.commit()
      log.debug("Index committed successfully")
    } catch {
      case e: Exception =>
        log.error(s"Failed to commit index: ${e.getMessage}", e)
        throw e
    }

    iw.close()
    log.info("IndexWriter closed")

    log.info(s"Shard $shard: Successfully indexed $docCount documents")
    if (parseErrors > 0) {
      log.warn(s"Shard $shard: $parseErrors parse errors encountered")
    }

    // ========================================================================
    // Step 6: Upload index to HDFS/S3
    // ========================================================================
    log.info("Step 6: Uploading index to distributed storage")

    // Get output path from job configuration
    val outRoot = FileOutputFormat.getOutputPath(ctx)
    val shardPath = new Path(outRoot, s"index_shard_$shard")
    log.info(s"Target HDFS path: $shardPath")

    // Get filesystem (HDFS, S3, etc.)
    val fs = shardPath.getFileSystem(ctx.getConfiguration)
    log.debug(s"Using filesystem: ${fs.getScheme}")

    // Create shard directory
    fs.mkdirs(shardPath)
    log.debug(s"Created directory: $shardPath")

    // Copy all index files to HDFS/S3
    copyLocalDirToHdfs(local, shardPath, fs)

    // ========================================================================
    // Step 7: Clean up local temporary files
    // ========================================================================
    log.info("Step 7: Cleaning up temporary files")

    try {
      // Delete directory and all contents (reverse order for directories)
      Files.walk(local)
        .sorted(Comparator.reverseOrder())
        .forEach(Files.delete)
      log.info(s"Deleted temporary directory: $local")
    } catch {
      case e: Exception =>
        log.warn(s"Failed to delete temporary directory $local: ${e.getMessage}")
      // Non-fatal, continue anyway
    }

    // ========================================================================
    // Step 8: Emit summary record
    // ========================================================================
    log.info("Step 8: Writing output summary")

    val summary = s"Indexed $docCount docs"
    ctx.write(new Text(s"shard=$shard"), new Text(summary))

    // Update counters
    ctx.getCounter("rag", s"shard_${shard}_docs").increment(docCount)

    // ========================================================================
    // Summary
    // ========================================================================
    log.info("=" * 80)
    log.info(s"Reducer completed for shard $shard")
    log.info(s"Summary:")
    log.info(s"  Documents indexed: $docCount")
    log.info(s"  Parse errors: $parseErrors")
    log.info(s"  Output location: $shardPath")
    log.info("=" * 80)
  }
}