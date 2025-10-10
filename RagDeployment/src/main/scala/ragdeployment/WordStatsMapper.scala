package ragdeployment

import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.slf4j.LoggerFactory

import java.io.InputStream
import java.nio.file.{Files, Paths}

/**
 * Mapper for Word Statistics and embeddings.
 *
 * Works both in distributed (HDFS/S3) and local environments.
 *
 * Pipeline:
 * 1. Read PDF (auto-detects local vs distributed FS)
 * 2. Extract text via PDFBox
 * 3. Normalize and tokenize text
 * 4. Filter stopwords
 * 5. Emit (word, 1) pairs for reducers
 */
class WordStatsMapper extends Mapper[LongWritable, Text, Text, IntWritable]:
  private val log = LoggerFactory.getLogger(getClass)

  private val stopwords = Set(
    "the", "is", "and", "a", "an", "in", "on", "of", "for", "to",
    "this", "that", "it", "with", "as", "by", "at", "from", "or", "be",
    "are", "was", "were", "has", "had", "have", "not", "but", "can", "could"
  )

  override def map(key: LongWritable, value: Text, ctx: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit =
    val pathStr = value.toString.trim
    if (pathStr.isEmpty) {
      log.warn("Empty path string received")
      ctx.getCounter("rag", "empty_paths").increment(1)
      return
    }

    log.info("=" * 80)
    log.info(s"WordStatsMapper started for input: $pathStr")

    val conf = ctx.getConfiguration
    val hadoopPath = new Path(pathStr)
    val docId = hadoopPath.getName
    var text: String = ""

    try {
      // Detect local vs distributed FS
      if (pathStr.startsWith("s3://") || pathStr.startsWith("s3a://") || pathStr.startsWith("hdfs://")) {
        val fs = FileSystem.get(hadoopPath.toUri, conf)
        log.info(s"Reading PDF from distributed FS: ${fs.getScheme}://${hadoopPath}")
        if (!fs.exists(hadoopPath)) {
          log.error(s"Path not found: $pathStr")
          ctx.getCounter("rag", "path_not_found").increment(1)
          return
        }

        val inputStream = fs.open(hadoopPath)
        try {
          text = Pdfs.readText(inputStream)
        } finally inputStream.close()
      } else {
        // Local file system (Mac/Windows)
        log.info(s"Reading PDF from local path: $pathStr")
        text = Pdfs.readText(Paths.get(pathStr))
      }

      if (text.trim.isEmpty) {
        log.warn(s"No text extracted from $docId (empty or image-only PDF)")
        ctx.getCounter("rag", "empty_pdfs").increment(1)
        return
      }

    } catch {
      case e: Exception =>
        log.error(s"Failed to read PDF $docId: ${e.getMessage}", e)
        ctx.getCounter("rag", "pdf_read_failures").increment(1)
        return
    }

    // Step 2: Preprocess text
    val cleanText = text.toLowerCase.replaceAll("[^a-z ]", " ")

    // Step 3: Split words and emit <word, 1>
    val words = cleanText.split("\\s+")
      .filter(word => word.nonEmpty && word.length >= 2 && !stopwords.contains(word))

    log.info(s"Extracted ${words.length} candidate words from $docId")

    words.foreach { word =>
      ctx.write(new Text(word), new IntWritable(1))
    }

    ctx.getCounter("wordstats", "processed_words").increment(1)
    log.info(s"Completed processing for $docId with ${words.length} words emitted.")
    log.info("=" * 80)
