package ragdeployment

import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.slf4j.LoggerFactory
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.InputStream

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

/**
 * Mapper for Word Stats and embeddings
 */
class WordStatsMapper extends Mapper[LongWritable, Text, Text, IntWritable]:
  private val log = LoggerFactory.getLogger(getClass)

  // Define a basic stopword set (you can expand this list)
  private val stopwords = Set(
    "the", "is", "and", "a", "an", "in", "on", "of", "for", "to",
    "this", "that", "it", "with", "as", "by", "at", "from", "or", "be",
    "are", "was", "were", "has", "had", "have", "not", "but", "can", "could"
  )

  override def map(key: LongWritable, value: Text, ctx: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit =
    // Each line is a PDF path
    val pathStr = value.toString.trim
    if (pathStr.isEmpty) {
      log.warn("Empty path string received")
      ctx.getCounter("rag", "empty_paths").increment(1)
      return
    }

    log.info(s"=== Processing path: $pathStr ===")

    val conf = ctx.getConfiguration
    val s3Path = new Path(pathStr)
    val docId = s3Path.getName

    log.info(s"DocId: $docId")

    // Step 1: Read PDF from S3
    val text: String = try {
      val fs = FileSystem.get(s3Path.toUri, conf)
      log.info(s"FileSystem URI: ${fs.getUri}")
      log.info(s"S3 Path exists: ${fs.exists(s3Path)}")

      if (!fs.exists(s3Path)) {
        log.error(s"Path does not exist: $pathStr")
        ctx.getCounter("rag", "path_not_found").increment(1)
        return
      }

      val inputStream: InputStream = fs.open(s3Path)
      Pdfs.readText(inputStream)
    } catch {
      case e: Exception =>
        log.error(s"Failed to read PDF $docId: ${e.getMessage}", e)
        ctx.getCounter("rag", "pdf_read_failures").increment(1)
        return
    }


    // Preprocess: lowercase + remove punctuation
    val cleanText = text.toLowerCase.replaceAll("[^a-z ]", " ")

    // Split into words, filter stopwords, and emit <word, 1>
    cleanText.split("\\s+")
      .filter(word => word.nonEmpty && word.length >= 2 && !stopwords.contains(word))
      .foreach { word =>
        ctx.write(new Text(word), new IntWritable(1))
      }
