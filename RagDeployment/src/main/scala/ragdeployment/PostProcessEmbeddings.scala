package ragdeployment

import org.slf4j.LoggerFactory
import java.io.{File, PrintWriter}
import scala.io.Source

/**
 * PostProcessEmbeddings — Final version
 * - Handles flexible input format: "word word,<freq>,[vector]"
 * - Exports full 1024-dim vectors in CSV (no dim column)
 * - Includes semantic neighbors, similarity, and analogy tasks
 */
object PostProcessEmbeddings {

  private val log = LoggerFactory.getLogger(getClass)

  case class NormalizedWord(word: String, vec: Array[Float], norm: Float, frequency: Int)

  def cosineSimilarity(v1: Array[Float], norm1: Float, v2: Array[Float], norm2: Float): Double = {
    if (norm1 == 0 || norm2 == 0) return 0.0
    if (v1.length != v2.length) {
      log.error(s"Vector dimension mismatch: ${v1.length} vs ${v2.length}")
      return 0.0
    }
    val dot = v1.zip(v2).foldLeft(0.0f)((sum, pair) => sum + pair._1 * pair._2)
    dot / (norm1 * norm2)
  }

  def computeNorm(v: Array[Float]): Float = {
    math.sqrt(v.foldLeft(0.0f)((sum, x) => sum + x * x)).toFloat
  }

  def add(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    v1.zip(v2).map { case (x, y) => x + y }
  }

  def sub(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    v1.zip(v2).map { case (x, y) => x - y }
  }

  /**
   * Robust parser for input lines like:
   * "word word,<freq>,[v1,v2,...]"
   * "word,<freq>,[v1,v2,...]"
   * Handles extra spaces, commas, or tabs.
   */
  def parseLine(line: String): Option[(String, Array[Float], Int)] = {
    try {
      if (line == null || line.trim.isEmpty) return None

      val bracketStart = line.indexOf('[')
      val bracketEnd = line.lastIndexOf(']')
      if (bracketStart == -1 || bracketEnd == -1) return None

      val beforeVec = line.substring(0, bracketStart).trim
      val vecStr = line.substring(bracketStart + 1, bracketEnd)
      val vec = vecStr.split(",").map(_.trim.toFloat)

      // Split before the vector: e.g. "word word,3," -> ["word", "word", "3"]
      val tokens = beforeVec.split("[,\\s]+").map(_.trim).filter(_.nonEmpty)
      if (tokens.isEmpty) return None

      val word = tokens.head
      val freqToken = tokens.find(t => t.forall(_.isDigit)).getOrElse("1")
      val frequency = freqToken.toInt

      Some((word, vec, frequency))
    } catch {
      case e: Exception =>
        log.warn(s"Failed to parse line: ${line.take(100)}... - ${e.getMessage}")
        None
    }
  }

  def run(inputPath: String): Unit = {
    log.info("=" * 80)
    log.info("Starting Word Embedding Post-Processing (FINAL VERSION)")
    log.info("=" * 80)
    println("=" * 80)
    println("DEBUG MODE: Enhanced logging enabled")
    println("=" * 80)
    println(s"Input path: $inputPath")

    // Determine input files
    val inputFile = new File(inputPath)
    val filesToProcess = if (inputFile.isDirectory) {
      log.info("Input is directory, searching for part-r-* files")
      inputFile.listFiles().filter(_.getName.startsWith("part-r-"))
    } else if (inputFile.isFile) {
      log.info("Input is single file")
      Array(inputFile)
    } else {
      log.error(s"Invalid input path: $inputPath")
      Array.empty[File]
    }

    if (filesToProcess.isEmpty) {
      println("ERROR: No valid input files found.")
      System.exit(1)
    }
    println(s"Found ${filesToProcess.length} file(s) to process")

    // Parse vocabulary
    println("Parsing vocabulary...")
    var lineCount = 0
    var parsedCount = 0

    val vocab = filesToProcess.flatMap { file =>
      println(s"  Reading file: ${file.getName}")
      Source.fromFile(file).getLines()
        .filter { line =>
          lineCount += 1
          line != null && line.trim.nonEmpty
        }
        .flatMap { line =>
          parseLine(line) match {
            case Some((word, vec, freq)) =>
              parsedCount += 1
              if (parsedCount <= 5)
                println(s"    Sample word ${parsedCount}: ${word} (freq: ${freq}, dim: ${vec.length})")
              Some((word, (vec, freq)))
            case None =>
              None
          }
        }
    }.toMap

    println(s"\nParsing complete:")
    println(s"  Total lines read: $lineCount")
    println(s"  Successfully parsed: $parsedCount")
    println(s"  Final vocabulary size: ${vocab.size}")

    if (vocab.isEmpty) {
      println("ERROR: No valid embeddings loaded. Check file format.")
      System.exit(1)
    }

    // Pre-compute norms
    println("\nPre-computing vector norms...")
    val normalizedVocab = vocab.map { case (word, (vec, freq)) =>
      val norm = computeNorm(vec)
      NormalizedWord(word, vec, norm, freq)
    }.toArray
    println(s"Pre-computed norms for ${normalizedVocab.length} words")

    // Export full vocabulary (no embedding_dim)
    println("\nExporting vocabulary.csv (full 1024-d embeddings)...")
    val vocabWriter = new PrintWriter("vocabulary.csv")
    vocabWriter.println("word,token_id,frequency,embedding")
    normalizedVocab.sortBy(_.word).zipWithIndex.foreach { case (nw, idx) =>
      val vecString = nw.vec.map(v => f"$v%.6f").mkString("[", ",", "]")
      vocabWriter.println(s"${nw.word},$idx,${nw.frequency},$vecString")
    }
    vocabWriter.close()
    println("✓ vocabulary.csv written successfully")

    // Task 1: Semantic Neighbors
    println("\n" + "=" * 80)
    println("Task 1: Computing Nearest Neighbors")
    println("=" * 80)
    val pwNN = new PrintWriter("nearest_neighbors.csv")
    pwNN.println("word,neighbor_1,sim_1,neighbor_2,sim_2,neighbor_3,sim_3,neighbor_4,sim_4,neighbor_5,sim_5")
    val startTime = System.currentTimeMillis()
    normalizedVocab.zipWithIndex.foreach { case (nw, index) =>
      val similarities = normalizedVocab
        .filter(_.word != nw.word)
        .map { other =>
          val sim = cosineSimilarity(nw.vec, nw.norm, other.vec, other.norm)
          (other.word, sim)
        }
        .sortBy(-_._2)
        .take(5)
      if (similarities.nonEmpty) {
        val row = similarities.flatMap { case (w, s) => Seq(w, f"$s%.6f") }.mkString(",")
        pwNN.println(s"${nw.word},$row")
      }
      if ((index + 1) % 100 == 0) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000
        println(f"  Progress: ${index + 1}%5d / ${normalizedVocab.length}%5d (${elapsed}s elapsed)")
        pwNN.flush()
      }
    }
    pwNN.close()
    println("✓ nearest_neighbors.csv written")

    // Task 2: Word Similarity
    println("\n" + "=" * 80)
    println("Task 2: Word Similarity")
    println("=" * 80)
    val wordMap = vocab.map { case (k, (v, _)) => k -> v }
    val testPairs = Seq(
      ("absolute", "abstract"),
      ("ability", "able"),
      ("academic", "accepted")
    )
    val simWriter = new PrintWriter("word_similarities.csv")
    simWriter.println("word_1,word_2,cosine_similarity")
    testPairs.foreach { case (w1, w2) =>
      if (wordMap.contains(w1) && wordMap.contains(w2)) {
        val norm1 = computeNorm(wordMap(w1))
        val norm2 = computeNorm(wordMap(w2))
        val sim = cosineSimilarity(wordMap(w1), norm1, wordMap(w2), norm2)
        simWriter.println(f"$w1,$w2,$sim%.6f")
        println(f"  $w1%-15s <-> $w2%-15s: $sim%.4f")
      } else {
        println(s"  Missing: $w1 or $w2")
      }
    }
    simWriter.close()
    println("✓ word_similarities.csv written")

    // Task 3: Word Analogies
    println("\n" + "=" * 80)
    println("Task 3: Word Analogies")
    println("=" * 80)
    val analogyWriter = new PrintWriter("word_analogies.csv")
    analogyWriter.println("word_a,word_b,word_c,expected,pred_1,score_1,pred_2,score_2,pred_3,score_3")
    val analogyTests = Seq(("ability", "able", "academic", "accept"))
    analogyTests.foreach { case (a, b, c, expected) =>
      if (Set(a, b, c).forall(wordMap.contains)) {
        val target = add(sub(wordMap(a), wordMap(b)), wordMap(c))
        val targetNorm = computeNorm(target)
        val results = normalizedVocab
          .filterNot(nw => Set(a, b, c).contains(nw.word))
          .map { nw => (nw.word, cosineSimilarity(target, targetNorm, nw.vec, nw.norm)) }
          .sortBy(-_._2)
          .take(3)
        val preds = results.map(_._1).mkString(",")
        val scores = results.map(p => f"${p._2}%.6f").mkString(",")
        analogyWriter.println(s"$a,$b,$c,$expected,$preds,$scores")
        println(s"  $a - $b + $c => ${results.head._1} (expected: $expected)")
      } else println(s"  $a - $b + $c => Missing words")
    }
    analogyWriter.close()
    println("✓ word_analogies.csv written")

    println("\n" + "=" * 80)
    println("Post-processing Complete!")
    println("=" * 80)
    println("Generated files:")
    println("  ✓ vocabulary.csv (full 1024-d vectors)")
    println("  ✓ nearest_neighbors.csv")
    println("  ✓ word_similarities.csv")
    println("  ✓ word_analogies.csv")
    println("=" * 80)
  }
}
