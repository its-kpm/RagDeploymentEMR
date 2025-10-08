package ragdeployment

import org.slf4j.LoggerFactory
import java.io.{File, PrintWriter}
import scala.io.Source

/**
 * Post-processing utility for word embeddings analysis.
 *
 * Performs three main semantic analysis tasks:
 * 1. Semantic Neighbors: Find top-k nearest words by cosine similarity
 * 2. Word Analogies: Solve relationships like "king - man + woman ≈ queen"
 * 3. Word Similarity: Compute cosine similarity between word pairs
 *
 * Input: MapReduce output from WordStatsReducer (tab-separated: word\tword,frequency,[embedding])
 * Output: CSV files with analysis results
 */
object PostProcessEmbeddings {

  private val log = LoggerFactory.getLogger(getClass)

  /**
   * Stores word with its embedding vector and pre-computed norm.
   * Pre-computing norms avoids redundant calculations during similarity comparisons.
   *
   * @param word The word string
   * @param vec The embedding vector
   * @param norm The L2 norm of the vector
   */
  case class NormalizedWord(word: String, vec: Array[Float], norm: Float)

  /**
   * Compute cosine similarity between two vectors using pre-computed norms.
   * Optimized with while loop for better performance on large vocabularies.
   *
   * Formula: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
   *
   * @param v1 First vector
   * @param norm1 Pre-computed norm of first vector
   * @param v2 Second vector
   * @param norm2 Pre-computed norm of second vector
   * @return Cosine similarity score (0 to 1)
   */
  def cosineSimilarity(v1: Array[Float], norm1: Float, v2: Array[Float], norm2: Float): Double = {
    // Handle zero vectors
    if (norm1 == 0 || norm2 == 0) return 0.0

    // Compute dot product using while loop (faster than functional operations)
    var dot = 0.0f
    var i = 0
    while (i < v1.length) {
      dot += v1(i) * v2(i)
      i += 1
    }

    // Return normalized similarity
    dot / (norm1 * norm2)
  }

  /**
   * Compute L2 norm (magnitude) of a vector.
   * Formula: ||v|| = sqrt(v1² + v2² + ... + vn²)
   *
   * @param v Input vector
   * @return L2 norm
   */
  def computeNorm(v: Array[Float]): Float = {
    var sum = 0.0f
    var i = 0
    while (i < v.length) {
      sum += v(i) * v(i)
      i += 1
    }
    math.sqrt(sum).toFloat
  }

  /**
   * Element-wise vector addition.
   * Used for word analogies: king - man + woman
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @return Sum vector
   */
  def add(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    val result = new Array[Float](v1.length)
    var i = 0
    while (i < v1.length) {
      result(i) = v1(i) + v2(i)
      i += 1
    }
    result
  }

  /**
   * Element-wise vector subtraction.
   * Used for word analogies: king - man + woman
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @return Difference vector
   */
  def sub(v1: Array[Float], v2: Array[Float]): Array[Float] = {
    val result = new Array[Float](v1.length)
    var i = 0
    while (i < v1.length) {
      result(i) = v1(i) - v2(i)
      i += 1
    }
    result
  }

  /**
   * Main post-processing pipeline.
   *
   * @param inputPath Path to MapReduce output file(s) or directory
   */
  def run(inputPath: String): Unit = {
    log.info("=" * 80)
    log.info("Starting Word Embedding Post-Processing")
    log.info("=" * 80)
    log.info(s"Input path: $inputPath")

    // Determine input files (directory or single file)
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
      log.error("No valid input files found")
      println("No valid input files found.")
      System.exit(1)
    }
    log.info(s"Found ${filesToProcess.length} file(s) to process")

    // Parse vocabulary from MapReduce output
    log.info("Parsing vocabulary from input files...")
    val vocab = filesToProcess.flatMap { file =>
      log.debug(s"Processing file: ${file.getName}")

      Source.fromFile(file).getLines()
        .filter(line => line != null && line.trim.nonEmpty && line.contains("\t"))
        .flatMap { line =>
          // Parse tab-separated format: word<TAB>word,frequency,[embedding]
          val parts = line.split("\t", 2)
          if (parts.length == 2) {
            // Parse CSV format: word,frequency,[vec1,vec2,...]
            val csvParts = parts(1).split(",", 3)
            if (csvParts.length == 3) {
              val word = csvParts(0).trim
              val vecStr = csvParts(2).replaceAll("[\\[\\]]", "")
              try {
                // Parse embedding vector
                val vec = vecStr.split(",").filter(_.nonEmpty).map(_.trim.toFloat)
                Some(word -> vec)
              } catch {
                case e: Exception =>
                  log.warn(s"Failed to parse embedding for word: $word - ${e.getMessage}")
                  None
              }
            } else None
          } else None
        }
    }.toMap

    log.info(s"Successfully loaded ${vocab.size} word embeddings")
    println(s"Loaded ${vocab.size} embeddings.")

    if (vocab.isEmpty) {
      log.error("No valid embeddings loaded - check input format")
      println("No valid embeddings loaded. Check file format.")
      System.exit(1)
    }

    // ========================================================================
    // Pre-compute vector norms (optimization)
    // ========================================================================
    log.info("Pre-computing vector norms for efficient similarity calculations...")
    val normalizedVocab = vocab.map { case (word, vec) =>
      NormalizedWord(word, vec, computeNorm(vec))
    }.toArray
    log.info(s"Pre-computed norms for ${normalizedVocab.length} words")

    // ========================================================================
    // Export vocabulary statistics
    // ========================================================================
    log.info("Exporting vocabulary.csv...")
    println("Exporting vocabulary.csv...")
    val vocabWriter = new PrintWriter("vocabulary.csv")
    vocabWriter.println("word,token_id,frequency,embedding_dim")
    normalizedVocab.sortBy(_.word).zipWithIndex.foreach { case (nw, idx) =>
      vocabWriter.println(s"${nw.word},$idx,1,${nw.vec.length}")
    }
    vocabWriter.close()
    log.info("vocabulary.csv written successfully")

    // ========================================================================
    // Task 1: Semantic Neighbors (Nearest Neighbors)
    // ========================================================================
    log.info("=" * 80)
    log.info("Task 1: Computing Semantic Neighbors")
    log.info("=" * 80)
    println(s"Computing nearest neighbors for ${normalizedVocab.length} words...")

    val pwNN = new PrintWriter("nearest_neighbors.csv")
    pwNN.println("word,neighbor_1,sim_1,neighbor_2,sim_2,neighbor_3,sim_3,neighbor_4,sim_4,neighbor_5,sim_5")

    var processed = 0
    val startTime = System.currentTimeMillis()

    normalizedVocab.foreach { nw =>
      // Find top 5 most similar words
      val similarities = normalizedVocab
        .filter(_.word != nw.word)
        .map { other =>
          (other.word, cosineSimilarity(nw.vec, nw.norm, other.vec, other.norm))
        }
        .sortBy(-_._2)
        .take(5)

      // Write to CSV
      val row = similarities.flatMap { case (w, s) => Seq(w, f"$s%.6f") }.mkString(",")
      pwNN.println(s"${nw.word},$row")

      processed += 1
      // Progress logging every 500 words
      if (processed % 500 == 0) {
        val elapsed = (System.currentTimeMillis() - startTime) / 1000
        log.info(s"Progress: $processed / ${normalizedVocab.length} (${elapsed}s elapsed)")
        println(s"  Progress: $processed / ${normalizedVocab.length}")
      }
    }
    pwNN.close()

    val totalTime = (System.currentTimeMillis() - startTime) / 1000
    log.info(s"Nearest neighbors computed in ${totalTime}s")
    println("Nearest neighbors written to nearest_neighbors.csv")

    // Show example neighbors
    println("\nExample Semantic Neighbors:")
    log.info("Example semantic neighbors:")
    normalizedVocab.take(3).foreach { nw =>
      val top3 = normalizedVocab
        .filter(_.word != nw.word)
        .map { other => (other.word, cosineSimilarity(nw.vec, nw.norm, other.vec, other.norm)) }
        .sortBy(-_._2)
        .take(3)
      val example = s"${nw.word}: ${top3.map { case (w, s) => f"$w($s%.3f)" }.mkString(", ")}"
      println(s"  $example")
      log.info(s"  $example")
    }

    // ========================================================================
    // Task 2: Word Similarity
    // ========================================================================
    log.info("=" * 80)
    log.info("Task 2: Computing Word Similarities")
    log.info("=" * 80)
    println("\nWord Similarity:")

    val wordMap = vocab
    val testPairs = Seq(
      ("unity", "unify"),
      ("sweet", "nice"),
      ("sword", "weapon"),
      ("eye", "sight"),
      ("unhappiness", "depressed")
    )
    log.info(s"Testing ${testPairs.size} word pairs")

    val simWriter = new PrintWriter("word_similarities.csv")
    simWriter.println("word_1,word_2,cosine_similarity")

    testPairs.foreach { case (w1, w2) =>
      if (wordMap.contains(w1) && wordMap.contains(w2)) {
        val norm1 = computeNorm(wordMap(w1))
        val norm2 = computeNorm(wordMap(w2))
        val sim = cosineSimilarity(wordMap(w1), norm1, wordMap(w2), norm2)
        simWriter.println(f"$w1,$w2,$sim%.6f")
        val result = f"$w1%-15s <-> $w2%-15s: $sim%.4f"
        println(s"  $result")
        log.info(s"  $result")
      } else {
        val missing = s"Missing: $w1 or $w2"
        println(s"  $missing")
        log.warn(s"  $missing")
      }
    }
    simWriter.close()
    log.info("word_similarities.csv written successfully")

    // ========================================================================
    // Task 3: Word Analogies
    // ========================================================================
    log.info("=" * 80)
    log.info("Task 3: Computing Word Analogies")
    log.info("=" * 80)
    println("\nWord Analogies:")

    val analogyWriter = new PrintWriter("word_analogies.csv")
    analogyWriter.println("word_a,word_b,word_c,expected,pred_1,score_1,pred_2,score_2,pred_3,score_3")

    val analogyTests = Seq(
      ("king", "man", "woman", "queen"),
      ("city", "country", "paris", "france"),
      ("day", "night", "summer", "winter"),
      ("love", "hate", "good", "bad")
    )
    log.info(s"Testing ${analogyTests.size} analogies")

    analogyTests.foreach { case (a, b, c, expected) =>
      if (Set(a, b, c).forall(wordMap.contains)) {
        // Compute analogy: a - b + c ≈ ?
        log.debug(s"Computing analogy: $a - $b + $c ≈ $expected")
        val target = add(sub(wordMap(a), wordMap(b)), wordMap(c))
        val targetNorm = computeNorm(target)

        // Find closest words to target vector
        val results = normalizedVocab
          .filterNot(nw => Set(a, b, c).contains(nw.word))
          .map { nw => (nw.word, cosineSimilarity(target, targetNorm, nw.vec, nw.norm)) }
          .sortBy(-_._2)
          .take(3)

        // Write results
        val preds = results.map(_._1).mkString(",")
        val scores = results.map(p => f"${p._2}%.6f").mkString(",")
        analogyWriter.println(s"$a,$b,$c,$expected,$preds,$scores")

        val result = s"$a - $b + $c => ${results.head._1} (expected: $expected)"
        println(s"  $result")
        log.info(s"  $result")

        // Log accuracy
        if (results.head._1 == expected) {
          log.info(s"    ✓ Correct prediction!")
        }
      } else {
        val missing = s"$a - $b + $c => Missing words in vocabulary"
        println(s"  $missing")
        log.warn(s"  $missing")
      }
    }
    analogyWriter.close()
    log.info("word_analogies.csv written successfully")

    // ========================================================================
    // Summary
    // ========================================================================
    log.info("=" * 80)
    log.info("Post-processing Complete")
    log.info("=" * 80)
    println("\nPost-processing complete.")
    println("Generated files:")
    println("  - vocabulary.csv")
    println("  - nearest_neighbors.csv")
    println("  - word_similarities.csv")
    println("  - word_analogies.csv")

    log.info("Generated output files:")
    log.info("  ✓ vocabulary.csv")
    log.info("  ✓ nearest_neighbors.csv")
    log.info("  ✓ word_similarities.csv")
    log.info("  ✓ word_analogies.csv")
    log.info("=" * 80)
  }
}