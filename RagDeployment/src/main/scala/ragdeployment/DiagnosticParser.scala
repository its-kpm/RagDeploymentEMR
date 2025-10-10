package ragdeployment

import scala.io.Source
import java.io.File

object DiagnosticParser {

  def parseLine(line: String): Option[(String, Int, Int)] = {
    try {
      val parts = line.split("\t", 2)
      if (parts.length != 2) return None

      val value = parts(1)
      val bracketIndex = value.indexOf('[')
      if (bracketIndex == -1) return None

      val firstComma = value.indexOf(',')
      if (firstComma == -1) return None

      val word = value.substring(0, firstComma).trim

      val secondComma = value.indexOf(',', firstComma + 1)
      val frequency = if (secondComma != -1 && secondComma < bracketIndex) {
        value.substring(firstComma + 1, secondComma).trim.toInt
      } else {
        1
      }

      val closeBracketIndex = value.lastIndexOf(']')
      if (closeBracketIndex == -1) return None

      val vecStr = value.substring(bracketIndex + 1, closeBracketIndex)
      val vecLength = vecStr.split(",").length

      Some((word, frequency, vecLength))
    } catch {
      case e: Exception =>
        None
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("Usage: DiagnosticParser <input_file_or_directory>")
      System.exit(1)
    }

    val inputPath = args(0)
    val inputFile = new File(inputPath)

    val filesToProcess = if (inputFile.isDirectory) {
      inputFile.listFiles().filter(_.getName.startsWith("part-r-"))
    } else if (inputFile.isFile) {
      Array(inputFile)
    } else {
      Array.empty[File]
    }

    println("=" * 80)
    println("DIAGNOSTIC: Checking for duplicate words and frequency issues")
    println("=" * 80)

    var totalLines = 0
    var parsedLines = 0
    val wordFrequencies = scala.collection.mutable.Map[String, scala.collection.mutable.ArrayBuffer[Int]]()

    filesToProcess.foreach { file =>
      println(s"\nProcessing file: ${file.getName}")

      Source.fromFile(file).getLines().foreach { line =>
        totalLines += 1

        parseLine(line) match {
          case Some((word, freq, dim)) =>
            parsedLines += 1

            if (!wordFrequencies.contains(word)) {
              wordFrequencies(word) = scala.collection.mutable.ArrayBuffer[Int]()
            }
            wordFrequencies(word) += freq

          case None =>
          // Ignore
        }
      }
    }

    println("\n" + "=" * 80)
    println("RESULTS:")
    println("=" * 80)
    println(s"Total lines read: $totalLines")
    println(s"Successfully parsed: $parsedLines")
    println(s"Unique words: ${wordFrequencies.size}")

    // Check for duplicates
    val duplicates = wordFrequencies.filter(_._2.length > 1)

    if (duplicates.nonEmpty) {
      println("\n" + "!" * 80)
      println("WARNING: DUPLICATE WORDS FOUND!")
      println("!" * 80)
      println(s"Found ${duplicates.size} words that appear multiple times:")

      duplicates.toSeq.sortBy(-_._2.length).take(20).foreach { case (word, freqs) =>
        println(f"  $word%-20s appears ${freqs.length}%3d times with frequencies: ${freqs.mkString(", ")}")
      }

      println("\nThis explains why frequencies are showing as 1!")
      println("The last occurrence of each word is overwriting earlier ones.")
    } else {
      println("\nNo duplicate words found.")
    }

    // Show sample of words with their frequencies
    println("\n" + "=" * 80)
    println("SAMPLE WORDS AND FREQUENCIES:")
    println("=" * 80)

    wordFrequencies.toSeq.sortBy(_._1).take(20).foreach { case (word, freqs) =>
      println(f"  $word%-20s frequency: ${freqs.head}")
    }

    // Check specific words
    println("\n" + "=" * 80)
    println("CHECKING SPECIFIC WORDS:")
    println("=" * 80)

    Seq("absolute", "academic", "accepted").foreach { word =>
      wordFrequencies.get(word) match {
        case Some(freqs) =>
          println(f"  $word%-20s: frequency = ${freqs.head}, appears ${freqs.length} time(s)")
        case None =>
          println(f"  $word%-20s: NOT FOUND")
      }
    }

    println("\n" + "=" * 80)
  }
}