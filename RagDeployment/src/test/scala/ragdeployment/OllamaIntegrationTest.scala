package ragdeployment

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll

/**
 * Integration tests for Ollama API client.
 * Requires Ollama to be running locally.
 *
 * To run: ollama serve
 *         ollama pull mxbai-embed-large
 */
class OllamaIntegrationTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  // Skip tests if Ollama is not available
  private def isOllamaAvailable: Boolean = {
    try {
      val client = new Ollama()
      val result = client.embed(Vector("test"), "mxbai-embed-large")
      client.close()
      result.head.nonEmpty
    } catch {
      case _: Exception => false
    }
  }

  override def beforeAll(): Unit = {
    assume(isOllamaAvailable, "Ollama is not available. Start with: ollama serve")
  }

  "Ollama.embed" should "generate embeddings for single text" in {
    val client = new Ollama()
    try {
      val texts = Vector("Hello world")
      val embeddings = client.embed(texts, "mxbai-embed-large")

      embeddings.size shouldBe 1
      embeddings.head.length should be > 0
      embeddings.head.length shouldBe 1024  // mxbai-embed-large dimension
    } finally {
      client.close()
    }
  }

  it should "generate embeddings for multiple texts" in {
    val client = new Ollama()
    try {
      val texts = Vector("First text", "Second text", "Third text")
      val embeddings = client.embed(texts, "mxbai-embed-large")

      embeddings.size shouldBe texts.size
      embeddings.foreach(_.length shouldBe 1024)
    } finally {
      client.close()
    }
  }

  it should "generate different embeddings for different texts" in {
    val client = new Ollama()
    try {
      val texts = Vector("cat", "dog")
      val embeddings = client.embed(texts, "mxbai-embed-large")

      // Embeddings should be different
      embeddings(0) should not equal embeddings(1)
    } finally {
      client.close()
    }
  }

  it should "generate similar embeddings for similar texts" in {
    val client = new Ollama()
    try {
      val texts = Vector("machine learning", "machine learning algorithms")
      val embeddings = client.embed(texts, "mxbai-embed-large")

      // Calculate cosine similarity
      val norm1 = math.sqrt(embeddings(0).map(x => x * x).sum)
      val norm2 = math.sqrt(embeddings(1).map(x => x * x).sum)
      val dot = embeddings(0).zip(embeddings(1)).map { case (a, b) => a * b }.sum
      val similarity = dot / (norm1 * norm2)

      // Similar texts should have high similarity
      similarity should be > 0.7
    } finally {
      client.close()
    }
  }

  "Ollama.chat" should "generate text responses" in {
    val client = new Ollama()
    try {
      val messages = Vector(
        ChatMessage("user", "What is 2+2?")
      )
      val response = client.chat(messages, "llama3")

      response should not be empty
      response.toLowerCase should include("4")
    } finally {
      client.close()
    }
  }
}