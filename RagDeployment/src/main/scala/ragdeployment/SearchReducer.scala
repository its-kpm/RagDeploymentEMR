package ragdeployment

import org.apache.hadoop.io.{FloatWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.Reducer
import org.slf4j.LoggerFactory
import scala.jdk.CollectionConverters._

/**
 * Reducer for merging search results into global top-k.
 *
 * Receives results sorted by score (negative) and takes top-k.
 *
 * Input: FloatWritable (negative score), Text (JSON result)
 * Output: NullWritable, Text (JSON result)
 */
class SearchReducer extends Reducer[FloatWritable, Text, NullWritable, Text] {

  private val log = LoggerFactory.getLogger(getClass)

  override def reduce(
                       key: FloatWritable,
                       values: java.lang.Iterable[Text],
                       ctx: Reducer[FloatWritable, Text, NullWritable, Text]#Context
                     ): Unit = {

    val k = ctx.getConfiguration.getInt("search.top.k", 5)
    var count = 0

    // Emit top-k results (already sorted by mapper output key)
    values.asScala.foreach { value =>
      if (count < k) {
        ctx.write(NullWritable.get(), value)
        count += 1
      }
    }

    log.debug(s"Emitted $count results for score ${-key.get()}")
  }
}