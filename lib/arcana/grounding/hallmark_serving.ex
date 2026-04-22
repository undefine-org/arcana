defmodule Arcana.Grounding.HallmarkServing do
  @moduledoc """
  Lazy-loaded NLI serving for hallucination detection via Hallmark.

  Uses Hallmark (Vectara HHEM model via Bumblebee) to score each sentence
  in the answer against the retrieved chunks. Runs one NLI pair per
  (sentence, chunk) and takes the max score per sentence as the final
  faithfulness score. Sentences scoring below the threshold are marked
  as hallucinated.

  ## Why per-chunk instead of concat-all

  HHEM's ModernBERT has a fixed input window. Concatenating all chunks
  into one context works for small chunk sets (Advanced RAG's 5-10
  chunks) but silently truncates once the pipeline starts accumulating
  more (Pipeline with decompose + rerank hits 20+ easily). Truncation
  drops the tail chunks from the context the NLI sees, so sentences
  whose evidence lived in those tail chunks get flagged as
  hallucinations even when the chunks were retrieved correctly.

  Per-chunk scoring sidesteps truncation entirely. Each chunk fits
  comfortably on its own, and `Hallmark.score_batch/2` sends the whole
  (sentence × chunk) grid through the NLI model as one batched forward
  pass, so the wall-clock cost stays reasonable.

  The model is downloaded automatically on first use via Bumblebee.
  """

  use GenServer

  alias Arcana.Grounding.{Attribution, Result}

  @default_threshold 0.5

  # Client API

  @doc """
  Runs grounding analysis on the given answer against the context chunks.

  Starts the serving if not already running, then runs inference.
  Returns `{:ok, %Arcana.Grounding.Result{}}` or `{:error, reason}`.
  """
  def run(question, chunks, answer, opts \\ []) do
    ensure_started(opts)

    :telemetry.span([:arcana, :grounding], %{question: question}, fn ->
      result = GenServer.call(__MODULE__, {:run, question, chunks, answer, opts}, :infinity)

      metadata =
        case result do
          {:ok, %Result{score: score}} -> %{score: score}
          {:error, _} -> %{}
        end

      {result, metadata}
    end)
  end

  @doc """
  Ensures the serving is started. Called automatically by `run/4`.
  """
  def ensure_started(opts \\ []) do
    case Process.whereis(__MODULE__) do
      nil -> start_serving(opts)
      _pid -> :ok
    end
  end

  @doc """
  Checks if the serving is currently running.
  """
  def running? do
    Process.whereis(__MODULE__) != nil
  end

  # Server callbacks

  @impl true
  def init(opts) do
    {:ok, %{model: nil, opts: opts}, {:continue, :load}}
  end

  @impl true
  def handle_continue(:load, %{opts: _opts} = state) do
    # No compiler arg. Hallmark loads weights via Bumblebee using the
    # global `Nx.default_backend/0`, and predict_batch runs Axon.predict
    # using `Nx.Defn.default_options/0`. As long as the host app sets
    # BOTH to a matching pair (e.g. EMLX.Backend + EMLX), everything
    # stays on one backend end-to-end and the classifier head matmul
    # doesn't crash on a cross-backend mix.
    #
    # The right place to declare which backend / compiler to use is
    # the host app's startup code, not on a per-Hallmark-load basis.
    # See guides/dashboard.md for the recommended setup.
    {:ok, model} = Hallmark.load()
    {:noreply, %{state | model: model}}
  end

  @impl true
  def handle_call({:run, question, chunks, answer, opts}, _from, state) do
    result = do_inference(state, question, chunks, answer, opts)
    {:reply, result, state}
  end

  # Private

  defp start_serving(opts) do
    :global.trans({__MODULE__, :start}, fn ->
      case Process.whereis(__MODULE__) do
        nil ->
          case GenServer.start_link(__MODULE__, opts, name: __MODULE__) do
            {:ok, _pid} -> :ok
            {:error, {:already_started, _pid}} -> :ok
          end

        _pid ->
          :ok
      end
    end)
  end

  # Hard cap on pairs per score_batch call to stay under Metal/EMLX's
  # per-buffer allocation limit. Doctor-who chunks run 500-1500 chars
  # each, so even 32 pairs can push past 14GB GPU buffer on Apple
  # Silicon. 8 is conservative but reliable; the NLI forward pass per
  # sub-batch is still vectorized across its pairs, so throughput stays
  # reasonable (for 120 pairs that's 15 forward passes, each ~150ms).
  @max_pairs_per_call 8

  defp do_inference(state, _question, chunks, answer, opts) do
    %{model: model} = state
    threshold = Keyword.get(opts, :threshold, @default_threshold)

    sentences = split_sentences(answer)
    chunk_texts = Enum.map(chunks, &chunk_text/1)

    scored_sentences =
      case chunk_texts do
        [] ->
          # No chunks to score against: every sentence is unsupported.
          Enum.map(sentences, &{&1, 0.0})

        _ ->
          # Full (sentence, chunk) grid as NLI pairs, chunked into
          # sub-batches that fit in one forward pass.
          pairs =
            for {sentence_text, _, _} <- sentences,
                chunk_text <- chunk_texts do
              {chunk_text, sentence_text}
            end

          flat_scores = batched_score(model, pairs)

          # Reassemble: the flat list is sentences × chunks, in that
          # order. Group by chunk count and take max per sentence.
          chunks_per_sentence = length(chunk_texts)

          sentence_score_groups =
            Enum.chunk_every(flat_scores, chunks_per_sentence)

          sentences
          |> Enum.zip(sentence_score_groups)
          |> Enum.map(fn {sentence, scores} ->
            {sentence, Enum.max(scores, fn -> 0.0 end)}
          end)
      end

    hallucinated_spans =
      scored_sentences
      |> Enum.filter(fn {_sentence, score} -> score < threshold end)
      |> Enum.map(fn {{text, start, stop}, score} ->
        %{text: text, start: start, end: stop, score: 1.0 - score}
      end)
      |> Attribution.attribute(chunks)

    faithful_spans =
      scored_sentences
      |> Enum.filter(fn {_sentence, score} -> score >= threshold end)
      |> Enum.map(fn {{text, start, stop}, score} ->
        %{text: text, start: start, end: stop, score: score}
      end)
      |> Attribution.attribute(chunks)

    total_weight = sentences |> Enum.map(fn {text, _, _} -> byte_size(text) end) |> Enum.sum()

    score =
      if total_weight > 0 do
        scored_sentences
        |> Enum.map(fn {{text, _, _}, s} -> byte_size(text) * s end)
        |> Enum.sum()
        |> Kernel./(total_weight)
      else
        1.0
      end

    {:ok,
     %Result{
       score: score,
       hallucinated_spans: hallucinated_spans,
       faithful_spans: faithful_spans,
       token_labels: nil
     }}
  end

  defp chunk_text(%{text: text}) when is_binary(text), do: text
  defp chunk_text(%{"text" => text}) when is_binary(text), do: text
  defp chunk_text(_), do: ""

  defp batched_score(model, pairs) do
    pairs
    |> Enum.chunk_every(@max_pairs_per_call)
    |> Enum.flat_map(fn batch ->
      {:ok, scores} = Hallmark.score_batch(model, batch)
      scores
    end)
  end

  @doc false
  def split_sentences(text) do
    # Split on sentence-ending punctuation followed by whitespace or end of string.
    # Keeps the punctuation with the sentence.
    regex = ~r/(?<=[.!?])\s+/

    parts = Regex.split(regex, text, include_captures: true)

    {sentences, _} =
      Enum.reduce(parts, {[], 0}, fn part, {acc, offset} ->
        part_bytes = byte_size(part)

        if String.trim(part) == "" do
          {acc, offset + part_bytes}
        else
          {[{part, offset, offset + part_bytes} | acc], offset + part_bytes}
        end
      end)

    Enum.reverse(sentences)
  end
end
