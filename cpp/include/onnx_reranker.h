#pragma once

#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// FunnelRAG Cross-Encoder Reranker — ONNX Runtime Backend
//
// Thread safety: This class is NOT thread-safe. Create one instance per
// thread, or wrap calls with external synchronization. The underlying ONNX
// session can be shared across threads if constructed with thread-safe
// options, but the tokenizer state and inference timers are not protected.
// ============================================================================

// ---------------------------------------------------------------------------
// Configuration structs
// ---------------------------------------------------------------------------

/// Tokenizer configuration for the cross-encoder model.
struct TokenizerConfig {
    std::string vocab_path;       ///< Path to vocab.txt (one token per line)
    int max_length      = 512;    ///< Maximum sequence length (input_ids dim)
    bool do_lower_case  = true;   ///< Whether to lowercase input text
    std::string cls_token   = "[CLS]";
    std::string sep_token   = "[SEP]";
    std::string pad_token   = "[PAD]";
    std::string unk_token   = "[UNK]";
    int cls_token_id    = 101;
    int sep_token_id    = 102;
    int pad_token_id    = 0;
    int unk_token_id    = 100;
};

/// Result of a single rerank inference call, including timing metadata.
struct InferenceResult {
    double score_ms           = 0.0;   ///< Inference wall-clock time in ms
    double tokenization_ms    = 0.0;   ///< Tokenization time in ms
    int    input_length       = 0;     ///< Number of tokens after truncation
    bool   truncated          = false;  ///< Whether input was truncated
};

/// Result of a batch rerank operation.
struct BatchResult {
    std::vector<RerankItem> items;         ///< Reranked items sorted by score desc
    double total_inference_ms  = 0.0;      ///< Total inference time for all pairs
    double total_tokenization_ms = 0.0;    ///< Total tokenization time
    int    batch_size         = 0;         ///< Number of (query, doc) pairs processed
    int    num_truncated      = 0;         ///< Number of inputs that were truncated
};

// ---------------------------------------------------------------------------
// Core data struct (kept for backward compatibility)
// ---------------------------------------------------------------------------

struct RerankItem {
    std::string doc_id;
    std::string text;
    double score = 0.0;
};

// ---------------------------------------------------------------------------
// ONNXReranker — Cross-Encoder reranking via ONNX Runtime
// ---------------------------------------------------------------------------

class ONNXReranker {
public:
    // --- Construction / Destruction ----------------------------------------

    /// Construct a reranker from an ONNX model file.
    /// @param model_path   Path to the cross-encoder ONNX model
    /// @param threads      Number of intra-op threads for ONNX Runtime
    /// @param config       Tokenizer configuration (vocab, max_length, etc.)
    /// @throws std::runtime_error if the model file cannot be loaded
    explicit ONNXReranker(const std::string& model_path,
                           int threads = 4,
                           const TokenizerConfig& config = TokenizerConfig());

    ~ONNXReranker() = default;

    // --- Move semantics ----------------------------------------------------
    // Copy is deleted (ONNX session is non-copyable); move is enabled.

    ONNXReranker(const ONNXReranker&)            = delete;
    ONNXReranker& operator=(const ONNXReranker&) = delete;

    ONNXReranker(ONNXReranker&&) noexcept            = default;
    ONNXReranker& operator=(ONNXReranker&&) noexcept = default;

    // --- Reranking API -----------------------------------------------------

    /// Rerank a list of documents against a query, returning top_k results.
    /// @param query   The search query
    /// @param items   Candidate documents with doc_id and text filled
    /// @param top_k   Number of top results to return (0 = all)
    /// @return Items sorted by cross-encoder score descending
    std::vector<RerankItem> rerank(const std::string& query,
                                    const std::vector<RerankItem>& items,
                                    int top_k = 5);

    /// Rerank a list of documents with detailed batch statistics.
    /// @param query   The search query
    /// @param items   Candidate documents
    /// @param top_k   Number of top results to return (0 = all)
    /// @return BatchResult with sorted items and timing/truncation stats
    BatchResult rerank_batch(const std::string& query,
                              const std::vector<RerankItem>& items,
                              int top_k = 5);

    /// Score a single (query, document) pair.
    /// @return Sigmoid-normalized relevance score in [0, 1]
    double score_pair(const std::string& query, const std::string& document);

    // --- Model introspection -----------------------------------------------

    /// Return a human-readable string with model metadata.
    /// Includes input/output names, tensor shapes, and provider info.
    std::string get_model_info() const;

    /// Return true if the ONNX session is initialized and the model loaded.
    bool is_healthy() const;

    /// Return the wall-clock time of the last single inference call in ms.
    double get_last_inference_time() const;

    // --- Warmup ------------------------------------------------------------

    /// Run a dummy inference to warm up the ONNX session (JIT compilation,
    /// memory allocation, etc.). Call this after construction before serving
    /// traffic to avoid cold-start latency spikes.
    /// @param warmup_runs  Number of dummy inferences to run
    void warmup(int warmup_runs = 3);

private:
    // --- ONNX Runtime members ---------------------------------------------
    Ort::Env            env_;
    Ort::Session        session_;
    Ort::SessionOptions session_options_;

    // Cached I/O name allocation (must outlive session calls)
    std::vector<std::string>    input_name_strings_;
    std::vector<std::string>    output_name_strings_;
    std::vector<const char*>    input_names_;
    std::vector<const char*>    output_names_;

    // --- Configuration & state ---------------------------------------------
    TokenizerConfig     tokenizer_config_;
    std::string         model_path_;
    double              last_inference_ms_  = 0.0;
    bool                is_initialized_     = false;

    // --- Tokenizer state ---------------------------------------------------
    /// Vocabulary: token string -> token id
    std::unordered_map<std::string, int> vocab_;
    /// Reverse vocabulary: token id -> token string (for debugging)
    std::unordered_map<int, std::string> id_to_token_;
    bool vocab_loaded_ = false;

    // --- Internal methods --------------------------------------------------

    /// Load vocabulary from tokenizer_config_.vocab_path.
    void load_vocabulary();

    /// Tokenize a single piece of text into token IDs.
    /// Uses basic whitespace + subword splitting (WordPiece-like).
    std::vector<int> tokenize(const std::string& text) const;

    /// Build cross-encoder input: [CLS] query_tokens [SEP] doc_tokens [SEP]
    /// Returns (input_ids, attention_mask, token_type_ids) with padding to
    /// max_length if needed, and truncation if exceeded.
    struct EncodedPair {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids;
        int sequence_length;
        bool was_truncated;
    };
    EncodedPair encode_pair(const std::string& query,
                             const std::string& document) const;

    /// Run ONNX inference on a single encoded pair.
    /// @return Raw logit from the model's output tensor
    double run_inference(const EncodedPair& encoded);

    /// Apply sigmoid: 1 / (1 + exp(-logit))
    static double sigmoid(double logit);

    /// Basic whitespace tokenizer that splits on spaces and punctuation.
    static std::vector<std::string> whitespace_tokenize(const std::string& text);

    /// Simple timestamped logger (writes to stderr).
    static void log_info(const std::string& tag, const std::string& msg);
    static void log_error(const std::string& tag, const std::string& msg);
    static void log_warn(const std::string& tag, const std::string& msg);
};
