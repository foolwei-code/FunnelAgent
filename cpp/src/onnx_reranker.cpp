#include "onnx_reranker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

// ============================================================================
// Timestamped logging helpers
// ============================================================================

namespace {
    std::string timestamp() {
        using clock = std::chrono::system_clock;
        auto now = clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;
        auto t = clock::to_time_t(now);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    const std::string LOG_TAG = "ONNXReranker";
}

void ONNXReranker::log_info(const std::string& tag, const std::string& msg) {
    std::cerr << "[" << timestamp() << "] [INFO] [" << tag << "] " << msg << "\n";
}

void ONNXReranker::log_error(const std::string& tag, const std::string& msg) {
    std::cerr << "[" << timestamp() << "] [ERROR] [" << tag << "] " << msg << "\n";
}

void ONNXReranker::log_warn(const std::string& tag, const std::string& msg) {
    std::cerr << "[" << timestamp() << "] [WARN] [" << tag << "] " << msg << "\n";
}

// ============================================================================
// Constructor
// ============================================================================

ONNXReranker::ONNXReranker(const std::string& model_path,
                           int threads,
                           const TokenizerConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "FunnelRAG-Reranker"),
      session_options_(),
      tokenizer_config_(config),
      model_path_(model_path) {

    log_info(LOG_TAG, "Initializing ONNXReranker...");
    log_info(LOG_TAG, "  Model path: " + model_path_);
    log_info(LOG_TAG, "  Intra-op threads: " + std::to_string(threads));
    log_info(LOG_TAG, "  Max sequence length: " + std::to_string(tokenizer_config_.max_length));

    // --- Validate model file exists ---
    {
        std::ifstream check(model_path);
        if (!check.good()) {
            log_error(LOG_TAG, "Model file not found: " + model_path_);
            throw std::runtime_error("ONNXReranker: model file not found: " + model_path_);
        }
    }

    // --- Configure ONNX session options ---
    session_options_.SetIntraOpNumThreads(threads);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable CPU execution provider (device_id = 0)
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options_, 0);

    // --- Create ONNX session ---
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
        is_initialized_ = true;
    } catch (const Ort::Exception& e) {
        log_error(LOG_TAG, std::string("Failed to create ONNX session: ") + e.what());
        throw std::runtime_error(
            std::string("ONNXReranker: failed to create ONNX session: ") + e.what());
    }

    // --- Cache input/output names ---
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_.GetInputCount();
    input_name_strings_.reserve(num_inputs);
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_name_strings_.emplace_back(name.get());
        input_names_.push_back(input_name_strings_.back().c_str());
    }

    size_t num_outputs = session_.GetOutputCount();
    output_name_strings_.reserve(num_outputs);
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_name_strings_.emplace_back(name.get());
        output_names_.push_back(output_name_strings_.back().c_str());
    }

    // --- Load vocabulary (if path provided) ---
    if (!tokenizer_config_.vocab_path.empty()) {
        load_vocabulary();
    } else {
        log_warn(LOG_TAG, "No vocab_path provided; tokenizer will use whitespace + UNK fallback");
    }

    log_info(LOG_TAG, "ONNXReranker initialized successfully");
    log_info(LOG_TAG, "  Inputs:  " + std::to_string(num_inputs));
    log_info(LOG_TAG, "  Outputs: " + std::to_string(num_outputs));
}

// ============================================================================
// Vocabulary loading
// ============================================================================

void ONNXReranker::load_vocabulary() {
    log_info(LOG_TAG, "Loading vocabulary from: " + tokenizer_config_.vocab_path);

    std::ifstream ifs(tokenizer_config_.vocab_path);
    if (!ifs.good()) {
        log_warn(LOG_TAG, "Vocabulary file not found: " + tokenizer_config_.vocab_path +
                          "; falling back to whitespace tokenization");
        return;
    }

    std::string line;
    int id = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        vocab_[line] = id;
        id_to_token_[id] = line;
        ++id;
    }

    vocab_loaded_ = true;
    log_info(LOG_TAG, "Vocabulary loaded: " + std::to_string(vocab_.size()) + " tokens");

    // Override special token IDs from the loaded vocabulary
    auto it = vocab_.find(tokenizer_config_.cls_token);
    if (it != vocab_.end()) tokenizer_config_.cls_token_id = it->second;

    it = vocab_.find(tokenizer_config_.sep_token);
    if (it != vocab_.end()) tokenizer_config_.sep_token_id = it->second;

    it = vocab_.find(tokenizer_config_.pad_token);
    if (it != vocab_.end()) tokenizer_config_.pad_token_id = it->second;

    it = vocab_.find(tokenizer_config_.unk_token);
    if (it != vocab_.end()) tokenizer_config_.unk_token_id = it->second;
}

// ============================================================================
// Whitespace tokenizer
// ============================================================================

std::vector<std::string> ONNXReranker::whitespace_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// ============================================================================
// WordPiece-like tokenizer
// ============================================================================

std::vector<int> ONNXReranker::tokenize(const std::string& text) const {
    std::string processed = text;
    if (tokenizer_config_.do_lower_case) {
        std::transform(processed.begin(), processed.end(), processed.begin(),
                       [](unsigned char c) { return std::tolower(c); });
    }

    std::vector<int> token_ids;
    auto words = whitespace_tokenize(processed);

    for (const auto& word : words) {
        if (vocab_loaded_) {
            // Try whole word first
            auto it = vocab_.find(word);
            if (it != vocab_.end()) {
                token_ids.push_back(it->second);
                continue;
            }

            // WordPiece: try longest prefix from start, then ##subtokens
            std::string remaining = word;
            bool is_first = true;
            bool all_found = true;

            while (!remaining.empty()) {
                int found_len = 0;
                int found_id = tokenizer_config_.unk_token_id;

                // Search from longest prefix down to 1 char
                for (int len = static_cast<int>(remaining.size()); len > 0; --len) {
                    std::string candidate = remaining.substr(0, len);
                    if (!is_first) {
                        candidate = "##" + candidate;
                    }
                    auto cit = vocab_.find(candidate);
                    if (cit != vocab_.end()) {
                        found_len = len;
                        found_id = cit->second;
                        break;
                    }
                }

                if (found_len == 0) {
                    all_found = false;
                    break;
                }

                token_ids.push_back(found_id);
                remaining = remaining.substr(found_len);
                is_first = false;
            }

            if (!all_found) {
                token_ids.push_back(tokenizer_config_.unk_token_id);
            }
        } else {
            // No vocabulary: use character-level tokenization with UNK fallback
            for (char c : word) {
                std::string char_str(1, c);
                auto it = vocab_.find(char_str);
                if (it != vocab_.end()) {
                    token_ids.push_back(it->second);
                } else {
                    token_ids.push_back(tokenizer_config_.unk_token_id);
                }
            }
        }
    }

    return token_ids;
}

// ============================================================================
// Cross-encoder pair encoding
// ============================================================================

ONNXReranker::EncodedPair ONNXReranker::encode_pair(const std::string& query,
                                                     const std::string& document) const {
    EncodedPair result;
    result.was_truncated = false;

    // Input validation
    if (query.empty() && document.empty()) {
        // Return minimal valid input: [CLS] [SEP] [SEP]
        result.input_ids = {tokenizer_config_.cls_token_id,
                            tokenizer_config_.sep_token_id,
                            tokenizer_config_.sep_token_id};
        result.attention_mask = {1, 1, 1};
        result.token_type_ids = {0, 0, 1};
        result.sequence_length = 3;
        return result;
    }

    // Tokenize query and document separately
    auto query_ids = tokenize(query);
    auto doc_ids   = tokenize(document);

    // Cross-encoder format: [CLS] query [SEP] doc [SEP]
    // Special tokens: CLS(1) + SEP(query-doc boundary) + SEP(trailing) = 3
    const int max_content_len = tokenizer_config_.max_length - 3;

    // Truncation strategy: prioritize query, truncate document first
    int q_len = static_cast<int>(query_ids.size());
    int d_len = static_cast<int>(doc_ids.size());

    if (q_len + d_len > max_content_len) {
        result.was_truncated = true;
        // Keep query up to half of content budget, give remainder to doc
        int query_budget = std::min(q_len, max_content_len / 2);
        int doc_budget   = max_content_len - query_budget;
        // If query is shorter, give the extra budget to doc
        if (q_len < max_content_len / 2) {
            doc_budget = std::min(d_len, max_content_len - q_len);
        }
        query_ids.resize(query_budget);
        doc_ids.resize(doc_budget);
    }

    // Build input_ids: [CLS] query [SEP] doc [SEP]
    result.input_ids.reserve(tokenizer_config_.max_length);
    result.input_ids.push_back(tokenizer_config_.cls_token_id);
    for (int id : query_ids) result.input_ids.push_back(static_cast<int64_t>(id));
    result.input_ids.push_back(tokenizer_config_.sep_token_id);
    for (int id : doc_ids)   result.input_ids.push_back(static_cast<int64_t>(id));
    result.input_ids.push_back(tokenizer_config_.sep_token_id);

    result.sequence_length = static_cast<int>(result.input_ids.size());

    // Build attention_mask: 1 for real tokens, 0 for padding
    // (We pad to max_length so the tensor shape is fixed for the model)
    int seq_len = static_cast<int>(result.input_ids.size());
    result.attention_mask.assign(seq_len, 1);

    // Build token_type_ids: 0 for [CLS] query [SEP], 1 for doc [SEP]
    int sep_pos = 1 + static_cast<int>(query_ids.size()) + 1; // CLS + query + SEP
    result.token_type_ids.assign(sep_pos, 0);
    result.token_type_ids.resize(seq_len, 1);

    // Pad to max_length
    if (seq_len < tokenizer_config_.max_length) {
        int pad_count = tokenizer_config_.max_length - seq_len;
        result.input_ids.assign(result.input_ids.begin(), result.input_ids.begin() + seq_len);
        // Re-extend with padding
        for (int i = 0; i < pad_count; ++i) {
            result.input_ids.push_back(tokenizer_config_.pad_token_id);
            result.attention_mask.push_back(0);
            result.token_type_ids.push_back(0);
        }
    }

    return result;
}

// ============================================================================
// ONNX Inference
// ============================================================================

double ONNXReranker::run_inference(const EncodedPair& encoded) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Create input tensors
    // Shape: [1, max_length] for all three inputs
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(encoded.input_ids.size())};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(3);

    // input_ids tensor (int64)
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(encoded.input_ids.data()),
        encoded.input_ids.size(), input_shape.data(), input_shape.size()));

    // attention_mask tensor (int64)
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(encoded.attention_mask.data()),
        encoded.attention_mask.size(), input_shape.data(), input_shape.size()));

    // token_type_ids tensor (int64)
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(encoded.token_type_ids.data()),
        encoded.token_type_ids.size(), input_shape.data(), input_shape.size()));

    // Run inference
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), input_tensors.data(), input_tensors.size(),
            output_names_.data(), output_names_.size());
    } catch (const Ort::Exception& e) {
        log_error(LOG_TAG, std::string("ONNX inference failed: ") + e.what());
        throw std::runtime_error(
            std::string("ONNXReranker::run_inference: ") + e.what());
    }

    // Extract logit from output tensor
    // Cross-encoder models typically output shape [1, 1] or [1]
    if (output_tensors.empty()) {
        log_error(LOG_TAG, "No output tensors returned from ONNX session");
        throw std::runtime_error("ONNXReranker::run_inference: empty output");
    }

    double logit = 0.0;
    auto& output_tensor = output_tensors[0];
    auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();

    if (type_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = output_tensor.GetTensorData<float>();
        logit = static_cast<double>(data[0]);
    } else if (type_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double* data = output_tensor.GetTensorData<double>();
        logit = data[0];
    } else {
        log_warn(LOG_TAG, "Unexpected output tensor type; attempting float read");
        const float* data = output_tensor.GetTensorData<float>();
        logit = static_cast<double>(data[0]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_inference_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return logit;
}

// ============================================================================
// Sigmoid
// ============================================================================

double ONNXReranker::sigmoid(double logit) {
    // Numerically stable sigmoid
    if (logit >= 0.0) {
        return 1.0 / (1.0 + std::exp(-logit));
    } else {
        double ez = std::exp(logit);
        return ez / (1.0 + ez);
    }
}

// ============================================================================
// score_pair — single (query, doc) scoring
// ============================================================================

double ONNXReranker::score_pair(const std::string& query, const std::string& document) {
    if (!is_initialized_) {
        throw std::runtime_error("ONNXReranker::score_pair: model not initialized");
    }

    auto encoded = encode_pair(query, document);
    double logit = run_inference(encoded);
    return sigmoid(logit);
}

// ============================================================================
// rerank — main API
// ============================================================================

std::vector<RerankItem> ONNXReranker::rerank(const std::string& query,
                                               const std::vector<RerankItem>& items,
                                               int top_k) {
    if (!is_initialized_) {
        throw std::runtime_error("ONNXReranker::rerank: model not initialized");
    }

    if (items.empty()) {
        log_warn(LOG_TAG, "rerank called with empty items");
        return {};
    }

    if (query.empty()) {
        log_warn(LOG_TAG, "rerank called with empty query; returning items with score 0");
        auto result = items;
        for (auto& item : result) item.score = 0.0;
        return result;
    }

    // Score each (query, doc) pair via cross-encoder
    std::vector<RerankItem> scored_items = items;

    for (auto& item : scored_items) {
        if (item.text.empty()) {
            item.score = 0.0;
            continue;
        }
        try {
            auto encoded = encode_pair(query, item.text);
            double logit = run_inference(encoded);
            item.score = sigmoid(logit);
        } catch (const std::exception& e) {
            log_error(LOG_TAG, "Scoring failed for doc_id=" + item.doc_id +
                               ": " + e.what());
            item.score = 0.0;
        }
    }

    // Sort by score descending
    std::sort(scored_items.begin(), scored_items.end(),
              [](const RerankItem& a, const RerankItem& b) {
                  return a.score > b.score;
              });

    // Truncate to top_k
    if (top_k > 0 && static_cast<int>(scored_items.size()) > top_k) {
        scored_items.resize(top_k);
    }

    return scored_items;
}

// ============================================================================
// rerank_batch — batch processing with statistics
// ============================================================================

BatchResult ONNXReranker::rerank_batch(const std::string& query,
                                         const std::vector<RerankItem>& items,
                                         int top_k) {
    BatchResult result;
    result.batch_size = static_cast<int>(items.size());

    if (!is_initialized_) {
        log_error(LOG_TAG, "rerank_batch: model not initialized");
        return result;
    }

    if (items.empty() || query.empty()) {
        log_warn(LOG_TAG, "rerank_batch: empty query or items");
        return result;
    }

    auto total_t0 = std::chrono::high_resolution_clock::now();

    std::vector<RerankItem> scored_items = items;
    double total_tok_ms = 0.0;
    double total_inf_ms = 0.0;
    int num_truncated = 0;

    for (auto& item : scored_items) {
        if (item.text.empty()) {
            item.score = 0.0;
            continue;
        }

        try {
            auto tok_t0 = std::chrono::high_resolution_clock::now();
            auto encoded = encode_pair(query, item.text);
            auto tok_t1 = std::chrono::high_resolution_clock::now();
            total_tok_ms += std::chrono::duration<double, std::milli>(tok_t1 - tok_t0).count();

            if (encoded.was_truncated) ++num_truncated;

            auto inf_t0 = std::chrono::high_resolution_clock::now();
            double logit = run_inference(encoded);
            auto inf_t1 = std::chrono::high_resolution_clock::now();
            total_inf_ms += std::chrono::duration<double, std::milli>(inf_t1 - inf_t0).count();

            item.score = sigmoid(logit);
        } catch (const std::exception& e) {
            log_error(LOG_TAG, "Batch scoring failed for doc_id=" + item.doc_id +
                               ": " + e.what());
            item.score = 0.0;
        }
    }

    // Sort descending by score
    std::sort(scored_items.begin(), scored_items.end(),
              [](const RerankItem& a, const RerankItem& b) {
                  return a.score > b.score;
              });

    if (top_k > 0 && static_cast<int>(scored_items.size()) > top_k) {
        scored_items.resize(top_k);
    }

    auto total_t1 = std::chrono::high_resolution_clock::now();

    result.items = std::move(scored_items);
    result.total_inference_ms    = total_inf_ms;
    result.total_tokenization_ms = total_tok_ms;
    result.num_truncated         = num_truncated;

    double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
    log_info(LOG_TAG, "Batch rerank: " + std::to_string(result.batch_size) +
                      " pairs in " + std::to_string(total_ms) + " ms" +
                      " (tokenize=" + std::to_string(total_tok_ms) +
                      " ms, infer=" + std::to_string(total_inf_ms) + " ms)" +
                      " truncated=" + std::to_string(num_truncated));

    return result;
}

// ============================================================================
// Model introspection
// ============================================================================

std::string ONNXReranker::get_model_info() const {
    std::ostringstream oss;
    oss << "ONNXReranker Model Info\n";
    oss << "  Model path:      " << model_path_ << "\n";
    oss << "  Initialized:     " << (is_initialized_ ? "yes" : "no") << "\n";
    oss << "  Vocab loaded:    " << (vocab_loaded_ ? "yes" : "no") << "\n";
    oss << "  Vocab size:      " << vocab_.size() << "\n";
    oss << "  Max seq length:  " << tokenizer_config_.max_length << "\n";
    oss << "  Do lower case:   " << (tokenizer_config_.do_lower_case ? "yes" : "no") << "\n";
    oss << "  Num inputs:      " << input_names_.size() << "\n";

    for (size_t i = 0; i < input_names_.size(); i++) {
        oss << "    Input[" << i << "]: " << input_name_strings_[i];
        try {
            auto type_info = session_.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            oss << " shape=[";
            for (size_t j = 0; j < shape.size(); j++) {
                if (j > 0) oss << ",";
                oss << shape[j];
            }
            oss << "]";
        } catch (...) {
            oss << " (shape unknown)";
        }
        oss << "\n";
    }

    oss << "  Num outputs:     " << output_names_.size() << "\n";
    for (size_t i = 0; i < output_names_.size(); i++) {
        oss << "    Output[" << i << "]: " << output_name_strings_[i] << "\n";
    }

    oss << "  Last inference:  " << std::fixed << std::setprecision(3)
        << last_inference_ms_ << " ms\n";

    return oss.str();
}

// ============================================================================
// Health check
// ============================================================================

bool ONNXReranker::is_healthy() const {
    return is_initialized_;
}

// ============================================================================
// Last inference time
// ============================================================================

double ONNXReranker::get_last_inference_time() const {
    return last_inference_ms_;
}

// ============================================================================
// Warmup
// ============================================================================

void ONNXReranker::warmup(int warmup_runs) {
    if (!is_initialized_) {
        log_warn(LOG_TAG, "warmup: model not initialized, skipping");
        return;
    }

    log_info(LOG_TAG, "Warming up ONNX session with " +
                      std::to_string(warmup_runs) + " dummy inferences...");

    for (int i = 0; i < warmup_runs; ++i) {
        try {
            auto encoded = encode_pair("warmup query", "warmup document for cold start mitigation");
            (void)run_inference(encoded);
        } catch (const std::exception& e) {
            log_warn(LOG_TAG, "Warmup inference " + std::to_string(i + 1) +
                               " failed: " + e.what());
        }
    }

    log_info(LOG_TAG, "Warmup complete. Last inference: " +
                      std::to_string(last_inference_ms_) + " ms");
}
