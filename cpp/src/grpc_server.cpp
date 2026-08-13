#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface_builder.h>
#include "onnx_reranker.h"

// Generated protobuf headers
#include "reranker.grpc.pb.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

// ============================================================================
// Server Metrics
// ============================================================================

struct ServerMetrics {
    std::atomic<uint64_t> request_count{0};
    std::atomic<uint64_t> error_count{0};
    std::atomic<double>  total_latency_ms{0.0};

    void record_request(double latency_ms) {
        request_count.fetch_add(1, std::memory_order_relaxed);
        total_latency_ms.fetch_add(latency_ms, std::memory_order_relaxed);
    }

    void record_error() {
        error_count.fetch_add(1, std::memory_order_relaxed);
    }

    std::string summary() const {
        uint64_t cnt = request_count.load(std::memory_order_relaxed);
        uint64_t err = error_count.load(std::memory_order_relaxed);
        double   lat = total_latency_ms.load(std::memory_order_relaxed);
        std::ostringstream oss;
        oss << "requests=" << cnt
            << " errors=" << err
            << " avg_latency_ms=";
        if (cnt > 0) {
            oss << std::fixed << std::setprecision(2) << (lat / cnt);
        } else {
            oss << "N/A";
        }
        return oss.str();
    }
};

static ServerMetrics g_metrics;

// ============================================================================
// Timestamped logging
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

    void log_info(const std::string& msg) {
        std::cout << "[" << timestamp() << "] [INFO] " << msg << std::endl;
    }

    void log_error(const std::string& msg) {
        std::cerr << "[" << timestamp() << "] [ERROR] " << msg << std::endl;
    }

    void log_warn(const std::string& msg) {
        std::cerr << "[" << timestamp() << "] [WARN] " << msg << std::endl;
    }
}

// ============================================================================
// Graceful shutdown
// ============================================================================

static std::atomic<bool> g_shutdown_requested{false};
static grpc::Server* g_server_ptr = nullptr;

void signal_handler(int signum) {
    log_info("Received signal " + std::to_string(signum) + ", initiating graceful shutdown...");
    g_shutdown_requested.store(true, std::memory_order_relaxed);
    if (g_server_ptr) {
        // Shutdown with a 5-second deadline to allow in-flight RPCs to complete
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        g_server_ptr->Shutdown(deadline);
    }
}

void install_signal_handlers() {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
    log_info("Signal handlers installed (SIGINT, SIGTERM)");
}

// ============================================================================
// Environment variable helpers
// ============================================================================

std::string get_env(const std::string& name, const std::string& default_val) {
    const char* val = std::getenv(name.c_str());
    return val ? std::string(val) : default_val;
}

int get_env_int(const std::string& name, int default_val) {
    const char* val = std::getenv(name.c_str());
    if (val) {
        try { return std::stoi(val); } catch (...) {}
    }
    return default_val;
}

// ============================================================================
// RerankerServiceImpl — gRPC service implementation
// ============================================================================

class RerankerServiceImpl final : public reranker::RerankerService::Service {
public:
    explicit RerankerServiceImpl(std::unique_ptr<ONNXReranker> reranker,
                                  int max_concurrent = 16)
        : reranker_(std::move(reranker)),
          max_concurrent_(max_concurrent),
          active_requests_(0) {}

    grpc::Status Rerank(grpc::ServerContext* context,
                        const reranker::RerankRequest* request,
                        reranker::RerankResponse* response) override {
        auto t0 = std::chrono::high_resolution_clock::now();

        // --- Concurrency control ---
        int current = active_requests_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (current > max_concurrent_) {
            active_requests_.fetch_sub(1, std::memory_order_relaxed);
            g_metrics.record_error();
            log_warn("Rerank rejected: max concurrent requests (" +
                     std::to_string(max_concurrent_) + ") exceeded");
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                "Too many concurrent requests");
        }

        // --- Request validation ---
        if (request->query().empty()) {
            active_requests_.fetch_sub(1, std::memory_order_relaxed);
            g_metrics.record_error();
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "Query must not be empty");
        }

        if (request->items_size() == 0) {
            active_requests_.fetch_sub(1, std::memory_order_relaxed);
            g_metrics.record_error();
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "Items must not be empty");
        }

        // --- Health check ---
        if (!reranker_->is_healthy()) {
            active_requests_.fetch_sub(1, std::memory_order_relaxed);
            g_metrics.record_error();
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "Reranker model not initialized");
        }

        // --- Convert protobuf → internal type ---
        std::vector<RerankItem> items;
        items.reserve(request->items_size());
        for (const auto& item : request->items()) {
            if (item.doc_id().empty() || item.text().empty()) {
                log_warn("Skipping item with empty doc_id or text");
                continue;
            }
            items.push_back({item.doc_id(), item.text(), 0.0});
        }

        if (items.empty()) {
            active_requests_.fetch_sub(1, std::memory_order_relaxed);
            g_metrics.record_error();
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "All items had empty doc_id or text");
        }

        // --- Execute reranking ---
        int top_k = request->top_k();
        if (top_k <= 0) top_k = 5;

        std::vector<RerankItem> results;
        try {
            results = reranker_->rerank(request->query(), items, top_k);
        } catch (const std::exception& e) {
            active_requests_.fetch_sub(1, std::memory_order_relaxed);
            g_metrics.record_error();
            log_error(std::string("Rerank failed: ") + e.what());
            return grpc::Status(grpc::StatusCode::INTERNAL,
                                std::string("Rerank inference error: ") + e.what());
        }

        // --- Convert internal type → protobuf ---
        for (const auto& r : results) {
            auto* out = response->add_items();
            out->set_doc_id(r.doc_id);
            out->set_score(r.score);
            out->set_text(r.text);
        }

        // --- Record metrics ---
        active_requests_.fetch_sub(1, std::memory_order_relaxed);
        auto t1 = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_metrics.record_request(latency_ms);

        // Request logging (sample: log every 100th or errors)
        uint64_t req_id = g_metrics.request_count.load(std::memory_order_relaxed);
        if (req_id % 100 == 0 || req_id <= 10) {
            log_info("Rerank #" + std::to_string(req_id) +
                     ": query='" + request->query().substr(0, 50) +
                     (request->query().size() > 50 ? "..." : "") +
                     "' items=" + std::to_string(items.size()) +
                     " top_k=" + std::to_string(top_k) +
                     " results=" + std::to_string(results.size()) +
                     " latency=" + std::to_string(latency_ms) + "ms");
        }

        return grpc::Status::OK;
    }

    /// Return a health summary string for monitoring.
    std::string health_summary() const {
        return g_metrics.summary() +
               " active=" + std::to_string(active_requests_.load(std::memory_order_relaxed));
    }

private:
    std::unique_ptr<ONNXReranker> reranker_;
    int max_concurrent_;
    std::atomic<int> active_requests_;
};

// ============================================================================
// Server configuration from environment
// ============================================================================

struct ServerConfig {
    std::string model_path;
    std::string listen_address;
    int         threads;
    int         max_concurrent;
    int         max_message_size_mb;
    int         keepalive_time_ms;
    int         keepalive_timeout_ms;
    bool        enable_health_service;
    int         warmup_runs;

    ServerConfig()
        : model_path(get_env("RERANKER_MODEL_PATH", "cpp/model/cross_encoder.onnx")),
          listen_address(get_env("RERANKER_LISTEN", "0.0.0.0:50051")),
          threads(get_env_int("RERANKER_THREADS", 4)),
          max_concurrent(get_env_int("RERANKER_MAX_CONCURRENT", 16)),
          max_message_size_mb(get_env_int("RERANKER_MAX_MESSAGE_MB", 64)),
          keepalive_time_ms(get_env_int("RERANKER_KEEPALIVE_MS", 30000)),
          keepalive_timeout_ms(get_env_int("RERANKER_KEEPALIVE_TIMEOUT_MS", 10000)),
          enable_health_service(true),
          warmup_runs(get_env_int("RERANKER_WARMUP_RUNS", 3)) {}

    void log_config() const {
        log_info("Server configuration:");
        log_info("  Model path:          " + model_path);
        log_info("  Listen address:      " + listen_address);
        log_info("  ONNX threads:        " + std::to_string(threads));
        log_info("  Max concurrent:      " + std::to_string(max_concurrent));
        log_info("  Max message size:    " + std::to_string(max_message_size_mb) + " MB");
        log_info("  Keepalive time:      " + std::to_string(keepalive_time_ms) + " ms");
        log_info("  Keepalive timeout:   " + std::to_string(keepalive_timeout_ms) + " ms");
        log_info("  Health service:      " + std::string(enable_health_service ? "enabled" : "disabled"));
        log_info("  Warmup runs:         " + std::to_string(warmup_runs));
    }
};

// ============================================================================
// Build and start the gRPC server
// ============================================================================

std::unique_ptr<grpc::Server> build_and_start_server(const ServerConfig& config,
                                                       RerankerServiceImpl& service) {
    grpc::ServerBuilder builder;

    // --- Channel arguments ---
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS,    config.keepalive_time_ms);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, config.keepalive_timeout_ms);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH,
                               config.max_message_size_mb * 1024 * 1024);
    builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH,
                               config.max_message_size_mb * 1024 * 1024);

    // --- Listening port ---
    builder.AddListeningPort(config.listen_address, grpc::InsecureServerCredentials());

    // --- Register service ---
    builder.RegisterService(&service);

    // --- Health check service ---
    if (config.enable_health_service) {
        grpc::EnableDefaultHealthCheckService(true);
        log_info("gRPC health check service enabled");
    }

    // --- Build ---
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        log_error("Failed to build and start gRPC server");
        return nullptr;
    }

    return server;
}

// ============================================================================
// Periodic metrics reporter
// ============================================================================

void metrics_reporter_thread(RerankerServiceImpl& service,
                              std::atomic<bool>& running) {
    while (running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        if (!running.load(std::memory_order_relaxed)) break;
        log_info("Metrics: " + service.health_summary());
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // --- Configuration ---
    ServerConfig config;

    // Command-line override for model path
    if (argc > 1) {
        config.model_path = argv[1];
    }

    config.log_config();

    // --- Install signal handlers ---
    install_signal_handlers();

    // --- Initialize reranker ---
    log_info("Loading ONNX reranker model...");
    std::unique_ptr<ONNXReranker> reranker;
    try {
        reranker = std::make_unique<ONNXReranker>(config.model_path, config.threads);
    } catch (const std::exception& e) {
        log_error(std::string("Failed to initialize ONNXReranker: ") + e.what());
        return 1;
    }

    if (!reranker->is_healthy()) {
        log_error("Reranker health check failed after initialization");
        return 1;
    }

    log_info("Model loaded successfully");
    log_info(reranker->get_model_info());

    // --- Warmup ---
    if (config.warmup_runs > 0) {
        log_info("Running warmup (" + std::to_string(config.warmup_runs) + " inferences)...");
        reranker->warmup(config.warmup_runs);
        log_info("Warmup complete");
    }

    // --- Create service ---
    RerankerServiceImpl service(std::move(reranker), config.max_concurrent);

    // --- Build and start gRPC server ---
    auto server = build_and_start_server(config, service);
    if (!server) {
        return 1;
    }

    g_server_ptr = server.get();
    log_info("C++ Reranker gRPC Server listening on " + config.listen_address);

    // --- Start metrics reporter thread ---
    std::atomic<bool> metrics_running{true};
    std::thread metrics_thread(metrics_reporter_thread,
                                std::ref(service), std::ref(metrics_running));

    // --- Wait with periodic shutdown check ---
    // Instead of infinite Wait(), poll for shutdown every second
    while (!g_shutdown_requested.load(std::memory_order_relaxed)) {
        auto deadline = gpr_time_add(
            gpr_now(GPR_CLOCK_REALTIME),
            gpr_time_from_millis(1000, GPR_TIMESPAN));
        server->Wait(deadline);
    }

    // --- Cleanup ---
    log_info("Shutting down...");
    metrics_running.store(false, std::memory_order_relaxed);
    if (metrics_thread.joinable()) {
        metrics_thread.join();
    }

    log_info("Final metrics: " + service.health_summary());
    log_info("Server shutdown complete");

    g_server_ptr = nullptr;
    return 0;
}
