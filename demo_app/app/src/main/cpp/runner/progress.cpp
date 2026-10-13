#include "runner/progress.h"

#include <algorithm>

namespace edgeslm {

using nlohmann::json;

GenProgress& GenProgress::instance() {
    static GenProgress g;
    return g;
}

void GenProgress::begin(int max_new, int window, size_t prompt_len) {
    std::lock_guard<std::mutex> lk(m_);
    running_ = true;
    done_ = false;
    error_.clear();
    max_new_ = max_new;
    window_ = window;
    prompt_len_ = prompt_len;
    ids_.clear();
    step_ms_.clear();
    ort_ms_.clear();
    ++rev_;
}

void GenProgress::push(int64_t id, double step_ms, double ort_ms) {
    std::lock_guard<std::mutex> lk(m_);
    ids_.push_back(id);
    step_ms_.push_back(step_ms);
    ort_ms_.push_back(ort_ms);
    ++rev_;
}

namespace {
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

void GenProgress::finish(const std::string& error) {
    std::lock_guard<std::mutex> lk(m_);
    running_ = false;
    done_ = true;
    error_ = error;
    ++rev_;
}

json GenProgress::snapshot() const {
    std::lock_guard<std::mutex> lk(m_);
    json j;
    j["running"] = running_;
    j["done"] = done_;
    j["rev"] = rev_;
    j["max_new"] = max_new_;
    j["window"] = window_;
    j["prompt_len"] = prompt_len_;
    j["ids"] = ids_;
    j["n"] = ids_.size();
    if (!error_.empty()) j["error"] = error_;

    if (!step_ms_.empty()) {
        // 中央値を出す（平均は初回の外れ値に引きずられる。基本設計① 5章と同じ理由）。
        const double med = median_of(step_ms_);
        j["last_ms"] = step_ms_.back();
        j["median_ms"] = med;                        // 1トークンの実時間
        j["median_ort_ms"] = median_of(ort_ms_);     // うち OrtRun
        j["tokens_per_sec"] = (med > 0.0) ? (1000.0 / med) : 0.0;
    }
    return j;
}

}  // namespace edgeslm
