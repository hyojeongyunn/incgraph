#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "incgraph/index.h"

namespace incgraph {

using KV = std::map<std::string, std::string>;

class Policy {
  public:
    Policy(const KV &kv, int threads);
    virtual ~Policy() = default;
    virtual std::string name() const = 0;
    virtual void setup(Index &) {}
    virtual void insert(Index &idx, const float *vecs, const uint32_t *labels, size_t n);
    virtual void remove(Index &idx, const uint32_t *labels, size_t n);
    virtual void consolidate(Index &idx);
    virtual void maintain(Index &, const float *, size_t, int) {}
    std::string describe() const;
    virtual std::string stats() const { return "{}"; }
    std::string last_round = "{}";
    void check_unused() const;

  protected:
    double getf(const std::string &k, double def);
    long geti(const std::string &k, long def);
    std::string gets(const std::string &k, const std::string &def);
    int threads_;

  private:
    KV kv_;
    std::set<std::string> used_;
    std::vector<std::pair<std::string, std::string>> params_;
};

std::unique_ptr<Policy> make_policy(const std::string &name, const KV &kv, int threads);

class NoopPolicy : public Policy {
  public:
    using Policy::Policy;
    std::string name() const override { return "noop"; }
};

class OODBridgePolicy : public Policy {
  public:
    OODBridgePolicy(const KV &kv, int threads);
    std::string name() const override { return "oodbridge"; }
    void setup(Index &idx) override;
    void insert(Index &idx, const float *vecs, const uint32_t *labels, size_t n) override;
    void remove(Index &idx, const uint32_t *labels, size_t n) override;
    void consolidate(Index &idx) override;
    std::string stats() const override;

  private:
    void calibrate(Index &idx);
    void bridge(Index &idx, const std::vector<uint32_t> &pts, const float *vecs, std::vector<uint32_t> &res,
                std::vector<float> &dist) const;
    uint32_t m_, L_bridge_;
    double q_;
    uint64_t seed_;
    double tau_ = -1;
    std::set<uint32_t> ood_;
    size_t n_id_ = 0, n_ood_ = 0, n_rebridged_ = 0;
    double bridge_s_ = 0;
};

class CoMemberPolicy : public Policy {
  public:
    CoMemberPolicy(const KV &kv, int threads);
    std::string name() const override { return "comember"; }
    void setup(Index &idx) override;
    void maintain(Index &idx, const float *queries, size_t nq, int epoch) override;
    std::string stats() const override;

  private:
    static uint64_t key(uint32_t u, uint32_t v) { return (static_cast<uint64_t>(u) << 32) | v; }
    void drop(Index &idx, uint64_t e);
    uint32_t K_, L_member_;
    float prune_alpha_;
    std::string removal_;
    double decay_, min_score_;
    size_t budget_;
    std::unordered_map<uint64_t, float> score_;
    size_t added_ = 0, removed_ = 0, evicted_ = 0, rounds_ = 0;
    double maintain_s_ = 0;
};

}
