#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include "json.hpp"

namespace psdpu {
using Input = std::array<float,237>;
struct Result {
  std::array<float,4> output{};
  int status=0;
  uint32_t saturation=0;
  bool switched=false;
  uint64_t start_us=0,dispatch_us=0,prep_end_us=0,run_start_us=0,run_end_us=0,end_us=0;
  nlohmann::json rejected_channels=nlohmann::json::array();
};
class LocalDpu {
 public:
  explicit LocalDpu(const std::string& models="");
  ~LocalDpu();
  LocalDpu(const LocalDpu&)=delete;
  LocalDpu& operator=(const LocalDpu&)=delete;
  Result infer(int numeric_id,const Input& raw);
  nlohmann::json identity() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
