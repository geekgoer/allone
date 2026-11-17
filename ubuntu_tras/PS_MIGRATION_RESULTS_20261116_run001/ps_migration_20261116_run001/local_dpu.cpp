// Reuse the frozen Pool::infer implementation byte-for-byte in this TU.
// The include path points to the SHA-verified handoff package, not an old pool.
#define main frozen_pool_runner_main
#include "pool237_runner.cc"
#undef main
#include "local_dpu.hpp"

namespace psdpu {
struct LocalDpu::Impl {
  Args args;
  std::unique_ptr<Pool> pool;
  explicit Impl(const std::string& names) {
    args.pool=names;
    pool.reset(new Pool(args));
  }
};
LocalDpu::LocalDpu(const std::string& models):impl_(new Impl(models)) {}
LocalDpu::~LocalDpu()=default;
Result LocalDpu::infer(int numeric_id,const Input& raw) {
  int index=impl_->pool->find(uint32_t(numeric_id));
  if(index<0) throw std::runtime_error("model is not resident: "+std::to_string(numeric_id));
  auto r=impl_->pool->infer(index,raw,false);
  Result out;
  out.output=r.y; out.status=r.status; out.saturation=r.saturation; out.switched=r.switched;
  out.start_us=r.start; out.dispatch_us=r.dispatch; out.prep_end_us=r.prep_end;
  out.run_start_us=r.run_start; out.run_end_us=r.run_end; out.end_us=r.end;
  if(r.status==3) {
    const auto& task=impl_->pool->tasks.at(index);
    for(size_t c=0;c<raw.size();++c) {
      const float normalized=(raw[c]-kXMean[c])/kXStd[c];
      const double scaled=double(normalized*task.in_scale);
      if(!std::isfinite(scaled)||scaled>=128.||scaled<=-129.)
        out.rejected_channels.push_back({{"index",c},{"raw",raw[c]},{"normalized",normalized},{"scaled",scaled}});
    }
  }
  return out;
}
nlohmann::json LocalDpu::identity() const {
  nlohmann::json models=nlohmann::json::array();
  for(const auto& t:impl_->pool->tasks)
    models.push_back({{"id",t.def->numeric_id},{"model",t.def->id},{"kernel",t.def->kernel},
      {"input_scale",t.in_scale},{"output_scale",t.out_scale},{"input_size",240},{"output_size",4}});
  return {{"models",models},{"normalization_sha256",kNormalizationSHA},
          {"wire_tag_hex","9253dd1fc87f8d7f"},{"backend","PS local DNNDK; frozen Pool::infer"},
          {"run_time_scope","DPU API wall-clock, not pure hardware time"}};
}
}
