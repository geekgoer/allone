"""Make a separate board build that records the first executor exception."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_one(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Expected one source anchor: " + before[:80])
    return source.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be new")
    shutil.copytree(args.source, args.output)
    header = args.output / "adaptive_live.hpp"
    code = header.read_text(encoding="utf-8")
    code = replace_one(code, "#include <deque>", "#include <deque>\n#include <exception>")
    code = replace_one(code,
        "bool stop_=false;std::atomic<bool> failed_{false};std::thread worker_;",
        "bool stop_=false;std::atomic<bool> failed_{false};std::string failure_reason_;std::thread worker_;")
    code = replace_one(code,
        "if(stop_||failed_.load()||payloads_.size()>=12||payloads_.count(name))throw std::runtime_error(\"DPU payload registry full/closed/failed/duplicate\");",
        """if(failed_.load())throw std::runtime_error("DPU executor failed: "+failure_reason_);
        if(stop_)throw std::runtime_error("DPU payload registry closed");
        if(payloads_.size()>=12)throw std::runtime_error("DPU payload registry full size="+std::to_string(payloads_.size()));
        if(payloads_.count(name))throw std::runtime_error("DPU payload registry duplicate "+name);""")
    code = replace_one(code,
        "void notify(){cv_.notify_all();}\n    bool failed()const{return failed_.load();}",
        "void notify(){cv_.notify_all();}\n    bool failed()const{return failed_.load();}\n    std::string failure_reason(){std::lock_guard<std::mutex> lock(mutex_);return failure_reason_;}")
    code = replace_one(code,
        """        }catch(...){
            failed_=true;
            if(slot)try{slot->promise.set_exception(std::current_exception());}catch(...){}
            // A structural failure is observable. Resolve any waiting payloads
            // rather than leaving per-vehicle workers blocked during shutdown.
            std::lock_guard<std::mutex> lock(mutex_);stop_=true;""",
        """        }catch(...){
            auto failure=std::current_exception();std::string reason="unknown non-std exception";
            try{std::rethrow_exception(failure);}catch(const std::exception& e){reason=e.what();}catch(...){}
            if(slot)try{slot->promise.set_exception(failure);}catch(...){}
            // A structural failure is observable. Resolve any waiting payloads
            // rather than leaving per-vehicle workers blocked during shutdown.
            std::lock_guard<std::mutex> lock(mutex_);failure_reason_=reason;failed_=true;stop_=true;""")
    header.write_text(code, encoding="utf-8")
    runtime = args.output / "adaptive_runtime.cpp"
    code = runtime.read_text(encoding="utf-8")
    code = replace_one(code,
        '{"raw_journal_rows",raw_journal.written()},{"planner_log_rows",plan_journal.written()},{"executor_failed",executor.failed()},',
        '{"raw_journal_rows",raw_journal.written()},{"planner_log_rows",plan_journal.written()},{"executor_failed",executor.failed()},{"executor_failure_reason",executor.failure_reason()},')
    runtime.write_text(code, encoding="utf-8")
    manifest = {p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.output.rglob("*") if p.is_file() and p.name != "manifest.json"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("trace-only source files", len(manifest))


if __name__ == "__main__":
    main()
