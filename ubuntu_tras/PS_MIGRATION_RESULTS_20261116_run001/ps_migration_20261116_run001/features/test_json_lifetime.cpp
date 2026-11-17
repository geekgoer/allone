// Review regression: indexing a temporary JSON in a range-for leaves a dangling
// subobject reference in C++14. Default demonstrates the correct lifetime.
#include "json.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using nlohmann::json;
json problem_inputs() { return json::parse("{\"inputs\":[{\"raw237\":[1,2,3],\"model_id\":11}]}"); }
int main(int argc,char** argv) {
    int count=0;
    if(argc==2&&std::string(argv[1])=="--unsafe") {
        for(const auto& row:problem_inputs()["inputs"]) {
            std::cout<<row.at("model_id").get<int>()<<'\n';++count;
        }
    } else {
        const json problems=problem_inputs();
        for(const auto& row:problems.at("inputs")) {
            if(row.at("model_id").get<int>()!=11)throw std::runtime_error("wrong row");++count;
        }
    }
    if(count!=1)throw std::runtime_error("wrong row count");
    std::cout<<"JSON_LIFETIME_SAFE_PASS rows="<<count<<'\n';
}
