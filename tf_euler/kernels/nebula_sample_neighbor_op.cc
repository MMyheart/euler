/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <vector>
#include <sstream>
#include <sys/time.h>
 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
 
#include "tf_euler/utils/euler_query_proxy.h"
 
#include "nebula/NebulaClient.h"
#include "nebula/ExecutionResponse.h"
 
namespace tensorflow {
class NebulaSampleNeighbor: public AsyncOpKernel {
public:
    explicit NebulaSampleNeighbor(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("space_name", &space_name_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("all_edge_types", &all_edge_types_));
        for (int i=0; i<all_edge_types_.size(); i++) {
            types_index_.emplace("e_" + all_edge_types_[i], i);
        }
    }
 
    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 
private:
    int count_;
    int default_node_;
    std::string space_name_;
    std::vector<std::string> all_edge_types_;
    std::unordered_map<std::string, int32_t> types_index_;
};
 
void NebulaSampleNeighbor::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
    auto nodes = ctx->input(0);
    auto nodes_flat = nodes.flat<int64>();
    size_t nodes_size = nodes_flat.size();
 
    auto edge_types = ctx->input(1);
    auto etypes_flat = edge_types.flat<string>();
    size_t etypes_size = etypes_flat.size();
 
    // Output
    TensorShape output_shape;
    output_shape.AddDim(nodes.shape().dim_size(0));
    output_shape.AddDim(count_);
 
    Tensor* output = nullptr;
    Tensor* weights = nullptr;
    Tensor* types = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &weights));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, output_shape, &types));
 
    auto output_data = output->flat<int64>().data();
    auto weights_data = weights->flat<float>().data();
    auto types_data = types->flat<int32>().data();
    auto output_size = output_shape.dim_size(0) * output_shape.dim_size(1);
    std::fill(output_data, output_data + output_size, default_node_);
    std::fill(weights_data, weights_data + output_size, 0.0);
    std::fill(types_data, types_data + output_size, -1);
 
    std::stringstream ss;
    ss << "USE "<< space_name_ <<"; sampleNB FROM ";
 
    //node
    std::unordered_set<int64> nodeSet;
 
    std::stringstream node_append;
    for (int i = 0; i < nodes_size; i++) {
        auto id = nodes_flat(i);
        if (nodeSet.find(id) == nodeSet.end()) {
            node_append << id << ",";
            nodeSet.emplace(id);
        }
    }
    std::string nodesStr = node_append.str();
 
    ss << nodesStr.substr(0, nodesStr.size()-1) << " OVER ";
 
    //edge
    for(int i=0; i<etypes_size; i++){
        if (i == etypes_size -1) {
            ss << "e_" << etypes_flat(i) ;
        } else {
            ss << "e_" << etypes_flat(i) << ",";
        }
    }
    ss << " limit " << count_;
 
    nebula::NebulaClient client;
    nebula::ExecutionResponse resp;
    auto result =client.execute(ss.str(), resp);
    if (result != nebula::kSucceed) {
        EULER_LOG(ERROR) << "nebula client query error.";
        done();
        return;
    }
    client.executeFinished();
 
    std::unordered_map<int64_t,
                       std::vector<std::tuple<int64_t, float, int32_t>>>
                       neighborInfos;
 
    for (auto& row : resp.getRows()) {
        auto columns = row.getColumns();
        auto type = types_index_.find(columns[0].getStrValue())->second;
        auto src = columns[1].getIntValue();
        auto dst = columns[2].getIntValue();
        auto weight = static_cast<float>(columns[3].getDoubleValue());
        neighborInfos[src].emplace_back(dst, weight, type);
    }
 
    int64 index = 0;
    for (auto i=0; i < nodes_size; i++) {
        auto id = nodes_flat(i);
        auto iter = neighborInfos.find(id);
        if (iter != neighborInfos.end()) {
            auto neighbors = iter->second;
            for(auto j=0; j< neighbors.size(); j++) {
                auto nb_start = output_data + index + j;
                auto w_start = weights_data + index + j;
                auto t_start  = types_data + index + j;
                std::fill(nb_start, nb_start + 1, std::get<0>(neighbors[j]));
                std::fill(w_start, w_start + 1, std::get<1>(neighbors[j]));
                std::fill(t_start, t_start + 1, std::get<2>(neighbors[j]));
            }
        }
        index += count_;
    }
 
    done();
}
 
REGISTER_KERNEL_BUILDER(
        Name("NebulaSampleNeighbor").Device(DEVICE_CPU), NebulaSampleNeighbor);
 
}  // namespace tensorflow
