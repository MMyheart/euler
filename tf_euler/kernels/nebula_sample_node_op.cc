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
 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
 
#include "tf_euler/utils/euler_query_proxy.h"
 
#include "nebula/NebulaClient.h"
#include "nebula/ExecutionResponse.h"
 
namespace tensorflow {
 
class NebulaSampleNode: public AsyncOpKernel {
    public:
    explicit NebulaSampleNode(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("condition", &condition_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("space_name", &space_name_));
    }
 
    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 
    private:
        std::string condition_;
        std::string space_name_;
};
 
void NebulaSampleNode::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
    auto count = ctx->input(0);
    auto node_type = ctx->input(1);
 
    OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(count.shape()),
              errors::InvalidArgument("count must be a scalar, saw shape: ",
                                      count.shape().DebugString()), done);
 
    OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(node_type.shape()),
              errors::InvalidArgument("node_type must be a scalar, saw shape: ",
                                      node_type.shape().DebugString()), done);
 
    int32_t count_value = (count.scalar<int32>())();
    string type_value = (node_type.scalar<string>())();
 
    TensorShape output_shape;
    output_shape.AddDim(count_value);
 
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    auto output_data = output->flat<int64>().data();
 
    std::stringstream ss;
    ss << "USE " << space_name_ << "; SAMPLE VERTEX n_" << type_value << " LIMIT " << count_value;
    nebula::NebulaClient client;
    nebula::ExecutionResponse resp;
    auto result = client.execute(ss.str(), resp);
    if (result != nebula::kSucceed) {
        EULER_LOG(ERROR) << "nebula client query error.";
        done();
        return;
    }
    client.executeFinished();
 
    auto data_start = output_data;
    for (auto& row : resp.getRows()) {
        auto columns = row.getColumns();
        auto node_id = columns[1].getIntValue();
        std::fill(data_start, data_start + 1, node_id);
        data_start = data_start + 1;
    }
 
    done();
}
 
REGISTER_KERNEL_BUILDER(Name("NebulaSampleNode").Device(DEVICE_CPU), NebulaSampleNode);
 
}  // namespace tensorflow
