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

#include <string.h>
 
#include <memory>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
 
#include "euler/common/logging.h"
#include "tf_euler/utils/euler_query_proxy.h"
 
#include "nebula/NebulaClient.h"
#include "nebula/ExecutionResponse.h"
 
 
namespace tensorflow {
 
class NebulaGetDenseFeature: public AsyncOpKernel {
 public:
  explicit NebulaGetDenseFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_names", &feature_names_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimensions", &dimensions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("space_name", &space_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("node_types", &node_types_));
    std::stringstream ss;
    for (size_t i = 0; i < feature_names_.size() * 2; ++i) {
      ss.str("");
      ss << "fea:" << i;
      res_names_.emplace_back(ss.str());
    }
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 
 private:
  std::vector<std::string> feature_names_;
  std::vector<int> dimensions_;
  std::string query_str_;
  std::vector<std::string> res_names_;
  int N_;
  std::string space_name_;
  std::vector<std::string> node_types_;
};
 
void NebulaGetDenseFeature::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto& shape = nodes.shape();
  std::vector<Tensor*> outputs(N_, nullptr);
  for (int i = 0; i < N_; ++i) {
    TensorShape output_shape;
    output_shape.AddDim(shape.dim_size(0));
    output_shape.AddDim(dimensions_[i]);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &outputs[i]));
    auto data = outputs[i]->flat<float>().data();
    auto end = data + shape.dim_size(0) * dimensions_[i];
    std::fill(data, end, 0.0);
  }
  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();
 
  std::stringstream ss;
  ss << "USE " << space_name_ << ";FETCH PROP ON * ";
  for (int i = 0; i < nodes_size-1; ++i) {
      ss << nodes_flat(i) << ", ";
  }
  ss << nodes_flat(nodes_size-1) << " YIELD ";
  for (auto type : node_types_) {
      ss << "n_" << type << ".w, ";
      for (auto name : feature_names_) {
          ss << "n_" << type << "." << "d_" << name << ", ";
      }
  }
  query_str_ = ss.str();
  query_str_ = query_str_.substr(0, query_str_.size()-2);
 
  nebula::NebulaClient client;
  nebula::ExecutionResponse response;
  client.execute(query_str_, response);
  client.executeFinished();
 
  std::unordered_map<std::string,
                     std::unordered_map<int64_t, std::vector<float>>
                     > featureResp;
  auto columnNames = response.getColumnNames();
  for (auto& row : response.getRows()) {
      auto columns = row.getColumns();
      std::string label("");
      for (int i = 1; i < columns.size(); ++i) {
          if (columnNames[i].substr(columnNames[i].find(".")+1) != "w") {
              continue;
          }
          if (columns[i].getDoubleValue() == 0) {
              continue;
          }
          label = columnNames[i].substr(0, columnNames[i].find("."));
          break;
      }
      for (int i = 1; i < columns.size(); ++i) {
          std::string labelName = columnNames[i].substr(0, columnNames[i].find("."));
          if (labelName != label) {
              continue;
          }
          if (columnNames[i].substr(columnNames[i].find(".")+1) == "w") {
              continue;
          }
          auto featureName = columnNames[i].substr(columnNames[i].find(".") +3);
          auto value = columns[i].getStrValue();
 
          std::string separator(" ");
          std::vector<float> values;
          std::string::size_type pos1, pos2;
          pos2 = value.find(separator);
          pos1 = 0;
          while (std::string::npos != pos2) {
              values.push_back(std::strtof(value.substr(pos1, pos2 - pos1).data(), nullptr));
              pos1 = pos2 + separator.size();
              pos2 = value.find(separator, pos1);
          }
          if (pos1 != value.length()) {
              values.push_back(strtof(value.substr(pos1).data(), nullptr));
          }
          featureResp[featureName][columns[0].getIdValue()] = values;
          break;
      }
  }
  for (size_t i = 0 ; i < feature_names_.size(); i++) {
      auto out_put = outputs[i]->flat<float>().data();
      for (size_t j = 0; j < nodes_size; ++j) {
          auto start = out_put + j * dimensions_[i];
          std::vector<float> feature = featureResp[feature_names_[i]][nodes_flat(j)];
          for (int k = 0; k < dimensions_[i]; k++, start++) {
              std::fill(start, start+1, feature[k]);
          }
      }
  }
  done();
}
 
REGISTER_KERNEL_BUILDER(
    Name("NebulaGetDenseFeature").Device(DEVICE_CPU), NebulaGetDenseFeature);
 
}  // namespace tensorflow
