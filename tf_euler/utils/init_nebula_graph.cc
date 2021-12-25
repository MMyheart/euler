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

#include "nebula/NebulaClient.h"

void splitString(const std::string &s, std::vector<std::string> &v,
                 const std::string &c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length())
  {
    v.push_back(s.substr(pos1));
  }
}

extern "C"
{
  bool InitNebulaGraph(char *arg, uint32_t port, int32_t timeout, uint32_t minConnectionNum, uint32_t maxConnectionNum)
  {
    std::string str = arg;
    std::vector<std::string> vec;
    splitString(str, vec, ",");
    std::vector<std::pair<std::string, uint32_t>> addrs;
    addrs.reserve(vec.size());
    for (int i = 0; i < vec.size(); i++)
    {
      std::string ip = vec[i];
      addrs.emplace_back(ip, port);
    }

    nebula::ConnectionInfo connectionInfo;
    connectionInfo.addrs = std::move(addrs);
    connectionInfo.timeout = timeout;
    connectionInfo.minConnectionNum = minConnectionNum;
    connectionInfo.maxConnectionNum = maxConnectionNum;
    nebula::NebulaClient::initConnectionPool(connectionInfo);
    return true;
  }
}
