#ifndef INCLUDE_FEATURE_REPO_H
#define INCLUDE_FEATURE_REPO_H
#include <string>
#include <json/json.h>
#include <list>
#include <vector>
#include <memory>
#include "faceEntity.h"
#include <mongoc/mongoc.h>

namespace kface {
struct FaceBuffer;
class FeatureRepo {
 public:
  FeatureRepo(mongoc_client_pool_t *pool, const std::string &dbName);
  std::shared_ptr<FaceBuffer> getFaceBuffer(const std::string &faceToken);
  int addFaceBuffer(const std::string &faceToken, std::shared_ptr<FaceBuffer> buffer);
  
 private:
  mongoc_client_pool_t *pool_{NULL};
  const std::string DBNAME_;
};
}
#endif
