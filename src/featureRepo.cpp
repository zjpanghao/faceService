#include "featureRepo.h"
#include "image_base64.h"
#include "faceEntity.h"
#include "faceConst.h"
#include "glog/logging.h"

#define FACE_FEATURE_LIB "face_feature_lib"
namespace kface {
FeatureRepo::FeatureRepo(mongoc_client_pool_t *pool, 
                              const std::string &dbName)
  : pool_(pool), DBNAME_(dbName) {
}

std::shared_ptr<FaceBuffer> FeatureRepo::getFaceBuffer(const std::string &faceToken) {
  if (faceToken.length() == 0) {
    return nullptr;
  }
  mongoc_client_t *client = mongoc_client_pool_pop(pool_);
  std::string collectionName = FACE_FEATURE_LIB;
  collectionName.append(1, faceToken[0]);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, DBNAME_.c_str(), collectionName.c_str());
  bson_t *query = BCON_NEW("face_token", faceToken.c_str());
  const bson_t *doc = NULL;
  mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(collection, query, NULL, NULL);
  char *result = NULL;
  std::shared_ptr<FaceBuffer> buffer;
  std::string featureBase64;
  Json::Value root;
  Json::Reader reader;
  int len = 0;
  std::string data;
  while (mongoc_cursor_next(cursor, &doc)) {
    result = bson_as_json(doc, NULL);
    break;
  }
  if (!result) {
    goto QUERY_END;
  }
  if (!reader.parse(result, root)) {
    goto QUERY_END;
  }
  featureBase64 = root["feature"].asString();
  if (featureBase64.empty()) {
    goto QUERY_END;
  }
  data = ImageBase64::decode(featureBase64.c_str(), featureBase64.length(), len);
  if (len == FaceFeature::FEATURE_VEC_LEN * sizeof(float)) {
    buffer.reset(new FaceBuffer());
    buffer->feature.assign((float*)&data[0], (float*)&data[0] + FaceFeature::FEATURE_VEC_LEN);
  }
  
QUERY_END:
  if (result) {
    bson_free(result);
  }
  bson_destroy(query);
  mongoc_cursor_destroy (cursor);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(pool_, client);
  return buffer;
}

int FeatureRepo::addFaceBuffer(const std::string &faceToken, std::shared_ptr<FaceBuffer> buffer) {
  if (faceToken.length() == 0) {
    return -1;
  }
  mongoc_client_t *client = mongoc_client_pool_pop(pool_);
  std::string collectionName = FACE_FEATURE_LIB;
  collectionName.append(1, faceToken[0]);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, DBNAME_.c_str(), collectionName.c_str());
  bson_t *insert = bson_new();
  bson_error_t error;
  std::string featureBase64 = ImageBase64::encode((unsigned char*)&buffer->feature[0], buffer->feature.size() * sizeof(float));
  BSON_APPEND_UTF8(insert, "feature", featureBase64.c_str());
  BSON_APPEND_UTF8(insert, "face_token", faceToken.c_str());
  BSON_APPEND_TIME_T(insert, "createTime", time(NULL));
  if (!mongoc_collection_insert_one(collection, insert, NULL, NULL, &error)) {
    LOG(ERROR) << error.message;
  }
  bson_destroy(insert);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(pool_, client);
  return 0;
}

}


