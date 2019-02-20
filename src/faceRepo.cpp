#include <fstream>
#include "faceRepo.h"
#include "faceAgent.h"
#include <glog/logging.h>
#include <sstream>
#include <mongoc/mongoc.h>
#include "util.h"
#include "image_base64.h"
#include "faceConst.h"

#define FACE_LIB  "face_lib_new"

namespace kface {
FaceRepo::FaceRepo(mongoc_client_pool_t *pool, 
                       const std::string &dbName)
  : pool_(pool), DBNAME_(dbName) {
}

void FaceRepo::repoLoadPersonFaces(std::list<PersonFace> &faces) {
  mongoc_client_t *client = mongoc_client_pool_pop(pool_);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, DBNAME_.c_str(), FACE_LIB);
  bson_t *query = bson_new();
  const bson_t *doc = NULL;
  mongoc_cursor_t *cursor = mongoc_collection_find_with_opts(collection, query, NULL, NULL);
  char *result = NULL;
  std::string featureBase64;
  Json::Value root;
  Json::Reader reader;
  int len = 0;
  std::string data;
  while (mongoc_cursor_next(cursor, &doc)) {
    result = bson_as_json(doc, NULL);
    if (!result) {
      continue;
    }
    if (!reader.parse(result, root)) {
      LOG(ERROR) << "load person parse json error";
      goto QUERY_END;
    }
    bson_free(result);
    PersonFace face;
    face.image.reset(new ImageFace());
    getJsonString(root, "faceToken",  face.image->faceToken);
    getJsonString(root, "userId", face.userId);
    getJsonString(root, "userName", face.userName);
    getJsonString(root, "groupId", face.groupId);
    getJsonString(root, "appName", face.appName);
    featureBase64.clear();
    if (!root["feature"].isNull() && root["feature"].isString()) {
      featureBase64 = root["feature"].asString();
    }
    // getJsonString(root, "feature", featureBase64);
    if (featureBase64.empty()) {
      goto QUERY_END;
    }
    len = 0;
    data = ImageBase64::decode(featureBase64.c_str(), featureBase64.length(), len);
    
    if (len == 128 * sizeof(float)) {
      face.image->feature.assign((float*)&data[0], (float*)&data[0] + 128);
    } else {
      LOG(ERROR) << "decode len:" << len;
      continue;
    }
    faces.push_back(face);
  }
  
QUERY_END:
  bson_destroy(query);
  mongoc_cursor_destroy(cursor);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(pool_, client);
 }

int FaceRepo::repoAddUserFace(const PersonFace &face) {
  int rc = 0;
  mongoc_client_t *client = mongoc_client_pool_pop(pool_);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, DBNAME_.c_str(), FACE_LIB);
  bson_t *insert = bson_new();
  bson_error_t error;
  std::string featureBase64 = ImageBase64::encode((unsigned char*)&face.image->feature[0], 
                                                   face.image->feature.size() * sizeof(float));
  BSON_APPEND_UTF8(insert, "feature", featureBase64.c_str());
  BSON_APPEND_UTF8(insert, "faceToken", face.image->faceToken.c_str());
  BSON_APPEND_UTF8(insert, "groupId", face.groupId.c_str());
  BSON_APPEND_UTF8(insert, "userId", face.userId.c_str());
  BSON_APPEND_UTF8(insert, "userName", face.userName.c_str());
  BSON_APPEND_UTF8(insert, "appName", face.appName.c_str());
  if (!mongoc_collection_insert_one(collection, insert, NULL, NULL, &error)) {
    LOG(ERROR) << error.message;
    rc = -1;
  }
  bson_destroy(insert);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(pool_, client);
  return rc;
}

int FaceRepo::repoDelUserFace(const PersonFace &face) {
  int count = 0;
  mongoc_client_t *client = mongoc_client_pool_pop(pool_);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, DBNAME_.c_str(), FACE_LIB);
  bson_t *query = bson_new();
  bson_error_t error;
  mongoc_cursor_t *cursor = NULL;
  BSON_APPEND_UTF8(query, "faceToken", face.image->faceToken.c_str());
  BSON_APPEND_UTF8(query, "groupId", face.groupId.c_str());
  BSON_APPEND_UTF8(query, "userId", face.userId.c_str());
  BSON_APPEND_UTF8(query, "appName", face.appName.c_str());
  cursor = mongoc_collection_find_with_opts (collection, query, NULL, NULL);
  const bson_t *doc = NULL;
  while (mongoc_cursor_next(cursor, &doc)) {
    if (!mongoc_collection_remove(
      collection, MONGOC_REMOVE_SINGLE_REMOVE, doc, NULL, &error)) {
      LOG(ERROR) << error.message;
      count = -1;
      break;
    }
    count++;
  }
  bson_destroy(query);
  mongoc_cursor_destroy(cursor);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(pool_, client);
  return count;
}

int FaceRepo::repoDelUser(const PersonFace &face) {
  int count = 0;
  mongoc_client_t *client = mongoc_client_pool_pop(pool_);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, DBNAME_.c_str(), FACE_LIB);
  bson_t *query = bson_new();
  bson_error_t error;
  mongoc_cursor_t *cursor = NULL;
 
  BSON_APPEND_UTF8(query, "groupId", face.groupId.c_str());
  BSON_APPEND_UTF8(query, "userId", face.userId.c_str());
  BSON_APPEND_UTF8(query, "appName", face.appName.c_str());
  cursor = mongoc_collection_find_with_opts (collection, query, NULL, NULL);
  const bson_t *doc = NULL;
  while (mongoc_cursor_next(cursor, &doc)) {
    if (!mongoc_collection_remove(
        collection, MONGOC_REMOVE_SINGLE_REMOVE, doc, NULL, &error)) {
      LOG(ERROR) << error.message;
      count = -1;
      break;
    }
    count++;
  }
  bson_destroy(query);
  mongoc_cursor_destroy(cursor);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(pool_, client);
  return count;
}

}
