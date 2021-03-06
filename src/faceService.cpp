#include "faceService.h"
#include <sys/time.h>
#include <set>
#include "image_base64.h"
#include "md5.h"
#include "faceAgent.h"
#include "faceRepo.h"
#include <glog/logging.h>
#include <regex>
#include <iterator>
#include "cv_help.h"
#include "util.h"
#include "predis/redis_pool.h"
#define BAIDU_FEATURE_KEY "baiduFeature"
#define MAX_FACE_TRACK 5
#define FEATURE_VEC_LEN 128
#define FACE_FEATURE_LIB "face_feature_lib"
using  cv::Mat;
using cv::RotatedRect;
using cv::Rect;

namespace kface {
FaceService& FaceService::getFaceService() {
  static FaceService faceService;
  return faceService;
}

FaceService::FaceService() {
}

int FaceService::initAgent() {
  pthread_rwlock_init(&faceLock_, NULL);
  FaceAgent &faceAgent = FaceAgent::getFaceAgent();
  std::list<PersonFace> faces;
  repoLoadPersonFaces(faces);
  LOG(INFO) << "load persons :" << faces.size();
  for (PersonFace &face : faces) {
    faceAgent.addPersonFace(face);
  }
  return 0;
}

int FaceService::init(mongoc_client_pool_t *mpool, const std::string &dbName, bool initFaceLib, int threadNum) {
  //faceApi_.reset(new FaceApi());
  //apiBuffers_.init(1);
  faceApiBuffer_.init(threadNum);
  featureBuffers_.reset(new FeatureBuffer(mpool, dbName));
  if (initFaceLib) {
    initRepoFaces(mpool, dbName);
    initAgent();
  }
  return 0;
}

int FaceService::detect(const std::vector<unsigned char> &data,
    int faceNum,
    std::vector<FaceDetectResult> &detectResult,
    bool smallFace) {
  if (data.size() < 10) {
    return -1;
  }
  
  int rc = 0;
  struct timeval tv[2];  
  cv::Mat m = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
  cv::Mat show = m.clone();
  std::vector<FaceLocation> locations;
  ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
  auto faceApi = faceApiWrapper.getApi();
  if (faceApi == nullptr) {
    return -1;
  }
  gettimeofday(&tv[0], NULL);
  faceApi->getLocations(m, locations, smallFace);
  gettimeofday(&tv[1], NULL);
  //std::unique_ptr<std::vector<TrackFaceInfo>> out(new std::vector<TrackFaceInfo>());
  //std::vector<TrackFaceInfo> *vec = out.get();
  //int nFace = api->track(vec, m, faceNum);
  if (locations.size() <= 0) {
    return -2;
  }
  LOG(INFO) << "data size:" << data.size();
  LOG(INFO) << "location size" << locations.size();
  for (FaceLocation &location : locations) {
    FaceDetectResult result;
    result.score = location.confidence();
    Rect &rect = location.rect();
    int x = rect.x;
    int y = rect.y;
    if (x < 0 || y < 0 || rect.width + x > m.cols || rect.height + y > m.rows) { 
      LOG(ERROR) << "Error face Rect detect (x)" << x <<"(y) "<< y;
      return -3;
    }
    result.location.x = x;
    result.location.y = y;
    result.location.width= rect.width;
    result.location.height= rect.height;
    result.location.rotation = 0;
    cv::rectangle(show, rect, cv::Scalar(0,0, 255));
    
    /*getfeature and save*/
    Mat child(m, rect); 
    std::vector<float> feature;
    std::vector<unsigned char> childImage;
    //Mat cut = child.clone();
    //cut.data, cut.rows * cut.cols * cut.channels()
    LOG(INFO) << "tv0:" << tv[0].tv_sec << "  " << tv[0].tv_usec;
    LOG(INFO) << "tv1:" << tv[1].tv_sec << "  " << tv[1].tv_usec;
    std::shared_ptr<FaceBuffer> buffer(new FaceBuffer());
    if (faceApi->getFeature(child, buffer->feature) != 0 || buffer->feature.size() != FEATURE_VEC_LEN) {
      continue;
    }
    //buffer->feature.assign(feature, feature + 128);
    #if 0
    result.attr = getAttr(&childImage[0], childImage.size(), api);
    if (result.attr == nullptr) {
      continue;
    }
    result.quality = faceQuality(&childImage[0], childImage.size(), api);
    if (result.quality == nullptr) {
      continue;
    }
    #endif
    
 
    imencode(".jpg", child, childImage);
    result.faceToken = MD5(ImageBase64::encode(&childImage[0], childImage.size())).toStr(); 
    featureBuffers_->addBuffer(result.faceToken, buffer);   
    detectResult.push_back(result);
  } 

  if (detectResult.size() > 0) {
    imwrite("latest.jpg", show);
  }
  return rc;
}

int FaceService::search(const std::set<std::string> &groupIds, 
                        const std::string &faceToken, 
                        int num,
                        std::vector<FaceSearchResult> &searchResult) {

  std::shared_ptr<FaceBuffer> featureBuffer = featureBuffers_->getBuffer(faceToken);
  if (featureBuffer == nullptr) {
    LOG(INFO) << "error buffer";
    return -1;
  }
  ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
  auto faceApi = faceApiWrapper.getApi();
  if (faceApi == nullptr) {
    return -1;
  }
  return search(faceApi, groupIds, featureBuffer->feature, num, searchResult);
}

int FaceService::match(const std::string &faceToken,                   
                         const std::string &faceTokenCompare,              
                         float &score) {

  std::shared_ptr<FaceBuffer> featureBuffer = featureBuffers_->getBuffer(faceToken);
  std::shared_ptr<FaceBuffer> featureBufferCompare = featureBuffers_->getBuffer(faceTokenCompare);
  if (featureBuffer == nullptr || featureBufferCompare == nullptr) {
    return -1;
  }
  ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
  auto faceApi = faceApiWrapper.getApi();
  if (faceApi == nullptr) {
    return -1;
  }
  score = faceApi->compareFeature(featureBuffer->feature, featureBufferCompare->feature);
  return 0;
}

 int FaceService::matchImageToken(const std::string &image64,                   
     const std::string &faceToken,              
     float &score) {
   std::shared_ptr<FaceBuffer> featureBuffer = featureBuffers_->getBuffer(faceToken);
   if (featureBuffer == nullptr) {
     return -1;
   }
   ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
   auto faceApi = faceApiWrapper.getApi();
   if (faceApi == nullptr) {
     return -1;
   }
   int len = 0;
   std::string data = ImageBase64::decode(image64.c_str(), image64.length(), len);
   std::vector<unsigned char> vdata(&data[0], &data[0] + len);
   cv::Mat m = cv::imdecode(vdata, CV_LOAD_IMAGE_COLOR);
   std::vector<float> feature;
   faceApi->getFeature(m, feature);
   if (feature.size() != FEATURE_VEC_LEN) {
     return -2;
   }
   score = faceApi->compareFeature(featureBuffer->feature, feature);
   return 0;
 }

int FaceService::matchImage(const std::string &image64,                   
                                 const std::string &image64Compare,              
                                 float &score) {
   ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
   auto faceApi = faceApiWrapper.getApi();
   if (faceApi == nullptr) {
     return -1;
   }
   int len = 0;
   std::string data = ImageBase64::decode(image64.c_str(), image64.length(), len);
   std::vector<unsigned char> vdata(&data[0], &data[0] + len);
   cv::Mat m = cv::imdecode(vdata, CV_LOAD_IMAGE_COLOR);
   std::vector<float> feature;
   faceApi->getFeature(m, feature);
   if (feature.size() != FEATURE_VEC_LEN) {
     return -2;
   }
   std::vector<float> featureCompare;
   std::string dataCompare = ImageBase64::decode(image64Compare.c_str(), image64Compare.length(), len);
   std::vector<unsigned char> vdataCompare(&dataCompare[0], &dataCompare[0] + len);
   cv::Mat mCompare = cv::imdecode(vdataCompare, CV_LOAD_IMAGE_COLOR);
   faceApi->getFeature(mCompare, featureCompare);
   if (featureCompare.size() != FEATURE_VEC_LEN) {
     return -3;
   }
   return faceApi->compareFeature(feature, featureCompare); 
}

int FaceService::searchByImage64(const std::set<std::string> &groupIds, 
                                 const std::string &imageBase64, 
                                 int num,
                                 std::vector<FaceSearchResult> &searchResult) {
   ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
   auto faceApi = faceApiWrapper.getApi();
   if (faceApi == nullptr) {
     return -1;
   }
   int len = 0;
   std::string data = ImageBase64::decode(imageBase64.c_str(), imageBase64.length(), len);
   std::vector<unsigned char> vdata(&data[0], &data[0] + len);
   cv::Mat m = cv::imdecode(vdata, CV_LOAD_IMAGE_COLOR);
   std::vector<float> feature;
   faceApi->getFeature(m, feature);
   if (feature.size() != FEATURE_VEC_LEN) {
     return -2;
   }
   return search(faceApi, groupIds, feature, num, searchResult);
}
            
int FaceService::search(std::shared_ptr<FaceApi> api,
    const std::set<std::string> &groupIds, 
    const std::vector<float> &feature,
    int num,
    std::vector<FaceSearchResult> &result) {

  std::vector<PersonFace> top;
  RLockMethod rlock;
  RWLockGuard guard(rlock, &faceLock_);
  FaceAgent &faceAgent = FaceAgent::getFaceAgent();
  std::list<PersonFace> faces;
  faceAgent.getDefaultPersonFaces(faces);
  
  /*get the highest mark of each userId*/
  std::map<std::string, float> userScore;
  for (auto &face : faces) {
    if (groupIds.count(face.groupId) == 0) {
      continue;
    }
    float score = api->compareFeature(face.image->feature, feature);
    std::string key = face.groupId + "_" +face.userId; 
    if (userScore[key] < score)  {
      userScore[key] = score;
    }
  }

  /* find the top num marks*/
  std::vector<std::pair<std::string,float>>  topPair(userScore.begin(), userScore.end());
  std::partial_sort(topPair.begin(), 
                    num > topPair.size() ? topPair.end() : topPair.begin() + num, 
                    topPair.end(),
                    [](const std::pair<std::string, float> &a, const std::pair<std::string, float>  &b) {
                    return a.second > b.second;});
  for (auto &p : topPair) {
    if (--num < 0) {
      return 0;
    }
    FaceSearchResult tmp;
    std::string key = p.first;
    std::regex re{"_"};
    std::vector<std::string> groupUser(std::sregex_token_iterator(key.begin(), key.end(), re, -1),
                                       std::sregex_token_iterator());
    if (groupUser.size() == 2) {
      tmp.groupId = groupUser[0];
      tmp.userId = groupUser[1];
      tmp.score = p.second;
      result.push_back(tmp);
    }
  }
  return 0;
}

template<class E>
static void getBaiString(Json::Value &value, const std::string &key, E &t) {
  Json::Value &tmp = value["data"]["result"];
  if (tmp.isNull() || tmp[key].isNull()) {
    return;
  }
  std::stringstream ss;
  ss << tmp[key].asString();
  ss >> t;
}

std::shared_ptr<FaceAttr>  FaceService::getAttr(const unsigned char *data, int len){
  std::shared_ptr<FaceAttr> attr;
  return nullptr;
}

std::shared_ptr<FaceAttr>  FaceService::getAttr(const unsigned char *data, 
    int len,
    std::shared_ptr<FaceApi> api){
  if (api == nullptr) {
    return nullptr;
  }
  return nullptr;
}

std::shared_ptr<FaceQuality>  FaceService::faceQuality(const unsigned char *data, int len){
  std::shared_ptr<FaceQuality> value;
  return nullptr;
}

std::shared_ptr<FaceQuality>  FaceService::faceQuality(const unsigned char *data, int len,
    std::shared_ptr<FaceApi> api){
  return nullptr;
}

int FaceService::addUserFace(const std::string &groupId,
    const std::string &userId,
    const std::string &userName,
    const std::string &dataBase64,
    std::string &faceToken){
  int len = 0;
  std::string data = ImageBase64::decode(dataBase64.c_str(), dataBase64.length(), len);
  if (len < 10) {
    return -1;
  }
  PersonFace face;
  face.appName = DEFAULT_APP_NAME;
  face.groupId = groupId;
  face.userId = userId;
  face.userName = userName;
  const float *feature = nullptr;
  std::vector<unsigned char> mdata(&data[0], &data[0] + len);
  std::vector<FaceDetectResult> results;
  if (0 != detect(mdata, 1, results)) {
    return -1;
  }
  if (results.size() != 1) {
    return -2;
  }
  
  FaceDetectResult &result = results[0];
  std::shared_ptr<FaceBuffer> featureBuffer = featureBuffers_->getBuffer(result.faceToken);
  if (featureBuffer == nullptr) {
    return -3;
  }
  face.image.reset(new ImageFace());
  face.image->feature= featureBuffer->feature;
  face.image->faceToken = result.faceToken;
  WLockMethod wlock;
  RWLockGuard guard(wlock, &faceLock_);
  FaceAgent &faceAgent = FaceAgent::getFaceAgent();
  std::map<std::string, std::shared_ptr<ImageFace>> faceMap;
  faceAgent.getUserFaces(DEFAULT_APP_NAME, groupId, userId, faceMap);
  if (faceMap.count(result.faceToken) > 0 || faceMap.size() >= 5) {
    LOG(ERROR) << "already contain picture or picture size max then 5" << result.faceToken;
    return -10;
  }
  if (0 != repoAddUserFace(face)) {
    LOG(ERROR) << "repo add userface error";
    return -9;
  }

  int rc = faceAgent.addPersonFace(face);
  if (rc == 0) {
    flushFaces();
  }
  faceToken = face.image->faceToken;
  LOG(INFO) << "add face token:" << faceToken << "userId" << userId <<  "rc:" << rc;
  return rc;
}

int FaceService::updateUserFace(const std::string &groupId,
    const std::string &userId,
    const std::string &userName,
    const std::string &dataBase64,
    FaceUpdateResult &updateResult) {
  int len = 0;
  updateResult.faceToken = "";
  std::string data = ImageBase64::decode(dataBase64.c_str(), dataBase64.length(), len);
  if (len < 10) {
    return -1;
  }
  PersonFace face;
  face.appName = DEFAULT_APP_NAME;
  face.groupId = groupId;
  face.userId = userId;
  face.userName = userName;
  const float *feature = nullptr;
  std::vector<unsigned char> mdata(&data[0], &data[0] + len);
  std::vector<FaceDetectResult> results;
  if (0 != detect(mdata, 1, results)) {
    return -1;
  }
  if (results.size() != 1) {
    return -2;
  }
  
  FaceDetectResult &result = results[0];
  std::shared_ptr<FaceBuffer> featureBuffer = featureBuffers_->getBuffer(result.faceToken);
  if (featureBuffer == nullptr) {
    return -3;
  }
  face.image.reset(new ImageFace());
  face.image->feature= featureBuffer->feature;
  face.image->faceToken = result.faceToken;
  WLockMethod wlock;
  RWLockGuard guard(wlock, &faceLock_);
  FaceAgent &faceAgent = FaceAgent::getFaceAgent();
  std::map<std::string, std::shared_ptr<ImageFace>> faceMap;
  int count = repoDelUser(face);
  LOG(INFO) << "repoDel: " << count;
  if (count < 0) {
    return -5;
  }

  int rc = faceAgent.delPerson(face);
  LOG(INFO) << "delete person :" << face.userId << rc;
  
  if (repoAddUserFace(face) < 0) {
    LOG(ERROR) << "repo add user face error";
    return -6;
  }
 
  rc = faceAgent.addPersonFace(face);
  if (rc == 0) {
    flushFaces();
  }
  updateResult.faceToken = face.image->faceToken;
  updateResult.location = result.location;
  LOG(INFO) << "add face token:" << face.image->faceToken << "userId" << userId <<  "rc:" << rc;
  return rc;
}

int FaceService::delUserFace(const std::string &groupId,
    const std::string &userId,
    const std::string &faceToken) {
  int rc = 0;
  PersonFace face;
  face.appName = DEFAULT_APP_NAME;
  face.groupId = groupId;
  face.userId = userId;
  face.image.reset(new ImageFace());
  face.image->faceToken = faceToken;
  WLockMethod wlock;
  RWLockGuard guard(wlock, &faceLock_);
  int count = 0;
  if ((count = repoDelUserFace(face)) <= 0) {
    LOG(ERROR) << "repo del face error" << count;
    return -5;
  }
  FaceAgent &agent = FaceAgent::getFaceAgent();
  rc = agent.delPersonFace(face);
  if (rc == 0) {
    flushFaces();
  }
  LOG(INFO) << "del face token:" << faceToken << "userId" << userId <<  "rc:" << rc;
  return 0;
}

int FaceService::delUser(const std::string &groupId,
    const std::string &userId) {
  int rc = 0;
  PersonFace face;
  face.appName = DEFAULT_APP_NAME;
  face.groupId = groupId;
  face.userId = userId;
  WLockMethod wlock;
  RWLockGuard guard(wlock, &faceLock_);
  int count = repoDelUser(face);
  LOG(ERROR) << "repo del user:" << count;
  if (count < 0) {
    return -1;
  }
  FaceAgent &agent = FaceAgent::getFaceAgent();
  rc = agent.delPerson(face);
  if (rc == 0) {
    flushFaces();
  }
  LOG(INFO) << "del user  userId" << userId <<  "rc:" << rc;
  return 0;
}

int FeatureBuffer::getBufferIndex() {
  time_t current = time(NULL);
  struct tm val;
  localtime_r(&current, &val);
  int inx = val.tm_mday % 2;
  return inx;
}

std::shared_ptr<FaceBuffer> FeatureBuffer::getMongoBuffer(const std::string &faceToken) {
  mongoc_client_t *client = mongoc_client_pool_pop(mongoPool_);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, dbName_.c_str(), FACE_FEATURE_LIB);
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
  if (len == FEATURE_VEC_LEN * sizeof(float)) {
    buffer.reset(new FaceBuffer());
    buffer->feature.assign((float*)&data[0], (float*)&data[0] + FEATURE_VEC_LEN);
  }
  
QUERY_END:
  if (result) {
    bson_free(result);
  }
  bson_destroy(query);
  mongoc_cursor_destroy (cursor);
  mongoc_collection_destroy(collection);
  mongoc_client_pool_push(mongoPool_, client);
  return buffer;
}

std::shared_ptr<FaceBuffer> FeatureBuffer::getRedisBuffer(const std::string &faceToken) {
  RedisControlGuard guard(redisPool_.get());
  std::shared_ptr<RedisControl> control = guard.GetControl();
  if (control == nullptr) {
    return nullptr;
  }
  std::string featureBase64;
  control->GetHashValue(BAIDU_FEATURE_KEY, faceToken, &featureBase64);
  if (featureBase64.empty()) {
    return nullptr;
  }
  int len = 0;
  std::string data = ImageBase64::decode(featureBase64.c_str(), featureBase64.length(), len);
  if (len != FEATURE_VEC_LEN * sizeof(float)) {
    return nullptr;
  }
  std::shared_ptr<FaceBuffer> buffer(new FaceBuffer());
  buffer->feature.assign((float*)&data[0], (float*)&data[0] + FEATURE_VEC_LEN);
  return buffer;
}


std::shared_ptr<FaceBuffer> FeatureBuffer::getBuffer(const std::string &faceToken) {
  if (type_ == BufferType::REDIS) {
    return getRedisBuffer(faceToken);
  } else if (type_ == BufferType::MONGO) {
    return getMongoBuffer(faceToken);
  }

  #if 0
  int inx = getBufferIndex();
  std::lock_guard<std::mutex> guard(lock_);
  auto it = faceBuffers[inx].find(faceToken);
  if (it == faceBuffers[inx].end()) {
    return nullptr;
  }
  return it->second;
  #endif
}

void FeatureBuffer::addRedisBuffer(const std::string &faceToken, std::shared_ptr<FaceBuffer> buffer) {
  RedisControlGuard guard(redisPool_.get());
  std::shared_ptr<RedisControl> control = guard.GetControl();
  if (control == nullptr) {
    return;
  }    
  std::string featureBase64 = ImageBase64::encode((unsigned char*)&buffer->feature[0], buffer->feature.size() * sizeof(float));
  control->SetHashValue(BAIDU_FEATURE_KEY, faceToken, featureBase64);
}

void FeatureBuffer::addMongoBuffer(const std::string &faceToken, 
                                          std::shared_ptr<FaceBuffer> buffer) {
  mongoc_client_t *client = mongoc_client_pool_pop(mongoPool_);
  mongoc_collection_t *collection = mongoc_client_get_collection(client, dbName_.c_str(), FACE_FEATURE_LIB);
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
  mongoc_client_pool_push(mongoPool_, client);
}

void FeatureBuffer::addBuffer(const std::string &faceToken, std::shared_ptr<FaceBuffer> buffer) {
  std::stringstream ss;
  if (type_ == BufferType::REDIS) {
    addRedisBuffer(faceToken, buffer);
  } else if (type_ == BufferType::MONGO) {
    struct timeval tv[2];
    gettimeofday(&tv[0], NULL);
    addMongoBuffer(faceToken, buffer);
    gettimeofday(&tv[1], NULL);
    LOG(INFO) << "tv0:" << tv[0].tv_sec << "  " << tv[0].tv_usec;
    LOG(INFO) << "tv1:" << tv[1].tv_sec << "  " << tv[1].tv_usec;
  }
#if 0
  int inx = getBufferIndex();
  int old = 1 - inx;
  std::lock_guard<std::mutex> guard(lock_);
  if (!faceBuffers[old].empty()) {
    LOG(INFO) <<"clear buffer size:" <<  faceBuffers[old].size();
    faceBuffers[old].clear();
  }
  faceBuffers[inx].insert(std::make_pair(faceToken, buffer));
  #endif
}
#if 0
int BaiduFaceApiBuffer::init(int bufferNums) {
  for (int i = 0; i < bufferNums; i++) {
    auto api = getInitApi();
    if (api != nullptr) {
      apis_.push_back(api);
    }
  }
  return apis_.empty() ? -1 : 0; 
}

std::shared_ptr<BaiduFaceApi> BaiduFaceApiBuffer::getInitApi() {
  std::shared_ptr<BaiduFaceApi> api(new BaiduFaceApi());
  int rc = api->sdk_init(false);
  if (rc != 0) {
    return nullptr;
  }
  if (!api->is_auth()) {
    return nullptr;
  }
  api->set_min_face_size(1);

  return api;
  return nullptr;
}

std::shared_ptr<BaiduFaceApi> BaiduFaceApiBuffer::borrowBufferedApi() {
  std::unique_lock<std::mutex> ulock(lock_);
  bufferFull_.wait(ulock, [this](){return !apis_.empty();});
  auto api = apis_.front();
  apis_.pop_front();
  return api;
}

void BaiduFaceApiBuffer::offerBufferedApi(std::shared_ptr<BaiduFaceApi> api) {
  if (api == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> ulock(lock_);
  if (apis_.empty()) {
    bufferFull_.notify_one();
  }
  apis_.push_back(api);
}
int FaceApiBuffer::init(int bufferNums) {
  for (int i = 0; i < bufferNums; i++) {
    auto api = getInitApi();
    if (api != nullptr) {
      apis_.push_back(api);
    }
  }
  return apis_.empty() ? -1 : 0; 
}

std::shared_ptr<FaceApi> FaceApiBuffer::getInitApi() {
  std::shared_ptr<FaceApi> api(new FaceApi());
  return api;
}

std::shared_ptr<FaceApi> FaceApiBuffer::borrowBufferedApi() {
  std::unique_lock<std::mutex> ulock(lock_);
  bufferFull_.wait(ulock, [this](){return !apis_.empty();});
  auto api = apis_.front();
  apis_.pop_front();
  return api;
}

void FaceApiBuffer::offerBufferedApi(std::shared_ptr<FaceApi> api) {
  if (api == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> ulock(lock_);
  if (apis_.empty()) {
    bufferFull_.notify_one();
  }
  apis_.push_back(api);
}
#endif
}
