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
#include "faceConst.h"
#include "featureRepo.h"

#define MAX_FACE_TRACK 5

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
  faceRepo_->repoLoadPersonFaces(faces);
  LOG(INFO) << "load persons :" << faces.size();
  for (PersonFace &face : faces) {
    faceAgent.addPersonFace(face);
  }
  return 0;
}

int FaceService::init(mongoc_client_pool_t *mpool, const std::string &dbName, bool initFaceLib, int threadNum) {
  faceApiBuffer_.init(threadNum);
  featureRepo_.reset(new FeatureRepo(mpool, dbName));
  if (initFaceLib) {
    faceRepo_.reset(new FaceRepo(mpool, dbName));
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
  struct timeval detectStamp[2];  
  cv::Mat m = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
  // cv::Mat show = m.clone();
  std::vector<FaceLocation> locations;
  ApiWrapper<FaceApi> faceApiWrapper(faceApiBuffer_);
  auto faceApi = faceApiWrapper.getApi();
  if (faceApi == nullptr) {
    return -1;
  }
  gettimeofday(&detectStamp[0], NULL);
  faceApi->getLocations(m, locations, smallFace);
  gettimeofday(&detectStamp[1], NULL);
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
    
    /*getfeature and save*/
    Mat child(m, rect); 
    std::vector<float> feature;
    std::vector<unsigned char> childImage;
    LOG(INFO) << "detect0:" << detectStamp[0].tv_sec << "  " << detectStamp[0].tv_usec;
    LOG(INFO) << "detect1:" << detectStamp[1].tv_sec << "  " << detectStamp[1].tv_usec;
    std::shared_ptr<FaceBuffer> buffer(new FaceBuffer());
    struct timeval featureStamp[2];
    gettimeofday(&featureStamp[0], NULL);
    if (faceApi->getFeature(child, buffer->feature) != 0 || buffer->feature.size() != FaceFeature::FEATURE_VEC_LEN) {
      continue;
    }
    gettimeofday(&featureStamp[1], NULL);
    LOG(INFO) << "feature0:" << featureStamp[0].tv_sec << "  " << featureStamp[0].tv_usec;
    LOG(INFO) << "feature1:" << featureStamp[1].tv_sec << "  " << featureStamp[1].tv_usec; 
    imencode(".jpg", child, childImage);
    result.faceToken = MD5(ImageBase64::encode(&childImage[0], childImage.size())).toStr(); 
    featureRepo_->addFaceBuffer(result.faceToken, buffer);
    detectResult.push_back(result);
  } 
  
#if DEBUG
  if (detectResult.size() > 0) {
    static int picLogInx = 0;
    for (FaceLocation &location : locations) {
      cv::rectangle(m, location.rect(), cv::Scalar(0,0, 255));
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "latest_%d.jpg", picLogInx);
    picLogInx = (picLogInx + 1) % 10;
    imwrite(buf, m);
  }
#endif
  return rc;
}

int FaceService::search(const std::set<std::string> &groupIds, 
                        const std::string &faceToken, 
                        int num,
                        std::vector<FaceSearchResult> &searchResult) {

  std::shared_ptr<FaceBuffer> featureBuffer = featureRepo_->getFaceBuffer(faceToken);
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

  std::shared_ptr<FaceBuffer> featureBuffer = featureRepo_->getFaceBuffer(faceToken);
  std::shared_ptr<FaceBuffer> featureBufferCompare = featureRepo_->getFaceBuffer(faceTokenCompare);
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
   std::shared_ptr<FaceBuffer> featureBuffer = featureRepo_->getFaceBuffer(faceToken);
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
   if (feature.size() != FaceFeature::FEATURE_VEC_LEN) {
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
   if (feature.size() != FaceFeature::FEATURE_VEC_LEN) {
     return -2;
   }
   std::vector<float> featureCompare;
   std::string dataCompare = ImageBase64::decode(image64Compare.c_str(), image64Compare.length(), len);
   std::vector<unsigned char> vdataCompare(&dataCompare[0], &dataCompare[0] + len);
   cv::Mat mCompare = cv::imdecode(vdataCompare, CV_LOAD_IMAGE_COLOR);
   faceApi->getFeature(mCompare, featureCompare);
   if (featureCompare.size() != FaceFeature::FEATURE_VEC_LEN) {
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
   if (feature.size() != FaceFeature::FEATURE_VEC_LEN) {
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
  std::shared_ptr<FaceBuffer> featureBuffer = featureRepo_->getFaceBuffer(result.faceToken);
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
  if (0 != faceRepo_->repoAddUserFace(face)) {
    LOG(ERROR) << "repo add userface error";
    return -9;
  }

  int rc = faceAgent.addPersonFace(face);
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
  std::shared_ptr<FaceBuffer> featureBuffer = featureRepo_->getFaceBuffer(result.faceToken);
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
  int count = faceRepo_->repoDelUser(face);
  LOG(INFO) << "repoDel: " << count;
  if (count < 0) {
    return -5;
  }

  int rc = faceAgent.delPerson(face);
  LOG(INFO) << "delete person :" << face.userId << rc;
  
  if (faceRepo_->repoAddUserFace(face) < 0) {
    LOG(ERROR) << "repo add user face error";
    return -6;
  }
 
  rc = faceAgent.addPersonFace(face);
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
  if ((count = faceRepo_->repoDelUserFace(face)) <= 0) {
    LOG(ERROR) << "repo del face error" << count;
    return -5;
  }
  FaceAgent &agent = FaceAgent::getFaceAgent();
  rc = agent.delPersonFace(face);
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
  int count = faceRepo_->repoDelUser(face);
  LOG(ERROR) << "repo del user:" << count;
  if (count < 0) {
    return -1;
  }
  FaceAgent &agent = FaceAgent::getFaceAgent();
  rc = agent.delPerson(face);
  LOG(INFO) << "del user  userId" << userId <<  "rc:" << rc;
  return 0;
}

}
