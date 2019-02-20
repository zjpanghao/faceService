#ifndef INCLUDE_FACE_SERVICE_H
#define INCLUDE_FACE_SERVICE_H
#include <map>
#include <memory>

#include <string>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <mutex>
#include "predis/redis_pool.h"
#include <mongoc/mongoc.h>
#include "face/faceApi.h"
#include <condition_variable>
#include "apipool/apiPool.h"

namespace kface {
class FeatureRepo;
class FaceRepo;
struct Location {
  int x;
  int y;
  int width;
  int height;
  float rotation;
};

struct FaceUpdateResult {
  std::string faceToken;
  Location location;
};


struct FaceAttr {
  int age;
  int gender;
  double genderConfidence;
  int expression;
  int glasses;
};

struct Occlution {
  double leftEye;
  double rightEye;
  double nose;
  double mouth;
  double leftCheek;
  double rightCheek;
  double chinContour;
};

struct FaceQuality {
  int illumination;
  double blur;
  int completeness;
  Occlution occlution;
};

struct FaceDetectResult {
  std::string faceToken;
  std::shared_ptr<FaceAttr> attr;
  std::shared_ptr<FaceQuality> quality;
  Location location;
  double score;
  //std::shared_ptr<TrackFaceInfo> trackInfo;
};

struct FaceSearchResult {
  std::string userId;
  std::string groupId;
  std::string userName;
  double score;
};

class FaceService {
 public:
  static FaceService& getFaceService();
  FaceService();
  /* init baiduapi(now only support one instance), facelib*/
  int init(mongoc_client_pool_t *mpool, const std::string &dbName, bool initFaceLib, int threadNum);
  /* detect face, caculate feature and buffer it with facetoken*/
  int detect(const std::vector<unsigned char> &data, 
             int faceNum,
             std::vector<FaceDetectResult> &result,
             bool smallFace = false);
  /*compare the feature with prestored facelib*/
  int search(const std::set<std::string> &groupIds, 
      const std::string &faceToken,
      int num,
      std::vector<FaceSearchResult> &result);
      
  int searchByImage64(const std::set<std::string> &groupIds, 
      const std::string &imageBase64,
      int num,
      std::vector<FaceSearchResult> &result);

  int addUserFace(const std::string &groupId,
                       const std::string &userId,
                       const std::string &userName,
                       const std::string &dataBase64,
                       std::string &faceToken);

  int delUser(const std::string &groupId,
                 const std::string &userId);

  int delUserFace(const std::string &groupId,
                       const std::string &userId,
                       const std::string &faceToken);

  int updateUserFace(const std::string &groupId,
                            const std::string &userId,
                            const std::string &userName,
                            const std::string &dataBase64,
                            FaceUpdateResult &updateResult);

  int match(const std::string &faceToken, 
              const std::string &faceTokenCompare,
              float &score);
              
  int matchImage(const std::string &image64,                   
                     const std::string &image64Compare,              
                     float &score);
                     
  int matchImageToken(const std::string &image64,                   
                      const std::string &faceToken,              
                      float &score);

 private:
  int search(std::shared_ptr<FaceApi> api,
      const std::set<std::string> &groupIds, 
      const std::vector<float> &feature,
      int num,
      std::vector<FaceSearchResult> &result);
      
  /*load facelib*/
  int initAgent();
  ApiBuffer<FaceApi> faceApiBuffer_;
  std::shared_ptr<FaceApi> faceApi_;
  std::shared_ptr<FeatureRepo> featureRepo_;
  std::shared_ptr<FaceRepo> faceRepo_;
 
  /*facelib lock*/
  pthread_rwlock_t faceLock_; 
};
}

#endif
