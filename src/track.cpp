#include "track.h"
#include <opencv2/opencv.hpp>
#include "pbase64/base64.h"
#include <json/json.h>
#include <glog/logging.h>
#include "faceEntity.h"
#include "faceService.h"
#include "faceConfig.h"

namespace ktrack {
Track::~Track() {
  stop();
  LOG(INFO) << "track task exit";
}

bool Track::initClient() {
  return true;
}

bool Track::initPersonClient() {
  return true;
}

int Track::init() {
  auto config = kface::FaceConfig::getFaceConfig().getConfig();
  std::stringstream ss;
  ss << config->get("track", "low_alert");
  if (ss.str() != "") {
    ss >> lowAlert_;
  }
  ss.clear();
  ss.str("");
  ss << config->get("track", "high_alert");
  if (ss.str() != "") {
    ss >> highAlert_;
  }
  int rc = Init("192.168.1.111:9092", "face_183", "group_face_identify");
  if (rc == 0) {
    rc |= StartAll();
  }
  if (rc != 0) {
    return -1;
  }
  index_ = 0;
  right_[0] = cv::imread("right.jpg");
  error_[0] = cv::imread("error.jpg");
  right_[1] = cv::imread("right.jpg", 0);
  error_[1] = cv::imread("error.jpg", 0);
  return 0;
}

cv::Mat Track::getLatestImage() {
  std::lock_guard<std::mutex> lck(lock_);
  return image_;
}

static bool errorRc(int rc) {
  return rc == 1 ? false : true;
}

void Track::ProcessMessage(const char *buf, int len) {
  LOG(INFO) << "recv: " << len;
  static int recvCnt = 0;
  recvCnt = (recvCnt + 1) % 3;
  if (recvCnt != 0) {
    //return;
  }
  std::string recv(buf, len);
  Json::Value root;
  Json::Reader reader;
  bool f = reader.parse(recv, root);
  if (!f) {
    return;
  }
  std::string image = root["image"].asString();
  std::vector<unsigned char> data;
  Base64::getBase64().decode(image, data);
  cv::Mat m1 = cv::imdecode(data, 1); 
  cv::Mat m(m1, cv::Rect(m1.cols * 2 / 7, 0, m1.cols * 3/ 7, m1.rows));
  std::vector<kface::FaceDetectResult> results;
  kface::FaceService &service = kface::FaceService::getFaceService();
  service.detect(m, 3, results, false);
  auto it = results.begin();
  while (it != results.end()) {
    if (it->score < 0.7 || it->location.width *1.7 < it->location.height) {
      it = results.erase(it);
    } else {
      it++;
    }
  }

  if (results.size() == 0) {
    std::lock_guard<std::mutex> lck(lock_);
    image_ = m;
    return;
  }
  std::set<std::string> groupIds{"221"};
  cv::Mat showImage = results.size() > 1 ? m.clone() : m;
  for (auto &faceResult : results) {
      std::vector<kface::FaceSearchResult> searchResults;
      service.search(groupIds, faceResult.faceToken, 1, searchResults);
      if (searchResults.size() == 0) {
	continue;
     }
      bool find = false;
      if (searchResults[0].score > lowAlert_ && searchResults[0].score < highAlert_) {
	continue;
      }
      if (searchResults[0].score >= highAlert_) {
	find = true;
      }
      cv::Rect box(faceResult.location.x, faceResult.location.y,
		faceResult.location.width, faceResult.location.height);
      static cv::Scalar  RED(0, 0, 255);
      static cv::Scalar  GREEN(0, 255, 0);
      static cv::Scalar BLUE(255, 0, 0);
      cv::Scalar  &scalar = BLUE;
      cv::Mat faceImage(m, box);
      scalar = (!find) ? RED :GREEN;
      cv::Mat &alert = (!find) ? error_[0] : right_[0];
      cv::Mat &alertMask = (!find) ? error_[1] : right_[1];
      cv::Rect alertBox(box.x, box.y - alert.rows < 0 ? 0 : box.y - alert.rows, 
		      alert.cols,  alert.rows);
      cv::Mat alertImage(showImage, alertBox);
      alert.copyTo(alertImage, alertMask);
      
      std::string scoreStr = "null";
      if ( searchResults.size() > 0 ) {
        std::stringstream ss;
        ss << searchResults[0].userName << ":";
        ss << searchResults[0].score;
        ss >> scoreStr;
      }
      cv::putText(showImage, scoreStr, cv::Point(alertBox.x, alertBox.y) , 4, 2.5, scalar, 4);
      cv::rectangle(showImage, box, scalar, 2, 1);
  }
  if (results.size() > 0) {
    std::stringstream file;
    file << "cv/" << index_ << ".jpg";
    index_ = (index_ + 1) % 1000;
    cv::imwrite(file.str(), showImage);
  }
  std::lock_guard<std::mutex> lck(lock_);
  image_ = showImage;
}

} //namespace ktrack
