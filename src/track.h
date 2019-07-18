#ifndef INCLUDE_TRACK_H
#define INCLUDE_TRACK_H
#include <memory>
#include <opencv2/opencv.hpp>
#include "pkafka/kafka_consumer.h"
#include <mutex>
#include "face/faceApi.h"
#include "config/config.h"

namespace ktrack {
class Track : public KafkaConsumer {
 public:
   ~Track();
   int init();
   void ProcessMessage(const char *buf, int len) override;
   bool initClient();
   bool initPersonClient();
   cv::Mat getLatestImage();

 private:
   int errorConnectCount_{0};
   long errorTime_{0};
   int errorPersonConnectCount_{0};
   long errorPersonTime_{0};
   int index_{0};
   cv::Mat image_;
   cv::Rect2d move_;
   std::mutex lock_;
   FaceApi  faceApi_;
   cv::Mat right_[2];
   cv::Mat error_[2];
   volatile bool stop_{true};
   float lowAlert_  {0.7};
   float highAlert_ {0.8};
   
};

} // namespace
#endif
