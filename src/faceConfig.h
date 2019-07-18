#ifndef INCLUDE_FACE_CONFIG_H
#define INCLUDE_FACE_CONFIG_H
#include "config/config.h"
#include <memory>
namespace kface {
class FaceConfig {
 public:
  static FaceConfig &getFaceConfig();
  FaceConfig();
  std::shared_ptr<const kunyan::Config> getConfig();
 private:
  std::shared_ptr<const kunyan::Config> config_;
};

} // namepsace 
#endif
