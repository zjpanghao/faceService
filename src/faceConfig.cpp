#include "faceConfig.h"

namespace kface {

FaceConfig::FaceConfig() {
  config_ = std::make_shared<kunyan::Config>("config.ini");
}

FaceConfig& FaceConfig::getFaceConfig() {
  static FaceConfig config;
  return config;
}

std::shared_ptr<const kunyan::Config> FaceConfig::getConfig() {
  return config_;
}

}
