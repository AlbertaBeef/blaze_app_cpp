#pragma once

#include "Base.hpp"
#include "HailoInference.hpp"

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <chrono>

// OpenCV is required
#include <opencv2/opencv.hpp>

namespace blaze {


/**
 * Landmark - C++ port of blaze_hailo/blazelandmark.py
 * 
 * This class implements blaze landmark using the HailoInference system
 * for actual Hailo hardware inference or mock implementation for development.
 */
class Landmark : public LandmarkBase {
public:
    // Constructor takes blaze_app type and HailoInference instance
    Landmark(const std::string& blaze_app, std::shared_ptr<HailoInference> hailo_infer);
    virtual ~Landmark() = default;
    
    // Model loading and initialization
    bool load_model(const std::string& model_path);
    
    // Main prediction interface
    //std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>>
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    predict(const std::vector<cv::Mat>& input_images);
    
    // Profiling accessors
    double get_profile_pre() const { return profile_pre_; }
    double get_profile_model() const { return profile_model_; }
    double get_profile_post() const { return profile_post_; }

private:
    // Preprocess image for inference
    cv::Mat preprocess(const cv::Mat& input);

    // Member variables
    std::string blaze_app_;
    std::shared_ptr<HailoInference> hailo_infer_;
    int hef_id_;
    
    // Model information
    std::vector<VStreamInfo> input_vstream_infos_;
    std::vector<VStreamInfo> output_vstream_infos_;
    
    // Model configuration
    cv::Size input_shape_;
    cv::Size output_shape1_;
    cv::Size output_shape2_;
    
    // Profiling
    double profile_pre_;
    double profile_model_;
    double profile_post_;
    
#ifdef HAILO_SDK_AVAILABLE
    // Hailo-specific members - not needed since we use HailoInference class
#endif
};

} // namespace blaze
