#pragma once

#include "Base.hpp"
#include "HailoInference.hpp"

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>

namespace blaze {

/**
 * Detector class - C++ port of blaze_hailo/blazedetector.py
 * 
 * This class implements blaze detection using the HailoInference system
 * for actual Hailo hardware inference or mock implementation for development.
 */
class Detector : public DetectorBase {
public:
    /**
     * Constructor
     * @param blaze_app The application type (e.g., "blazepalm", "blazeface")
     * @param hailo_infer Pointer to HailoInference instance
     */
    Detector(const std::string& blaze_app, std::shared_ptr<HailoInference> hailo_infer);
    
    /**
     * Destructor
     */
    virtual ~Detector() = default;
    
    /**
     * Load a Hailo model from file
     * @param model_path Path to the HEF model file
     */
    void load_model(const std::string& model_path);
    
    /**
     * Preprocess image data for inference
     * @param input Input image as cv::Mat
     * @return Preprocessed data as cv::Mat ready for inference
     */
    cv::Mat preprocess(const ImageType& input);
    
    /**
     * Make prediction on a single image
     * @param img Input image of shape (H, W, 3)
     * @return Detection results
     */
    std::vector<Detection> predict_on_image(const ImageType& img);
    
    /**
     * Make prediction on a batch of images
     * @param x Input batch of images of shape (b, H, W, 3)
     * @return Vector of detection results for each image
     */
    std::vector<std::vector<Detection>> predict_on_batch(const std::vector<ImageType>& x);
    
    // Profiling accessors
    double get_profile_pre() const { return profile_pre; }
    double get_profile_model() const { return profile_model; }
    double get_profile_post() const { return profile_post; }
    
    // Configuration methods
    void set_min_score_threshold(float threshold);

private:
    /**
     * Process raw model outputs into standardized tensor format
     * @param infer_results Raw model output from Hailo inference
     * @return Pair of processed tensors (scores, boxes)
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_model_outputs(const std::map<std::string, cv::Mat>& infer_results);
    
    /**
     * Process palm detection v0.07 outputs (6 outputs)
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_palm_v07_outputs(const std::map<std::string, cv::Mat>& infer_results);
    
    /**
     * Process palm detection lite outputs (4 outputs)  
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_palm_lite_outputs(const std::map<std::string, cv::Mat>& infer_results);

private:
    std::string blaze_app_;
    std::shared_ptr<HailoInference> hailo_infer_;
    int hef_id_;
    
    // Model information
    std::vector<VStreamInfo> input_vstream_infos_;
    std::vector<VStreamInfo> output_vstream_infos_;
    int num_inputs_;
    int num_outputs_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape1_;
    std::vector<size_t> output_shape2_;
    
    // Profiling
    double profile_pre;
    double profile_model;
    double profile_post;
};

} // namespace blaze
