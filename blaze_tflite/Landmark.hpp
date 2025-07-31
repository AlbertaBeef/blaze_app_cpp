#pragma once

#include "Base.hpp"

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter_builder.h"

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
    /**
     * Constructor
     * @param blaze_app The application type (e.g., "blazehandlandmark", "blazefacelandmark")
     */

    Landmark(const std::string& blaze_app);
    virtual ~Landmark() = default;
    
    // Model loading and initialization
    bool load_model(const std::string& model_path);
    
    // Main prediction interface
    //std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>>
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    predict(const std::vector<cv::Mat>& input_images);
    
    // Profiling accessors
    double get_profile_pre() const { return profile_pre; }
    double get_profile_model() const { return profile_model; }
    double get_profile_post() const { return profile_post; }

private:
    // Preprocess image for inference
    cv::Mat preprocess(const cv::Mat& input);

    // Member variables
    std::string blaze_app;
    
    // Model information
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interp_landmark;
    TfLiteTensor* in_tensor;
    TfLiteTensor* out1_tensor;
    TfLiteTensor* out2_tensor;
    TfLiteTensor* out3_tensor;
    TfLiteIntArray *in_dims;
    TfLiteIntArray *out1_dims;
    TfLiteIntArray *out2_dims;
    TfLiteIntArray *out3_dims;
    
    int num_inputs;
    int num_outputs;

    int in_idx;
    int out1_confidence_idx;
    int out2_landmarks_idx;
    int out3_handedness_idx;    
    
    std::vector<int> in_shape;
    std::vector<int> out1_shape;
    std::vector<int> out2_shape;
    std::vector<int> out3_shape;
    int out1_size;
    int out2_size;
    int out3_size;
        
    // Model configuration
    cv::Size input_shape;
    cv::Size output_shape1;
    cv::Size output_shape2;
    cv::Size output_shape3;
    
    // Profiling
    double profile_pre;
    double profile_model;
    double profile_post;    
};

} // namespace blaze
