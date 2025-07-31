#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "Landmark.hpp"

namespace blaze {

// ============================================================================
// Landmark Implementation
// ============================================================================

Landmark::Landmark(const std::string& blaze_app)
    : LandmarkBase()
    , blaze_app(blaze_app)
    , input_shape(256, 256)
    , output_shape1(1, 1)
    , output_shape2(1, 63)
    , profile_pre(0.0)
    , profile_model(0.0)
    , profile_post(0.0) {
}

bool Landmark::load_model(const std::string& model_path) {
    if (DEBUG) {
        std::cout << "[Landmark.load_model] blaze_app= " << blaze_app << std::endl;
        std::cout << "[Landmark.load_model] model_path=" << model_path << std::endl;
    }

    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interp_landmark);

    interp_landmark->AllocateTensors();

    // Get input/output details
    num_inputs = interp_landmark->inputs().size();
    num_outputs = interp_landmark->outputs().size();

    const std::vector<int>& input_indices = interp_landmark->inputs();
    const std::vector<int>& output_indices = interp_landmark->outputs();

    if (DEBUG) {
        std::cout << "[Landmark.load_model] Number of Inputs : " << num_inputs << std::endl;
        for (int i = 0; i < num_inputs; ++i) {
            int tensor_index = input_indices[i];
            TfLiteTensor* input_tensor = interp_landmark->tensor(tensor_index);
            // Now you can use input_tensor, e.g., input_tensor->name, input_tensor->dims, etc.
            std::cout << "[Landmark.load_model] Input[" << i << "] "
                      << "Name=" << input_tensor->name << " "
                      << "Type=" << TfLiteTypeGetName(input_tensor->type) << " "
                      << "Shape=";
            TfLiteIntArray *dims = input_tensor->dims;
            std::cout << "[";
            for (int i = 0; i < dims->size; ++i) {
                std::cout << dims->data[i];
                if (i < dims->size - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
	    
        std::cout << "[Landmark.load_model] Number of Outputs : " << num_outputs << std::endl;
        for (int i = 0; i < num_outputs; ++i) {
            int tensor_index = output_indices[i];
            TfLiteTensor* output_tensor = interp_landmark->tensor(tensor_index);
            // Now you can use output_tensor, e.g., input_tensor->name, input_tensor->dims, etc.
            std::cout << "[Landmark.load_model] Output[" << i << "] "
                      << "Name=" << output_tensor->name << " "
                      << "Type=" << TfLiteTypeGetName(output_tensor->type) << " "
                      << "Shape=";
            TfLiteIntArray *dims = output_tensor->dims;
            std::cout << "[";
            for (int i = 0; i < dims->size; ++i) {
                std::cout << dims->data[i];
                if (i < dims->size - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    in_idx = input_indices[0];
    out1_confidence_idx = output_indices[1];
    out2_landmarks_idx = output_indices[0];
    if (blaze_app == "blazehandlandmark") {
        out3_handedness_idx = output_indices[2];
    }
    else {
        out3_handedness_idx = -1;
    }
    
    in_tensor = interp_landmark->tensor(in_idx);
    in_dims = in_tensor->dims;
    in_shape = std::vector<int>(in_dims->data, in_dims->data + in_dims->size);
    if (DEBUG) {
        std::cout << "[Landmark.load_model] in_shape = [";
        for (size_t i = 0; i < in_shape.size(); ++i) {
            std::cout << in_shape[i];
            if (i != in_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;        
    }

    out1_tensor = interp_landmark->tensor(out1_confidence_idx);
    out1_dims = out1_tensor->dims;
    out1_shape = std::vector<int>(out1_dims->data, out1_dims->data + out1_dims->size);
    out1_size = 1;
    for (size_t i = 0; i < out1_shape.size(); ++i) { 
        out1_size = out1_size * out1_shape[i];
    }
    if (DEBUG) {
        std::cout << "[Landmark.load_model] out1 (confidence) size=" << out1_size << " shape=[";
        for (size_t i = 0; i < out1_shape.size(); ++i) {
            std::cout << out1_shape[i];
            if (i != out1_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;        
    }

    out2_tensor = interp_landmark->tensor(out2_landmarks_idx);
    out2_dims = out2_tensor->dims;
    out2_shape = std::vector<int>(out2_dims->data, out2_dims->data + out2_dims->size);
    out2_size = 1;
    for (size_t i = 0; i < out2_shape.size(); ++i) { 
        out2_size = out2_size * out2_shape[i];
    }
    if (DEBUG) {
        std::cout << "[Landmark.load_model] out2 (landmarks) size=" << out2_size << " shape=["; 
        for (size_t i = 0; i < out2_shape.size(); ++i) {
            std::cout << out2_shape[i];
            if (i != out2_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;        
    }

    if (blaze_app == "blazehandlandmark") {
        out3_tensor = interp_landmark->tensor(out3_handedness_idx);
        out3_dims = out3_tensor->dims;
        out3_shape = std::vector<int>(out3_dims->data, out3_dims->data + out3_dims->size);
        out3_size = 1;
        for (size_t i = 0; i < out3_shape.size(); ++i) { 
            out3_size = out3_size * out3_shape[i];
        }
        if (DEBUG) {
            std::cout << "[Landmark.load_model] out3 (handedness) size=" << out3_size << " shape=["; 
            for (size_t i = 0; i < out3_shape.size(); ++i) {
                std::cout << out3_shape[i];
                if (i != out3_shape.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;        
        }    
    }
        
    input_shape.width = in_shape[1];
    input_shape.height = in_shape[2];
        
    // Update extraction resolution to match model input
    resolution = input_shape.width;
    
    // Configure output shapes based on blaze_app type
    if (blaze_app == "blazehandlandmark") {
        if (this->resolution == 224) { // hand_landmark_lite
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(21, 3);
            output_shape3 = cv::Size(1, 1);
        } else if (this->resolution == 256) { // hand_landmark_v0_07
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(21, 3);
            output_shape3 = cv::Size(1, 1);
        }
    } else if (blaze_app == "blazefacelandmark") {
        output_shape1 = cv::Size(1, 1);
        output_shape2 = cv::Size(468, 3);
    } else if (blaze_app == "blazeposelandmark") {
        if ( out2_size == 124 ) {
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(31, 4);
        } else if ( out2_size == 195) {
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(39, 5);
        }
    }
    
    if (DEBUG) {
        std::cout << "[Landmark.load_model] Input Shape: " << input_shape << std::endl;
        std::cout << "[Landmark.load_model] Output1 Shape: " << output_shape1 << std::endl;
        std::cout << "[Landmark.load_model] Output2 Shape: " << output_shape2 << std::endl;
        if (blaze_app == "blazehandlandmark") {
            std::cout << "[Landmark.load_model] Output3 Shape: " << output_shape3 << std::endl;
        }
        std::cout << "[Landmark.load_model] Input Resolution: " << resolution << std::endl;
    }

    return true;
}

cv::Mat Landmark::preprocess(const cv::Mat& input) {
    // image was already pre-processed by extract_roi in blaze_common/Base.cpp
    // format = RGB
    // dtype = float32
    // range = 0.0 - 1.0
    return input;
}

//std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>>
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
Landmark::predict(const std::vector<cv::Mat>& input_images) {
    
    profile_pre = 0.0;
    profile_model = 0.0;
    profile_post = 0.0;
    
    std::vector<std::vector<double>> out1_list;
    std::vector<std::vector<std::vector<double>>> out2_list;
    std::vector<std::vector<double>> out3_list;

    for (const auto& input : input_images) {
        if (DEBUG) {
            std::cout << "[Landmark.predict] Processing input image of size: " 
                      << input.size() << " channels: " << input.channels() << std::endl;
        }
        
        // 1. Preprocessing
        auto pre_start = std::chrono::high_resolution_clock::now();
        cv::Mat processed_input = preprocess(input);
        auto pre_end = std::chrono::high_resolution_clock::now();
        profile_pre += std::chrono::duration<double>(pre_end - pre_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Preprocessed input size: " 
                      << processed_input.size() << " channels: " << processed_input.channels() << std::endl;
        }
        
        // 2. Convert to input data format for TFLite
        float *input_layer = interp_landmark->typed_tensor<float>(in_idx);
        memcpy(input_layer, processed_input.ptr<float>(0), in_shape[1] * in_shape[2] * in_shape[3] * sizeof(float));
        
        // 3. Inference using TFLite
        auto inference_start = std::chrono::high_resolution_clock::now();
        TfLiteStatus status = interp_landmark->Invoke();
        if (status != kTfLiteOk)
        {
            std::cout << "[Detector.predict_on_batch] Failed to run inference!!" << std::endl;
            exit(1);
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        profile_model = std::chrono::duration<double>(inference_end - inference_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Inference completed, processing results..." << std::endl;
        }
        
        // 4. Process outputs
        auto post_start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> out1;
        std::vector<std::vector<double>> out2;
        std::vector<double> out3;
        
        // out1_confidence    shape: [1, 1] 
        const float* data1 = out1_tensor->data.f;
        out1 = {static_cast<double>(data1[0])};

        // out2_landmarks    shape: [landmarks, 3] 
        int nb_landmarks  = output_shape2.width;
        int nb_components = output_shape2.height;
        size_t idx = 0;
        out2.resize(nb_landmarks);
        idx = 0;
        const float* data2 = out2_tensor->data.f;
        for (int l = 0; l < nb_landmarks; ++l) {
            std::vector<double> elements(nb_components);
            for (int e = 0; e < nb_components; ++e) {
                double val = static_cast<double>(data2[idx++]);
                elements[e] = val;
            }
            out2[l] = elements;
        }        

        if (blaze_app == "blazehandlandmark") {
            // out3_handedness    shape: [1, 1] 
            const float* data3 = out3_tensor->data.f;
            out3 = {static_cast<double>(data3[0])};
        }
        
        auto post_end = std::chrono::high_resolution_clock::now();
        profile_post += std::chrono::duration<double>(post_end - post_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] out1 (confidence) size: " << out1.size() << std::endl;
            std::cout << "[Landmark.predict] out2 (landmarks) size: " << out2.size() << "x" << out2[0].size() << std::endl;
            if (blaze_app == "blazehandlandmark") {
                std::cout << "[Landmark.predict] out3 (handedness) size: " << out3.size() << std::endl;
            }
        }
        
        out1_list.push_back(out1);
        out2_list.push_back(out2);
        if (blaze_app == "blazehandlandmark") {        
            out3_list.push_back(out3);
        }

    }
    
    //return std::make_pair(out1_list, out2_list);
    return std::make_tuple(out1_list, out2_list, out3_list);
}

} // namespace blaze
