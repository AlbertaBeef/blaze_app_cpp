#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cassert>

#include "Detector.hpp"

namespace blaze {

// ============================================================================
// Detector Implementation
// ============================================================================

Detector::Detector(const std::string& blaze_app)
    : DetectorBase()
    , blaze_app(blaze_app)
    , num_inputs(0)
    , num_outputs(0)
    , profile_pre(0.0)
    , profile_model(0.0)
    , profile_post(0.0) { 
}

void Detector::load_model(const std::string& model_path) {
    if (DEBUG) {
        std::cout << "[Detector.load_model] blaze_app= " << blaze_app << std::endl;
        std::cout << "[Detector.load_model] model_path=" << model_path << std::endl;
    }

    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model)
    {
        std::cerr << "[Detector.load_model] FAILED to load model" << model_path.c_str() << std::endl;
        exit(1);
    }    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interp_detector);
    if (!interp_detector) {
        std::cerr << "[Detector.load_model] FAILED to construct interpreter" << std::endl;
        exit(1);
    }
    
    TfLiteStatus status = interp_detector->AllocateTensors();
    if (status != kTfLiteOk)
    {
        std::cerr << "[Detector.load_model] FAILED to allocate the memory for tensors" << std::endl;
        exit(1);
    }
    
    // Get input/output details
    num_inputs = interp_detector->inputs().size();
    num_outputs = interp_detector->outputs().size();

    const std::vector<int>& input_indices = interp_detector->inputs();
    const std::vector<int>& output_indices = interp_detector->outputs();

    if (DEBUG) {
        std::cout << "[Detector.load_model] Number of Inputs : " << num_inputs << std::endl;
        for (int i = 0; i < num_inputs; ++i) {
            int tensor_index = input_indices[i];
            TfLiteTensor* input_tensor = interp_detector->tensor(tensor_index);
            // Now you can use input_tensor, e.g., input_tensor->name, input_tensor->dims, etc.
            std::cout << "[Detector.load_model] Input[" << i << "] "
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
	    
        std::cout << "[Detector.load_model] Number of Outputs : " << num_outputs << std::endl;
        for (int i = 0; i < num_outputs; ++i) {
            int tensor_index = output_indices[i];
            TfLiteTensor* output_tensor = interp_detector->tensor(tensor_index);
            // Now you can use output_tensor, e.g., input_tensor->name, input_tensor->dims, etc.
            std::cout << "[Detector.load_model] Output[" << i << "] "
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
    out_reg_idx = output_indices[0];
    out_clf_idx = output_indices[1];

    in_tensor = interp_detector->tensor(in_idx);
    in_dims = in_tensor->dims;
    in_shape = std::vector<int>(in_dims->data, in_dims->data + in_dims->size);
    if (DEBUG) {
        std::cout << "[Detector.load_model] in_shape = [";
        for (size_t i = 0; i < in_shape.size(); ++i) {
            std::cout << in_shape[i];
            if (i != in_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;        
    }

    out_reg_tensor = interp_detector->tensor(out_reg_idx);
    out_reg_dims = out_reg_tensor->dims;
    out_reg_shape = std::vector<int>(out_reg_dims->data, out_reg_dims->data + out_reg_dims->size);
    if (DEBUG) {
        std::cout << "[Detector.load_model] out_reg_shape = [";
        for (size_t i = 0; i < out_reg_shape.size(); ++i) {
            std::cout << out_reg_shape[i];
            if (i != out_reg_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;        
    }

    out_clf_tensor = interp_detector->tensor(out_clf_idx);
    out_clf_dims = out_clf_tensor->dims;
    out_clf_shape = std::vector<int>(out_clf_dims->data, out_clf_dims->data + out_clf_dims->size);
    if (DEBUG) {
        std::cout << "[Detector.load_model] out_clf_shape = ["; 
        for (size_t i = 0; i < out_clf_shape.size(); ++i) {
            std::cout << out_clf_shape[i];
            if (i != out_clf_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;        
    }

    x_scale = in_shape[1];
    y_scale = in_shape[2];
    h_scale = in_shape[1];
    w_scale = in_shape[2];

    num_anchors = out_clf_shape[1];

    if (DEBUG) {
        std::cout << "[Detector.load_model] Num Anchors : " << num_anchors << std::endl;
    }

    config_model(blaze_app);
}


std::string cv_type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += std::to_string(chans);

    return r;
}

cv::Mat Detector::preprocess(const ImageType& input) {
    cv::Mat preprocessed;

    //std::cout << "[Detector.preprocess] input size=" << input.size << " type=" << cv_type2str(input.type()) << std::endl;
    // [Detector.preprocess] input size=192 x 192 type=8UC3
        
    // Convert to float, scaling values to [0, 1] range
    input.convertTo(preprocessed, CV_32FC3, 1.0 / 255.0);

    //std::cout << "[Detector.preprocess] preprocessed size=" << preprocessed.size << " type=" << cv_type2str(preprocessed.type()) << std::endl;
    // [Detector.preprocess] preprocessed size=192 x 192 type=32FC3


    //double minVal, maxVal;
    //cv::Point minLoc, maxLoc;
    //std::vector<cv::Mat> channels;
    //cv::split(input, channels);    
    //cv::split(preprocessed, channels);    
    //cv::minMaxLoc(channels[1], &minVal, &maxVal, &minLoc, &maxLoc);
    //std::cout << "[Detector.preprocess] G min|max = " << minVal << " " << maxVal << std::endl;
    // for input : 
    //    [Detector.preprocess] G min|max = 0 219
    // for preprocessed : 
    //    [Detector.preprocess] G min|max = 0 0.87451

    
    //if (DEBUG) {
    //    std::cout << "[Detector.preprocess] input(" << cv::typeToString(input.type()) << ") => preprocessed(" << cv::typeToString(preprocessed.type()) << ")" << std::endl;
    //    // [Detector.preprocess] input(CV_8UC3) => preprocessed(CV_32FC3)
    //}
    
    return preprocessed;
}

std::vector<Detection> Detector::predict_on_image(const ImageType& img) {
    // Use resize_pad to handle arbitrary input image sizes
    auto [resized_img, scale, pad] = resize_pad(img);
    
    // Convert single image to batch format
    std::vector<ImageType> batch = {resized_img};
    
    // Call predict_on_batch with properly sized image
    auto detections = predict_on_batch(batch);
    
    // Return first element from batch results
    if (!detections.empty()) {
        return detections[0];
    } else {
        return {};
    }
}

std::vector<std::vector<Detection>> Detector::predict_on_batch(const std::vector<ImageType>& x) {
    profile_pre = 0.0;
    profile_model = 0.0;
    profile_post = 0.0;
    
    // Validate input dimensions
    assert(x[0].channels() == 3);
    assert(x[0].rows == static_cast<int>(y_scale));
    assert(x[0].cols == static_cast<int>(x_scale));
    
    // 1. Preprocess the images
    auto start = std::chrono::high_resolution_clock::now();
    auto preprocessed = preprocess(x[0]);
    float *input_layer = interp_detector->typed_tensor<float>(in_idx);
    memcpy(input_layer, preprocessed.ptr<float>(0), in_shape[1] * in_shape[2] * in_shape[3] * sizeof(float));
    auto end = std::chrono::high_resolution_clock::now();
    profile_pre = std::chrono::duration<double>(end - start).count();

    // 2. Run neural network
    start = std::chrono::high_resolution_clock::now();
    TfLiteStatus status = interp_detector->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "[Detector.predict_on_batch] Failed to run inference!!" << std::endl;
        exit(1);
    }
    end = std::chrono::high_resolution_clock::now();
    profile_model = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();

    int batch_size;
    int nb_elements;
    size_t idx = 0;
    
    // out_clf    shape: [1, num_anchors, 1] 
    batch_size = out_clf_shape[0];
    nb_elements = out_clf_shape[2];
    std::vector<std::vector<std::vector<double>>> out1(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        out1[b].resize(num_anchors);
    }
    idx = 0;
    const float* data1 = out_clf_tensor->data.f;
    for (int b = 0; b < batch_size; ++b) {
        for (int a = 0; a < num_anchors; ++a) {
            double val = static_cast<double>(data1[idx++]);
            out1[b][a] = {val};
        }
    }        

    // out_reg    shape: [1, num_anchors, num_coords]   
    batch_size = out_reg_shape[0];
    std::vector<std::vector<std::vector<double>>> out2(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        out2[b].resize(num_anchors);
    }
    idx = 0;
    const float* data2 = out_reg_tensor->data.f;
    for (int b = 0; b < batch_size; ++b) {
        for (int a = 0; a < num_anchors; ++a) {
            std::vector<double> coords(num_coords);
            for (int e = 0; e < num_coords; ++e) {
                double val = static_cast<double>(data2[idx++]);
                coords[e] = val;
            }
            out2[b][a] = coords;
        }
    }        

    // For DEBUG output, use std::cout or logging as needed

    assert(out1.size() == 1); // batch
    assert(out1[0].size() == num_anchors);
    assert(out1[0][0].size() == 1);

    assert(out2.size() == 1);
    assert(out2[0].size() == num_anchors);
    assert(out2[0][0].size() == num_coords);

    // 3. Postprocess the raw predictions
    std::vector<std::vector<Detection>> detections = tensors_to_detections(out2, out1, anchors_);

    // 4. Non-maximum suppression
    std::vector<std::vector<Detection>> filtered_detections;

    for (size_t i = 0; i < detections.size(); ++i) {
        std::vector<Detection> wnms_detections = weighted_non_max_suppression(detections[i]);
        if (!wnms_detections.empty()) {
            filtered_detections.push_back(wnms_detections);
        }
    }

    end = std::chrono::high_resolution_clock::now();
    profile_post = std::chrono::duration<double>(end - start).count();

    return filtered_detections;
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>>
Detector::process_model_outputs(const std::map<std::string, cv::Mat>& infer_results) {
    
    if (blaze_app == "blazepalm" && num_outputs == 6) {
        //return process_palm_v07_outputs(infer_results);
    } else if (blaze_app == "blazepalm" && num_outputs == 4) {
        //return process_palm_lite_outputs(infer_results);
    } else {
        // Default to lite processing
        //return process_palm_lite_outputs(infer_results);
    }
    
    std::vector<std::vector<std::vector<float>>> out1;
    std::vector<std::vector<std::vector<float>>> out2;
    return std::make_pair(out1, out2);
}


void Detector::set_min_score_threshold(float threshold) {
    min_score_thresh = static_cast<double>(threshold);
    std::cout << "[Detector.set_min_score_threshold] Set threshold to: " << min_score_thresh << std::endl;
}

} // namespace blaze
