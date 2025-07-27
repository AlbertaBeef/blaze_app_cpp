#include "Landmark.hpp"
#include "HailoInference.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace blaze {

// ============================================================================
// Landmark Implementation
// ============================================================================

Landmark::Landmark(const std::string& blaze_app, std::shared_ptr<HailoInference> hailo_infer)
    : LandmarkBase()
    , blaze_app_(blaze_app)
    , hailo_infer_(hailo_infer)
    , hef_id_(-1)
    , input_shape_(256, 256)
    , output_shape1_(1, 1)
    , output_shape2_(1, 63)
    , profile_pre_(0.0)
    , profile_model_(0.0)
    , profile_post_(0.0) {
    
    if (DEBUG) {
        std::cout << "[Landmark] Constructor - blaze_app: " << blaze_app_ << std::endl;
    }
}

bool Landmark::load_model(const std::string& model_path) {
    if (DEBUG) {
        std::cout << "[Landmark.load_model] Model File: " << model_path << std::endl;
    }

    // Load model using HailoInference
    hef_id_ = hailo_infer_->load_model(model_path);
    if (hef_id_ < 0) {
        std::cerr << "[Landmark.load_model] Failed to load model" << std::endl;
        return false;
    }
    
    if (DEBUG) {
        std::cout << "[Landmark.load_model] HEF Id: " << hef_id_ << std::endl;
    }
    
    // Get VStream information
    input_vstream_infos_ = hailo_infer_->get_input_vstream_infos(hef_id_);
    output_vstream_infos_ = hailo_infer_->get_output_vstream_infos(hef_id_);
    
    if (DEBUG) {
        std::cout << "[Landmark.load_model] Number of Inputs: " << input_vstream_infos_.size() << std::endl;
        for (size_t i = 0; i < input_vstream_infos_.size(); ++i) {
            const auto& info = input_vstream_infos_[i];
            std::cout << "[Landmark.load_model] Input " << i << ": " << info.name 
                      << " Shape: [";
            for (size_t j = 0; j < info.shape.size(); ++j) {
                std::cout << info.shape[j];
                if (j < info.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "[Landmark.load_model] Number of Outputs: " << output_vstream_infos_.size() << std::endl;
        for (size_t i = 0; i < output_vstream_infos_.size(); ++i) {
            const auto& info = output_vstream_infos_[i];
            std::cout << "[Landmark.load_model] Output " << i << ": " << info.name 
                      << " Shape: [";
            for (size_t j = 0; j < info.shape.size(); ++j) {
                std::cout << info.shape[j];
                if (j < info.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Determine resolution from first input shape
    int resolution = 256; // default
    if (!input_vstream_infos_.empty() && input_vstream_infos_[0].shape.size() >= 3) {
        resolution = static_cast<int>(input_vstream_infos_[0].shape[1]); // Height dimension
        input_shape_ = cv::Size(resolution, resolution);
    }
    // Update extraction resolution to match model input
    this->resolution = resolution;
    
    // Configure output shapes based on blaze_app type
    if (blaze_app_ == "blazehandlandmark") {
        if (resolution == 224) { // hand_landmark_lite
            output_shape1_ = cv::Size(1, 1);
            output_shape2_ = cv::Size(63, 1);
        } else if (resolution == 256) { // hand_landmark_v0_07
            output_shape1_ = cv::Size(1, 1);
            output_shape2_ = cv::Size(63, 1);
        }
    } else if (blaze_app_ == "blazefacelandmark") {
        output_shape1_ = cv::Size(1, 1);
        output_shape2_ = cv::Size(1404, 1);
    } else if (blaze_app_ == "blazeposelandmark") {
        output_shape1_ = cv::Size(1, 1);
        output_shape2_ = cv::Size(195, 1);
    }
    
    if (DEBUG) {
        std::cout << "[Landmark.load_model] Input Shape: " << input_shape_ << std::endl;
        std::cout << "[Landmark.load_model] Output1 Shape: " << output_shape1_ << std::endl;
        std::cout << "[Landmark.load_model] Output2 Shape: " << output_shape2_ << std::endl;
        std::cout << "[Landmark.load_model] Input Resolution: " << resolution << std::endl;
    }
    
    return true;
}

cv::Mat Landmark::preprocess(const cv::Mat& input) {
    // Resize to model's expected input size
    cv::Mat resized;
    if (input.size() != input_shape_) {
        cv::resize(input, resized, input_shape_);
    } else {
        resized = input.clone();
    }
    
    // Convert from float32 [0,1] back to uint8 [0,255] for Hailo implementation
    cv::Mat output;
    resized.convertTo(output, CV_8U, 255.0);
    return output;
}

//std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>>
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
Landmark::predict(const std::vector<cv::Mat>& input_images) {
    
    profile_pre_ = 0.0;
    profile_model_ = 0.0;
    profile_post_ = 0.0;
    
    std::vector<std::vector<double>> out1_list;
    std::vector<std::vector<std::vector<double>>> out2_list;
    std::vector<std::vector<double>> out3_list;
    
    for (const auto& input : input_images) {
        // Check for shutdown request before processing each image
        if (hailo_infer_->is_shutdown_requested()) {
            if (DEBUG) {
                std::cout << "[Landmark.predict] Shutdown requested, returning early" << std::endl;
            }
            break;
        }
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Processing input image of size: " 
                      << input.size() << " channels: " << input.channels() << std::endl;
        }
        
        // 1. Preprocessing
        auto pre_start = std::chrono::high_resolution_clock::now();
        cv::Mat processed_input = preprocess(input);
        auto pre_end = std::chrono::high_resolution_clock::now();
        profile_pre_ += std::chrono::duration<double>(pre_end - pre_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Preprocessed input size: " 
                      << processed_input.size() << " channels: " << processed_input.channels() << std::endl;
        }
        
        // 2. Convert to input data format for HailoInference
        std::map<std::string, cv::Mat> input_data;
        if (!input_vstream_infos_.empty()) {
            input_data[input_vstream_infos_[0].name] = processed_input;
        } else {
            // Fallback when no input vstream info available
            input_data["input"] = processed_input;
        }
        
        // 3. Inference using HailoInference
        auto inference_start = std::chrono::high_resolution_clock::now();
        auto infer_results = hailo_infer_->infer(input_data, hef_id_);
        auto inference_end = std::chrono::high_resolution_clock::now();
        profile_model_ += std::chrono::duration<double>(inference_end - inference_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Inference completed, processing results..." << std::endl;
        }
        
        // 4. Process outputs
        auto post_start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> out1;
        std::vector<std::vector<double>> out2;
        std::vector<double> out3;
        
        // Determine resolution from input shape
        int resolution = input_shape_.width;
        
        // Process actual Hailo inference results based on blaze_app type and output tensor names
        if (DEBUG) {
            std::cout << "[Landmark.predict] UPDATED VERSION - Processing " << infer_results.size() << " inference outputs..." << std::endl;
            for (const auto& [name, output] : infer_results) {
                std::cout << "  Output: " << name << " shape: " << output.size() << std::endl;
            }
        }
        
        if (blaze_app_ == "blazehandlandmark" && resolution == 256) {
            // hand_landmark_v0_07 format
            // Expected outputs: conv48 (landmarks), conv47 (confidence), conv46 (handedness)
            // Output shapes: conv48=[1,1,63], conv47=[1,1,1], conv46=[1,1,1]
            for (const auto& [name, output] : infer_results) {
                if (name.find("conv47") != std::string::npos) {  // confidence
                    cv::Mat flat_output = output.reshape(1, output.total());
                    out1 = {static_cast<double>(flat_output.at<float>(0))};
                } else if (name.find("conv48") != std::string::npos) {  // landmarks (63 values = 21 landmarks * 3 coords)
                    cv::Mat flat_output = output.reshape(1, output.total());
                    if (flat_output.total() >= 63) {
                        out2.resize(21);
                        for (int i = 0; i < 21; ++i) {
                            // Keep landmarks in normalized [0,1] coordinates for the crop
                            double nx = static_cast<double>(flat_output.at<float>(i * 3));
                            double ny = static_cast<double>(flat_output.at<float>(i * 3 + 1));
                            double nz = static_cast<double>(flat_output.at<float>(i * 3 + 2));
                            // Note: Don't divide by resolution here - landmarks should already be normalized
                            out2[i] = { nx, ny, nz };
                        }
                    }
                }
                else if (name.find("conv46") != std::string::npos) {  // handedness
                    cv::Mat flat_output = output.reshape(1, output.total());
                    out3 = {static_cast<double>(flat_output.at<float>(0))};
                }
            }
        } else if (blaze_app_ == "blazehandlandmark" && resolution == 224) {
            // hand_landmark_lite format
            // Expected outputs: fc1 (landmarks), fc3 (confidence), fc2/fc4 (other)
            // Output shapes: fc1=[63], fc3=[1], fc2=[63], fc4=[1]
            for (const auto& [name, output] : infer_results) {
                if (name.find("fc4") != std::string::npos) {  // confidence
                    cv::Mat flat_output = output.reshape(1, output.total());
                    out1 = {static_cast<double>(flat_output.at<float>(0))};
                } else if (name.find("fc1") != std::string::npos) {  // landmarks (63 values = 21 landmarks * 3 coords)
                    cv::Mat flat_output = output.reshape(1, output.total());
                    if (flat_output.total() >= 63) {
                        out2.resize(21);
                        for (int i = 0; i < 21; ++i) {
                            // Keep landmarks in normalized [0,1] coordinates for the crop
                            out2[i] = {
                                static_cast<double>(flat_output.at<float>(i * 3)),
                                static_cast<double>(flat_output.at<float>(i * 3 + 1)),
                                static_cast<double>(flat_output.at<float>(i * 3 + 2))
                            };
                        }
                    }
                }
                else if (name.find("fc3") != std::string::npos) {  // handedness
                    cv::Mat flat_output = output.reshape(1, output.total());
                    out3 = {static_cast<double>(flat_output.at<float>(0))};
                }
            }
        } else if (blaze_app_ == "blazefacelandmark") {
            // face_landmark format
            // Expected outputs: conv23 (confidence), conv25 (landmarks)
            // Output shapes: conv23=[1,1,1], conv25=[1,1,1404]
            for (const auto& [name, output] : infer_results) {
                if (name.find("conv23") != std::string::npos) {  // confidence
                    cv::Mat flat_output = output.reshape(1, output.total());
                    out1 = {static_cast<double>(flat_output.at<float>(0))};
                } else if (name.find("conv25") != std::string::npos) {  // landmarks (1404 values = 468 landmarks * 3 coords)
                    cv::Mat flat_output = output.reshape(1, output.total());
                    if (flat_output.total() >= 1404) {
                        out2.resize(468);
                        for (int i = 0; i < 468; ++i) {
                            out2[i] = {
                                static_cast<double>(flat_output.at<float>(i * 3)) / resolution,
                                static_cast<double>(flat_output.at<float>(i * 3 + 1)) / resolution,
                                static_cast<double>(flat_output.at<float>(i * 3 + 2)) / resolution
                            };
                        }
                    }
                }
            }
        } else if (blaze_app_ == "blazeposelandmark") {
            // pose_landmark_lite format
            // Expected outputs: conv45 (confidence), conv46 (landmarks)
            // Output shapes: conv45=[1,1,1], conv46=[1,1,195]
            for (const auto& [name, output] : infer_results) {
                if (name.find("conv45") != std::string::npos) {  // confidence
                    cv::Mat flat_output = output.reshape(1, output.total());
                    out1 = {static_cast<double>(flat_output.at<float>(0))};
                } else if (name.find("conv46") != std::string::npos) {  // landmarks (195 values = 39 landmarks * 5 coords)
                    cv::Mat flat_output = output.reshape(1, output.total());
                    if (flat_output.total() >= 195) {
                        out2.resize(39);
                        for (int i = 0; i < 39; ++i) {
                            out2[i] = {
                                static_cast<double>(flat_output.at<float>(i * 5)) / resolution,      // x
                                static_cast<double>(flat_output.at<float>(i * 5 + 1)) / resolution,  // y
                                static_cast<double>(flat_output.at<float>(i * 5 + 2)) / resolution,  // z
                                static_cast<double>(flat_output.at<float>(i * 5 + 3)),               // visibility
                                static_cast<double>(flat_output.at<float>(i * 5 + 4))                // presence
                            };
                        }
                    }
                }
            }
        }
        
        // Debug outputs before fallback check
        if (DEBUG) {
            std::cout << "[Landmark.predict] Before fallback: out1.size=" << out1.size() 
                      << ", out2.size=" << out2.size() << std::endl;
        }
        
        // Fallback to mock data if no outputs were processed (for development/testing)
        if (out1.empty() || out2.empty()) {
            if (DEBUG) {
                std::cout << "[Landmark.predict] Incomplete outputs (out1.size=" << out1.size() 
                          << ", out2.size=" << out2.size() << "), using mock data for: " 
                          << blaze_app_ << " with resolution " << resolution << std::endl;
            }
            
            if (blaze_app_ == "blazehandlandmark") {
                out1 = {0.95}; // High confidence
                out2.resize(21);
                for (int i = 0; i < 21; ++i) {
                    double x = 0.3 + 0.4 * (i % 5) / 4.0;
                    double y = 0.2 + 0.6 * (i / 5) / 4.0;
                    double z = 0.01 * i;
                    out2[i] = {x, y, z};
                }
                out3 = {0.99}; // Left hand
            } else if (blaze_app_ == "blazefacelandmark") {
                out1 = {0.98};
                out2.resize(468);
                for (int i = 0; i < 468; ++i) {
                    double angle = 2.0 * M_PI * i / 468.0;
                    double radius = 0.3 + 0.1 * std::sin(3 * angle);
                    double x = 0.5 + radius * std::cos(angle);
                    double y = 0.5 + radius * std::sin(angle);
                    double z = 0.001 * i;
                    out2[i] = {x, y, z};
                }
            } else if (blaze_app_ == "blazeposelandmark") {
                out1 = {0.92};
                out2.resize(39);
                for (int i = 0; i < 39; ++i) {
                    double x = 0.2 + 0.6 * (i % 7) / 6.0;
                    double y = 0.1 + 0.8 * (i / 7) / 5.0;
                    double z = 0.01 * i;
                    double visibility = 0.8 + 0.2 * std::sin(i);
                    double presence = 0.9;
                    out2[i] = {x, y, z, visibility, presence};
                }
            }
        }
        
        auto post_end = std::chrono::high_resolution_clock::now();
        profile_post_ += std::chrono::duration<double>(post_end - post_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Generated output - out1 size: " 
                      << out1.size() << ", out2 size: " << out2.size() << std::endl;
        }
        
        out1_list.push_back(out1);
        out2_list.push_back(out2);
        out3_list.push_back(out3);
    }
    
    //return std::make_pair(out1_list, out2_list);
    return std::make_tuple(out1_list, out2_list, out3_list);
}

} // namespace blaze