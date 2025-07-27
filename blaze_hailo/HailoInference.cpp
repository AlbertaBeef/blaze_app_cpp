#include "HailoInference.hpp"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cassert>

#ifdef HAILO_SDK_AVAILABLE
using namespace hailort;
#endif

namespace blaze {

// ============================================================================
// HailoInference Implementation
// ============================================================================

HailoInference::HailoInference() : next_model_id_(0), shutdown_requested_(false) {
#ifdef HAILO_SDK_AVAILABLE
    auto expected_device = VDevice::create();
    if (!expected_device) {
        throw std::runtime_error("Failed to create Hailo VDevice");
    }
    device_ = std::move(expected_device.value());
    std::cout << "[HailoInference] Hailo VDevice created successfully" << std::endl;
#else
    std::cout << "[HailoInference] Mock Hailo VDevice initialized (SDK not available)" << std::endl;
#endif
}

int HailoInference::load_model(const std::string& hef_path) {
    int model_id = next_model_id_++;
    
    std::cout << "[HailoInference.load_model] Loading: " << hef_path << std::endl;
    std::cout << "[HailoInference.load_model] Assigned Model ID: " << model_id << std::endl;
    
    ModelInfo model_info;
    model_info.hef_path = hef_path;
    
#ifdef HAILO_SDK_AVAILABLE
    try {
        // Load HEF file
        auto expected_hef = Hef::create(hef_path);
        if (!expected_hef) {
            throw std::runtime_error("Failed to load HEF file: " + hef_path);
        }
        model_info.hef = std::make_shared<Hef>(std::move(expected_hef.value()));
        
        // Configure network group
        auto expected_network_groups = device_->configure(*model_info.hef);
        if (!expected_network_groups) {
            throw std::runtime_error("Failed to configure network group");
        }
        auto network_groups = expected_network_groups.value();
        if (network_groups.empty()) {
            throw std::runtime_error("No network groups configured");
        }
        model_info.network_group = network_groups[0];
        
        // Get input/output stream info
        auto expected_input_vstream_infos = model_info.hef->get_input_vstream_infos();
        if (!expected_input_vstream_infos) {
            throw std::runtime_error("Failed to get input vstream infos");
        }
        auto input_vstream_infos = expected_input_vstream_infos.value();
        
        auto expected_output_vstream_infos = model_info.hef->get_output_vstream_infos();
        if (!expected_output_vstream_infos) {
            throw std::runtime_error("Failed to get output vstream infos");
        }
        auto output_vstream_infos = expected_output_vstream_infos.value();
        
        for (const auto& info : input_vstream_infos) {
            // Convert hailo_vstream_info_t to VStreamInfo
            std::vector<size_t> shape_vec = {info.shape.height, info.shape.width, info.shape.features};
            model_info.input_infos.emplace_back(info.name, shape_vec);
        }
        
        for (const auto& info : output_vstream_infos) {
            // Convert hailo_vstream_info_t to VStreamInfo
            std::vector<size_t> shape_vec = {info.shape.height, info.shape.width, info.shape.features};
            model_info.output_infos.emplace_back(info.name, shape_vec);
        }
        
        // Create VStreams once during model loading (but with optimized approach)
        std::cout << "[HailoInference.load_model] Creating VStreams..." << std::endl;
        
        // Create VStreams using Hailo SDK: obtain vstream parameters and request FLOAT32 outputs
        auto expected_input_params = model_info.network_group->make_input_vstream_params(
            false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!expected_input_params) {
            throw std::runtime_error("Failed to get input vstream params");
        }
        auto input_params = std::move(expected_input_params.value());
        auto expected_input_vstreams = model_info.network_group->create_input_vstreams(input_params);
        if (!expected_input_vstreams) {
            throw std::runtime_error("Failed to create input VStreams");
        }
        model_info.input_vstreams = std::move(expected_input_vstreams.value());
        std::cout << "[HailoInference.load_model] Created " << model_info.input_vstreams.size() << " input VStreams" << std::endl;

        auto expected_output_params = model_info.network_group->make_output_vstream_params(
            false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!expected_output_params) {
            throw std::runtime_error("Failed to get output vstream params");
        }
        auto output_params = std::move(expected_output_params.value());
        std::cout << "[HailoInference.load_model] Requesting FLOAT32 outputs" << std::endl;
        auto expected_output_vstreams = model_info.network_group->create_output_vstreams(output_params);
        if (!expected_output_vstreams) {
            throw std::runtime_error("Failed to create output VStreams");
        }
        model_info.output_vstreams = std::move(expected_output_vstreams.value());
        std::cout << "[HailoInference.load_model] Created " << model_info.output_vstreams.size() << " output VStreams" << std::endl;
        
        std::cout << "[HailoInference.load_model] Successfully loaded HEF with " 
                  << model_info.input_infos.size() << " inputs and " 
                  << model_info.output_infos.size() << " outputs" << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "[HailoInference.load_model] Error: " << e.what() << std::endl;
        throw;
    }
        // End real SDK branch
#endif  // HAILO_SDK_AVAILABLE
    
    models_[model_id] = std::move(model_info);
    return model_id;
}

std::map<std::string, cv::Mat> 
HailoInference::infer(const std::map<std::string, cv::Mat>& input_data, int hef_id) {
    std::map<std::string, cv::Mat> results;
    
    // Check for shutdown request immediately
    if (shutdown_requested_.load()) {
        return results; // Return empty results
    }
    
    if (models_.find(hef_id) == models_.end()) {
        throw std::runtime_error("Invalid model ID: " + std::to_string(hef_id));
    }
    
    const auto& model_info = models_[hef_id];
    
// #ifdef HAILO_SDK_AVAILABLE
    try {
        // Use pre-created VStreams with optimized approach (no sleep delays)
        auto& input_vstreams = const_cast<std::vector<InputVStream>&>(model_info.input_vstreams);
        auto& output_vstreams = const_cast<std::vector<OutputVStream>&>(model_info.output_vstreams);
        
        // Write input data to all streams first
        for (auto& input_vstream : input_vstreams) {
            // Check for shutdown before processing each stream
            if (shutdown_requested_.load()) {
                return results; // Return empty results
            }
            
            const std::string& stream_name = input_vstream.name();
            if (input_data.find(stream_name) != input_data.end()) {
                const cv::Mat& input_mat = input_data.at(stream_name);
                
                // Convert cv::Mat to uint8 buffer
                cv::Mat input_uint8;
                input_mat.convertTo(input_uint8, CV_8UC3);
                
                size_t data_size = input_uint8.total() * input_uint8.elemSize();
                
                // Create MemoryView for input data
                MemoryView input_buffer(input_uint8.data, data_size);
                auto status = input_vstream.write(input_buffer);
                if (HAILO_SUCCESS != status) {
                    std::cerr << "[HailoInference.infer] Failed to write to input stream " << stream_name 
                              << " with status: " << status << std::endl;
                    throw std::runtime_error("Failed to write to input stream: " + stream_name);
                }
            }
        }
        
        // Read float32 output data immediately (no sleep delay)
        for (auto& output_vstream : output_vstreams) {
            if (shutdown_requested_.load()) {
                return results;
            }
            const std::string& stream_name = output_vstream.name();
            auto info = output_vstream.get_info();
            // Calculate output size
            size_t output_size = info.shape.height * info.shape.width * info.shape.features;
            size_t buffer_bytes = output_size * sizeof(float);
            cv::Mat output_float32(1, static_cast<int>(output_size), CV_32F);
            MemoryView view(output_float32.data, buffer_bytes);
            auto status = output_vstream.read(view);
            if (HAILO_SUCCESS != status) {
                std::cerr << "[HailoInference.infer] Failed to read float32 from " << stream_name
                          << " status: " << status << std::endl;
                continue;
            }
            int h = static_cast<int>(info.shape.height);
            int w = static_cast<int>(info.shape.width);
            int c = static_cast<int>(info.shape.features);
            cv::Mat shaped(h, w, CV_32FC(c), output_float32.data);
            results[stream_name] = shaped.clone();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[HailoInference.infer] Error: " << e.what() << std::endl;
        throw;
    }
        // End real SDK inference branch
// #endif  // HAILO_SDK_AVAILABLE
    
    return results;
}

std::vector<VStreamInfo> HailoInference::get_input_vstream_infos(int hef_id) const {
    if (models_.find(hef_id) == models_.end()) {
        throw std::runtime_error("Invalid model ID: " + std::to_string(hef_id));
    }
    return models_.at(hef_id).input_infos;
}

std::vector<VStreamInfo> HailoInference::get_output_vstream_infos(int hef_id) const {
    if (models_.find(hef_id) == models_.end()) {
        throw std::runtime_error("Invalid model ID: " + std::to_string(hef_id));
    }
    return models_.at(hef_id).output_infos;
}


} // namespace blaze
