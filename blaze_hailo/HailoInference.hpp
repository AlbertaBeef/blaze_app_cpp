#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>

// Hailo SDK includes
#ifdef HAILO_SDK_AVAILABLE
#include "hailo/hailort.hpp"
// Using hailort.hpp which includes vstream, network_group, and defaults
using namespace hailort;
#else
// Mock types when Hailo SDK is not available
namespace hailort {
    enum hailo_status : uint32_t {
        HAILO_SUCCESS = 0
    };
    
    enum hailo_format_type_t {
        HAILO_FORMAT_TYPE_UINT8,
        HAILO_FORMAT_TYPE_FLOAT32
    };
    
    struct hailo_vstream_info_t {
        std::string name;
        std::vector<size_t> shape;
        hailo_format_type_t format_type;
    };
    
    struct hailo_vstream_params_t {
        // Mock parameters structure
    };
    
    class MemoryView {
    public:
        MemoryView(void* /*data*/, size_t /*size*/) {}
    };
    
    template<typename T> 
    class Expected { 
    public: 
        T value() { return T{}; } 
        T release() { return T{}; }
        bool has_value() { return true; }
        operator bool() const { return true; }
    };
    
    // Forward declarations
    class ConfiguredNetworkGroup;
    class ActivatedNetworkGroup;
    
    class Hef {
    public:
        static Expected<Hef> create(const std::string& /*hef_path*/) {
            return Expected<Hef>();
        }
        
        Expected<std::vector<hailo_vstream_info_t>> get_input_vstream_infos() {
            return Expected<std::vector<hailo_vstream_info_t>>();
        }
        
        Expected<std::vector<hailo_vstream_info_t>> get_output_vstream_infos() {
            return Expected<std::vector<hailo_vstream_info_t>>();
        }
    };
    
    class VDevice {
    public:
        static Expected<std::unique_ptr<VDevice>> create() {
            return Expected<std::unique_ptr<VDevice>>();
        }
        
        Expected<std::vector<std::shared_ptr<ConfiguredNetworkGroup>>> configure(const Hef& /*hef*/) {
            return Expected<std::vector<std::shared_ptr<ConfiguredNetworkGroup>>>();
        }
    };
    
    class InputVStream {
    public:
        std::string name() const { return "mock_input"; }
        hailo_status write(const MemoryView& /*buffer*/) { return HAILO_SUCCESS; }
    };
    
    class OutputVStream {
    public:
        std::string name() const { return "mock_output"; }
        hailo_vstream_info_t get_info() const { return {}; }
        hailo_status read(MemoryView /*buffer*/) { return HAILO_SUCCESS; }
    };
    
    class ActivatedNetworkGroup {};
    
    class ConfiguredNetworkGroup {
    public:
        Expected<std::vector<InputVStream>> create_input_vstreams(const std::map<std::string, hailo_vstream_params_t>& /*params*/) {
            return Expected<std::vector<InputVStream>>();
        }
        
        Expected<std::vector<OutputVStream>> create_output_vstreams(const std::map<std::string, hailo_vstream_params_t>& /*params*/) {
            return Expected<std::vector<OutputVStream>>();
        }
        
        Expected<ActivatedNetworkGroup> activate() {
            return Expected<ActivatedNetworkGroup>();
        }
    };
}
#endif

namespace blaze {

/**
 * VStream Info wrapper
 * Abstracts Hailo VStream information
 */
struct VStreamInfo {
    std::string name;
    std::vector<size_t> shape;
    
    VStreamInfo() = default;
    VStreamInfo(const std::string& stream_name, const std::vector<size_t>& stream_shape)
        : name(stream_name), shape(stream_shape) {}
        
#ifdef HAILO_SDK_AVAILABLE
    VStreamInfo(const hailo_vstream_info_t& hailo_info) 
        : name(hailo_info.name) {
        // Convert hailo_3d_image_shape_t to std::vector<size_t>
        shape = {hailo_info.shape.height, hailo_info.shape.width, hailo_info.shape.features};
    }
#endif
};

/**
 * Hailo Inference class
 * Handles Hailo device management and model inference
 */
class HailoInference {
public:
    HailoInference();
    ~HailoInference() = default;
    
    // Model management
    int load_model(const std::string& hef_path);
    
    // Inference
    std::map<std::string, cv::Mat> infer(const std::map<std::string, cv::Mat>& input_data, int hef_id);
    
    // Get model information
    std::vector<VStreamInfo> get_input_vstream_infos(int hef_id) const;
    std::vector<VStreamInfo> get_output_vstream_infos(int hef_id) const;
    
    // Shutdown control
    void request_shutdown() { shutdown_requested_ = true; }
    bool is_shutdown_requested() const { return shutdown_requested_; }
    
private:
    struct ModelInfo {
        std::string hef_path;
        std::vector<VStreamInfo> input_infos;
        std::vector<VStreamInfo> output_infos;
        
#ifdef HAILO_SDK_AVAILABLE
        std::shared_ptr<Hef> hef;
        std::shared_ptr<ConfiguredNetworkGroup> network_group;
        std::vector<InputVStream> input_vstreams;
        std::vector<OutputVStream> output_vstreams;
#endif
    };
    
    std::map<int, ModelInfo> models_;
    int next_model_id_;
    
    // Shutdown control
    std::atomic<bool> shutdown_requested_;
    
#ifdef HAILO_SDK_AVAILABLE
    std::unique_ptr<VDevice> device_;
#endif
};


} // namespace blaze
