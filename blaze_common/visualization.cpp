#include "visualization.hpp"
#include <iostream>

namespace blaze {

// Hand landmark connections (21 points - MediaPipe format)
const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
    // Thumb
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    // Index finger
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    // Middle finger
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    // Ring finger
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    // Pinky finger
    {0, 17}, {17, 18}, {18, 19}, {19, 20}
};

// Face landmark connections (simplified - key facial features)
const std::vector<std::pair<int, int>> FACE_CONNECTIONS = {
    // Face outline (simplified)
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8},
    {8, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16},
    // Eyebrows
    {17, 18}, {18, 19}, {19, 20}, {20, 21},
    {22, 23}, {23, 24}, {24, 25}, {25, 26},
    // Eyes
    {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 36},
    {42, 43}, {43, 44}, {44, 45}, {45, 46}, {46, 47}, {47, 42},
    // Nose
    {27, 28}, {28, 29}, {29, 30}, {31, 32}, {32, 33}, {33, 34}, {34, 35},
    // Mouth
    {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54}, {54, 55}, {55, 56},
    {56, 57}, {57, 58}, {58, 59}, {59, 48}
};

// Pose landmark connections - upper body (simplified)
const std::vector<std::pair<int, int>> POSE_UPPER_BODY_CONNECTIONS = {
    // Head and shoulders
    {0, 1}, {1, 2}, {2, 3}, {3, 7},
    {0, 4}, {4, 5}, {5, 6}, {6, 8},
    // Torso
    {9, 10}, {11, 12}, {11, 13}, {13, 15},
    {12, 14}, {14, 16}, {11, 23}, {12, 24}, {23, 24}
};

// Pose landmark connections - full body
const std::vector<std::pair<int, int>> POSE_FULL_BODY_CONNECTIONS = {
    // Head and shoulders
    {0, 1}, {1, 2}, {2, 3}, {3, 7},
    {0, 4}, {4, 5}, {5, 6}, {6, 8},
    // Torso
    {9, 10}, {11, 12}, {11, 13}, {13, 15},
    {12, 14}, {14, 16}, {11, 23}, {12, 24}, {23, 24},
    // Legs
    {23, 25}, {25, 27}, {27, 29}, {29, 31},
    {24, 26}, {26, 28}, {28, 30}, {30, 32}
};

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections,
                      bool with_keypoints) {
    for (const auto& detection : detections) {
        // Create bounding box from detection coordinates
        cv::Rect bbox(detection.xmin, detection.ymin, 
                     detection.xmax - detection.xmin, 
                     detection.ymax - detection.ymin);
        
        // Draw bounding box
        //cv::rectangle(image, bbox, color, thickness);
        cv::rectangle(image, bbox, cv::Scalar(255, 0, 0), 1);
        
        // Draw score if confidence is reasonable
        if (detection.score > 0.0f) {
            std::string score_text = std::to_string(detection.score).substr(0, 4);
            cv::Point text_pos(bbox.x, bbox.y - 5);
            cv::putText(image, score_text, text_pos, 
                       //cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        }
        
        // Draw keypoints if available
        if (with_keypoints == true) {
            for (const auto& kp : detection.keypoints) {
                cv::circle(image, cv::Point2f(kp.x, kp.y), 2, cv::Scalar(0, 0, 255), 2);
            }
        }        
    }
}

void draw_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                   const std::vector<std::pair<int, int>>& connections,
                   const cv::Scalar& color, int thickness, int radius) {
    
    // Draw landmark points
    for (const auto& point : landmarks) {
        if (point.x >= 0 && point.y >= 0 && 
            point.x < image.cols && point.y < image.rows) {
            cv::circle(image, point, radius, color, -1);
        }
    }
    
    // Draw connections
    for (const auto& connection : connections) {
        int idx1 = connection.first;
        int idx2 = connection.second;
        
        if (idx1 >= 0 && idx1 < static_cast<int>(landmarks.size()) &&
            idx2 >= 0 && idx2 < static_cast<int>(landmarks.size())) {
            
            const cv::Point2f& p1 = landmarks[idx1];
            const cv::Point2f& p2 = landmarks[idx2];
            
            // Check if points are valid
            if (p1.x >= 0 && p1.y >= 0 && p1.x < image.cols && p1.y < image.rows &&
                p2.x >= 0 && p2.y >= 0 && p2.x < image.cols && p2.y < image.rows) {
                cv::line(image, p1, p2, cv::Scalar(0, 0, 0), thickness);
            }
        }
    }
}

void draw_roi(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& roi_boxes) {
    // Each roi_box should be a vector of 4 cv::Point2f points: {P1, P2, P3, P4}
    for (const auto& box : roi_boxes) {
        if (box.size() != 4) continue; // Ensure each ROI is a quadrilateral

        const cv::Point2f& p1 = box[0];
        const cv::Point2f& p2 = box[1];
        const cv::Point2f& p3 = box[2];
        const cv::Point2f& p4 = box[3];

        // Python color mapping:
        // (0,0,0) - black, (0,255,0) - green
        cv::line(image, p1, p2, cv::Scalar(0,0,0), 2);
        cv::line(image, p1, p3, cv::Scalar(0,255,0), 2);
        cv::line(image, p2, p4, cv::Scalar(0,0,0), 2);
        cv::line(image, p3, p4, cv::Scalar(0,0,0), 2);
    }
}

cv::Mat draw_detection_scores( std::vector<std::vector<double>> detection_scores, double min_score_thresh ) {

    cv::Mat plot = cv::Mat::zeros(500, 500, CV_8UC1);
    if (!detection_scores.empty() && !detection_scores[0].empty()) {
        int num_anchors = static_cast<int>(detection_scores[0].size());
        int xdiv = (num_anchors / 500) + 1;
        for (int i = 1; i < num_anchors; ++i) {
            int x1 = static_cast<int>((i - 1) / xdiv);
            int y1 = static_cast<int>(500 - detection_scores[0][i - 1] * 500);
            int x2 = static_cast<int>(i / xdiv);
            int y2 = static_cast<int>(500 - detection_scores[0][i] * 500);
            cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 1);
        }
        // Draw threshold level
        int x1 = 0;
        int x2 = 499;
        int y1 = static_cast<int>(500 - min_score_thresh * 500);
        int y2 = y1;
        cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 1);
    }
    
    return plot;         
}

// Example latency data
// std::vector<std::string> chart_title; // Each pipeline name
// std::vector<double> prof_resize;
// std::vector<double> prof_detector_pre;
// std::vector<double> prof_detector_model;
// std::vector<double> prof_detector_post;
// std::vector<double> prof_extract_roi;
// std::vector<double> prof_landmark_pre;
// std::vector<double> prof_landmark_model;
// std::vector<double> prof_landmark_post;
// std::vector<double> prof_annotate;

// Colors for each bar (BGR)
const std::vector<cv::Scalar> stacked_bar_generic_colors = {
    cv::Scalar(255, 0, 0),     // Blue
    cv::Scalar(0, 255, 0),     // Green
    cv::Scalar(0, 0, 255),     // Red
    cv::Scalar(255, 255, 0),   // Cyan
    cv::Scalar(255, 0, 255),   // Magenta
    cv::Scalar(0, 255, 255),   // Yellow
    cv::Scalar(128, 128, 128), // Gray
    cv::Scalar(0, 128, 255),   // Orange
    cv::Scalar(0, 0, 0),       // Black
    cv::Scalar(255, 255, 255), // White
};

const std::vector<cv::Scalar> stacked_bar_latency_colors = {
    cv::Scalar(255,   0,   0), // resize         : blue 
    cv::Scalar(  0, 255,   0), // detector_pre   : green
    cv::Scalar(255,   0, 255), // detector_model : magenta
    cv::Scalar(255, 255,   0), // detector_post  : cyan
    cv::Scalar(255,   0,   0), // extract_roi    : blue
    cv::Scalar(  0, 255,   0), // landmark_pre   : green
    cv::Scalar(255,   0, 255), // landmark_model : magenta
    cv::Scalar(255, 255,   0), // landmark_post  : cyan
    cv::Scalar(  0,   0,   0), // annotate       : black
};

const std::vector<cv::Scalar> stacked_bar_performance_colors = {
    cv::Scalar(255,   0, 255), // pipeline_fps   : magenta
};

// Usage example inside your main loop:
/*
// Define labels/descriptors for each latency value
const std::vector<std::string> component_labels = {
    "resize",
    "detector[pre]",
    "detector[model]",
    "detector[post]",
    "extract_roi",
    "landmark[pre]",
    "landmark[model]",
    "landmark[post]",
    "annotate"
};
if (bProfileView) {
    // Gather pipeline latency values into vectors of equal length
    // Each prof_* is vector<double> with values for each pipeline
    std::vector<std::vector<double>> component_values = {
        prof_resize,
        prof_detector_pre,
        prof_detector_model,
        prof_detector_post,
        prof_extract_roi,
        prof_landmark_pre,
        prof_landmark_model,
        prof_landmark_post,
        prof_annotate
    };
    draw_stacked_bar_chart(chart_title, component_values, "Latency (sec)");
}
*/


cv::Mat draw_stacked_bar_chart(
    const std::vector<std::string>& pipeline_titles,
    const std::vector<std::string> component_labels,
    const std::vector<std::vector<double>>& component_values, // [component][pipeline]
    const std::vector<cv::Scalar> component_colors,
    const std::string& chart_name)
{
    int pipelines = static_cast<int>(pipeline_titles.size());
    int components = static_cast<int>(component_labels.size());

    // Find max stacked bar value (sum of components for each pipeline)
    double max_stacked = 0.0;
    for (int i = 0; i < pipelines; ++i) {
        double sum = 0.0;
        for (int j = 0; j < components; ++j) {
            sum += component_values[j][i];
        }
        if (sum > max_stacked) max_stacked = sum;
    }

    // Chart size
    //int chart_width = 700;
    int chart_width = 800;
    int legend_spacing = 10; // Space between chart and legend
    int max_legend_per_line = 4;
    int legend_lines = (components + max_legend_per_line - 1) / max_legend_per_line;
    int legend_line_height = 28;
    int chart_height = 40 * pipelines + 80 + legend_spacing + legend_lines * legend_line_height;
    int left_margin = 160;
    int bar_height = 28;
    int spacing = 12;

    cv::Mat chart(chart_height, chart_width, CV_8UC3, cv::Scalar(255,255,255));
    cv::putText(chart, chart_name, cv::Point(left_margin, 36),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(40,40,40), 2, cv::LINE_AA);

    // Draw y labels (pipeline names)
    for (int i = 0; i < pipelines; ++i) {
        int y = 60 + i * (bar_height + spacing) + bar_height/2 + 5;
        cv::putText(chart, pipeline_titles[i], cv::Point(8, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(40,40,40), 1, cv::LINE_AA);
    }

    // Draw bars (stacked, normalized so max stacked bar fits in chart)
    for (int i = 0; i < pipelines; ++i) {
        int y = 60 + i * (bar_height + spacing);
        int x = left_margin;
        double sum = 0.0;
        for (int j = 0; j < components; ++j) {
            sum += component_values[j][i];
        }
        //double norm_factor = (max_stacked > 0.0) ? ((chart_width - left_margin - 30) / max_stacked) : 0.0;
        double norm_factor = (max_stacked > 0.0) ? ((chart_width - left_margin - 100) / max_stacked) : 0.0;
        int x_local = x;
        for (int j = 0; j < components; ++j) {
            double val = component_values[j][i];
            int bar_w = (norm_factor > 0.0) ? static_cast<int>(val * norm_factor) : 0;
            if (bar_w > 0) {
                cv::rectangle(chart, cv::Rect(x_local, y, bar_w, bar_height), component_colors[j], cv::FILLED);
                // Optionally draw value
                if (val >= 0.001)
                    cv::putText(chart, cv::format("%.3f", val), cv::Point(x_local+4, y+bar_height-8),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1, cv::LINE_AA);
            }
            x_local += bar_w;
        }
        // draw total
        if (true) {
            cv::putText(chart, cv::format("%.3f", sum), cv::Point(x_local+4, y+bar_height-8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(40,40,40), 1, cv::LINE_AA);
        }
    }

    // Draw legend on multiple lines, max 4 per line
    int legend_start_x = left_margin;
    int legend_start_y = chart_height - legend_lines * legend_line_height + 6;
    int legend_item_width = (chart_width - left_margin - 30) / max_legend_per_line;
    for (int line = 0; line < legend_lines; ++line) {
        int leg_y = legend_start_y + line * legend_line_height;
        for (int j = 0; j < max_legend_per_line; ++j) {
            int idx = line * max_legend_per_line + j;
            if (idx >= components)
                break;
            int leg_x = legend_start_x + j * legend_item_width;
            cv::rectangle(chart, cv::Rect(leg_x, leg_y, 20, 18), component_colors[idx], cv::FILLED);
            cv::putText(chart, component_labels[idx], cv::Point(leg_x + 28, leg_y + 16),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(40,40,40), 1, cv::LINE_AA);
        }
    }

    return chart;
}

} // namespace blaze
