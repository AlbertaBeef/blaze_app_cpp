set -x

mkdir -p blaze_hailo/build
rm blaze_hailo/build/*.o

cxxflags="-O3 -pthread -DHAILO_SDK_AVAILABLE"
incflags="-I/usr/include/opencv4 -I/usr/include/hailo -I/opt/hailo/include -I./blaze_common -I./blaze_hailo -I.."

g++ $cxxflags $incflags -c blaze_common/visualization.cpp -o blaze_hailo/build/visualization.o
g++ $cxxflags $incflags -c blaze_common/Config.cpp -o blaze_hailo/build/Config.o
g++ $cxxflags $incflags -c blaze_common/Base.cpp -o blaze_hailo/build/Base.o

g++ $cxxflags $incflags -c blaze_hailo/HailoInference.cpp -o blaze_hailo/build/HailoInference.o
g++ $cxxflags $incflags -c blaze_hailo/Detector.cpp -o blaze_hailo/build/Detector.o
g++ $cxxflags $incflags -c blaze_hailo/Landmark.cpp -o blaze_hailo/build/Landmark.o
g++ $cxxflags $incflags -c blaze_hailo/blaze_detect_live.cpp -o blaze_hailo/build/blaze_detect_live.o

opencv_libs="-lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui"

g++ -o blaze_hailo/blaze_detect_live \
    blaze_hailo/build/blaze_detect_live.o \
    blaze_hailo/build/Config.o \
    blaze_hailo/build/Base.o \
    blaze_hailo/build/visualization.o \
    blaze_hailo/build/Detector.o \
    blaze_hailo/build/Landmark.o \
    blaze_hailo/build/HailoInference.o \
    -L. -Wl,-rpath,. -lhailort $opencv_libs -lpthread

