set -x

mkdir -p blaze_tflite/build
rm -f blaze_tflite/build/*.o

tflite_include_dir=./blaze_tflite/tflite_libs_cpp/include
tflite_library_dir=./blaze_tflite/tflite_libs_cpp/lib

cxxflags="-O3 -pthread"
incflags="-I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.."

g++ $cxxflags $incflags -c blaze_common/visualization.cpp -o blaze_tflite/build/visualization.o
g++ $cxxflags $incflags -c blaze_common/Config.cpp -o blaze_tflite/build/Config.o
g++ $cxxflags $incflags -c blaze_common/Base.cpp -o blaze_tflite/build/Base.o

g++ $cxxflags $incflags -c blaze_tflite/Detector.cpp -o blaze_tflite/build/Detector.o
g++ $cxxflags $incflags -c blaze_tflite/Landmark.cpp -o blaze_tflite/build/Landmark.o
g++ $cxxflags $incflags -c blaze_tflite/blaze_detect_live.cpp -o blaze_tflite/build/blaze_detect_live.o

opencv_libs="-lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui"

g++ -o blaze_tflite/blaze_detect_live \
    blaze_tflite/build/blaze_detect_live.o \
    blaze_tflite/build/Config.o \
    blaze_tflite/build/Base.o \
    blaze_tflite/build/visualization.o \
    blaze_tflite/build/Detector.o \
    blaze_tflite/build/Landmark.o \
    -L. -L$tflite_library_dir -Wl,-rpath,. -ltensorflowlite $opencv_libs -lpthread

