mkdir -p blaze_tflite/build
rm blaze_tflite/build/*.o

tflite_include_dir=./blaze_tflite/tflite_libs_cpp/include
tflite_library_dir=./blaze_tflite/tflite_libs_cpp/lib
echo tflite include directory : $tflite_include_dir 
echo tflite library directory : $tflite_library_dir 

echo g++ -std=c++17 -Wall -Wextra -O3 -DNDEBUG -I/usr/include/opencv4 -I./blaze_common -I./blaze_tflite -I.. -fPIC -c blaze_common/visualization.cpp -o blaze_tflite/build/visualization.o
g++ -std=c++17 -Wall -Wextra -O3 -DNDEBUG -I/usr/include/opencv4 -I./blaze_common -I./blaze_tflite -I.. -fPIC -c blaze_common/visualization.cpp -o blaze_tflite/build/visualization.o

echo g++ -I. -I/usr/include/opencv4 -I./blaze_common -I./blaze_tflite -I.. -c blaze_common/Config.cpp -o blaze_tflite/build/Config.o
g++ -I. -I/usr/include/opencv4 -I./blaze_common -I./blaze_tflite -I.. -c blaze_common/Config.cpp -o blaze_tflite/build/Config.o

echo g++ -I. -I/usr/include/opencv4 -I./blaze_common -I./blaze_tflite -I.. -c blaze_common/Base.cpp -o blaze_tflite/build/Base.o
g++ -I. -I/usr/include/opencv4 -I./blaze_common -I./blaze_tflite -I.. -c blaze_common/Base.cpp -o blaze_tflite/build/Base.o

echo g++ -I. -I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.. -c blaze_tflite/Detector.cpp -o blaze_tflite/build/Detector.o
g++ -I. -I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.. -c blaze_tflite/Detector.cpp -o blaze_tflite/build/Detector.o

echo g++ -I. -I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.. -c blaze_tflite/Landmark.cpp -o blaze_tflite/build/Landmark.o
g++ -I. -I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.. -c blaze_tflite/Landmark.cpp -o blaze_tflite/build/Landmark.o

echo g++ -I. -I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.. -c -o blaze_tflite/build/blaze_detect_live.o blaze_tflite/blaze_detect_live.cpp
g++ -I. -I/usr/include/opencv4 -I$tflite_include_dir -I./blaze_common -I./blaze_tflite -I.. -c -o blaze_tflite/build/blaze_detect_live.o blaze_tflite/blaze_detect_live.cpp

echo g++ -o blaze_tflite/blaze_detect_live blaze_tflite/build/blaze_detect_live.o blaze_tflite/build/Config.o blaze_tflite/build/Base.o blaze_tflite/build/visualization.o blaze_tflite/build/Detector.o blaze_tflite/build/Landmark.o -L. -L$tflite_library_dir -Wl,-rpath,. -ltensorflowlite -lopencv_stitching -lopencv_alphamat -lopencv_aruco -lopencv_barcode -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_wechat_qrcode -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -lpthread
g++ -o blaze_tflite/blaze_detect_live blaze_tflite/build/blaze_detect_live.o blaze_tflite/build/Config.o blaze_tflite/build/Base.o blaze_tflite/build/visualization.o blaze_tflite/build/Detector.o blaze_tflite/build/Landmark.o -L. -L$tflite_library_dir -Wl,-rpath,. -ltensorflowlite -lopencv_stitching -lopencv_alphamat -lopencv_aruco -lopencv_barcode -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_wechat_qrcode -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -lpthread
