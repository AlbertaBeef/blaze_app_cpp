# blaze_app_cpp
C++ application demonstration code for mediapipe models (blazepalm/hand, blazeface, blazepose). 

## Build instructions

### For TFLite:

1. Download the repo and get the submodule:
    ```
    git clone https://github.com/AlbertaBeef/blaze_app_cpp.git
    cd blaze_app_cpp
    git submodule update --init --recursive
    ```

2. Install Bazel:
```sudo apt update && sudo apt install bazel-7.4.1```


3. Clone Tensorflow and build Tensorflow Lite:
    ```
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    ./configure  # Follow prompts, select appropriate options
    bazel build -c opt //tensorflow/lite:libtensorflowlite.so
    ```


4. Build the application:
```make -f Makefile.blaze_tflite all```

### For Hailo
1. Create a login for the [Hailo developer portal](https://hailo.ai/developer-zone/request-access)
2. Download and install the [HailoRT Developer SDK](https://hailo.ai/developer-zone/documentation/hailort-v4-22-0/?sp_referrer=install/install.html#ubuntu-installer-requirements)
3. Build the application:
```make -f Makefile.blaze_hailo all```