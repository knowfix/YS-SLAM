/*
* This file is part of YDM-SLAM
* Author: Balveer Singh
* GitHub: https://github.com/balveersinghyt/YDM-SLAM
*/
#include <iostream>
#include "yolov8_seg_onnx.h"
#include <opencv2/core.hpp> // For cv::Scalar
#include<time.h>
#include<opencv2/opencv.hpp>

#include <YoloDetect.h>

// using namespace dnn;
Yolov8SegOnnx		model;
YoloDetection::YoloDetection()
{
    std::cout << "Loading Yolo model..." << std::endl;

    std::string model_path_seg = "/home/knowfix/Downloads/YDM-SLAM/models/yolov8s-seg.onnx";

	// loading model
    if (model.ReadModel(model_path_seg, true)) {
		std:: cout << "read net ok!" << endl;
	}
    else {
        std:: cout << "read net failed!" << endl;
		// return -1;
	}

    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};
}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()
{

    // loading image
    cv::Mat img;
    if(mRGB.empty())
    {
        std::cout << "Read image failed!" << std::endl;
        return -1;
    }
    
    cv:: cvtColor(mRGB, img, cv::COLOR_BGR2RGB);
    if (img.empty()) {
        std::cerr << "Error: Converted image is empty!" << std::endl;
        return false;
    }
    cv:: Mat image = img.clone();

     // Periksa apakah gambar hasil konversi kosong
    std::vector<cv::Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }
    std::vector<OutputParams> result;

    // Periksa apakah hasil deteksi kosong
    // if (result.empty()) {
    //     std::cerr << "Warning: No objects detected!" << std::endl;
    //     return false;
    // }

    objectMask = cv::Mat::zeros(image.size(), CV_8UC1);
    if (objectMask.empty()) {
        std::cerr << "Error: Failed to allocate objectMask!" << std::endl;
        return false;
    }
    // std:: cout<< "objectMask size: " << objectMask.size() << " image size: " << image.size() << endl;
    std::int64_t data;
    data = model.OnnxDetect(image,result);
    // std::cout<<"Cek : "<< data <<endl;
    // std::cout<<"\n"<<endl;
    // for (size_t i = 0; i < result.size(); i++) {
    //     std::cout << "Object " << i + 1 << ":" << std::endl;
    //     std::cout << "  Class ID: " << result[i].id << std::endl;
    //     std::cout << "  Bounding Box: x=" << result[i].box.x
    //               << ", y=" << result[i].box.y
    //               << ", width=" << result[i].box.width
    //               << ", height=" << result[i].box.height << std::endl;
    // }
    if (data) {
        
        mask = cv::Mat::zeros(image.size(), CV_8UC3);
        // Pastikan `mask` dan `objectMask` sudah dialokasikan dengan ukuran yang benar
        if (mask.empty()) {
            std::cerr << "Error: Failed to allocate mask!" << std::endl;
            return false;
        }
        // std::cout<<"mask : "<< mask << endl;
        for (int i = 0; i < result.size(); i++) {
            int left, top;
            int color_num = i;
            if (result[i].box.area() > 0) {
                rectangle(img, result[i].box, color[result[i].id], 2, 8);
                left = result[i].box.x;
                top = result[i].box.y;
            }
            if (result[i].rotatedBox.size.width * result[i].rotatedBox.size.height > 0) {
                DrawRotatedBox(img, result[i].rotatedBox, color[result[i].id], 2);
                left = result[i].rotatedBox.center.x;
                top = result[i].rotatedBox.center.y;
            }
            
            // add masked image to mvDynamicMask
            if (result[i].boxMask.rows && result[i].boxMask.cols > 0){
                mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
            }
            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), model._className[result[i].id])){
                objectMask(result[i].box).setTo(cv::Scalar(255, 255, 255), result[i].boxMask);
                mvDynamicMask.push_back(objectMask);
                cv::Rect2i DynamicArea(left, top, (result[i].box.width), (result[i].box.height));
                mvDynamicArea.push_back(DynamicArea);
            }
            
            cv:: Rect2i DetectArea(left, top, (result[i].box.width), (result[i].box.height));
            mmDetectMap[model._className[result[i].id]].push_back(DetectArea);
           
        }
        if (mvDynamicArea.size() == 0)
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }

    }
    else
        cout << "Detect Failed!" << endl;
    
    return true;
}


void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}
