#include <iostream>
#include <bitset>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <boost/math/tools/polynomial.hpp>
#include <boost/math/tools/roots.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <experimental/filesystem>
#include <mgl2/mgl.h>
#include "matplotlibcpp.h"
#include "predict.hpp"
#include "unwrap.hpp"

int main(){
    // define graycode to binary code
    std::unordered_map<int, int> g_grayDecoderBasic = {{0,0},{1,1},{3,2},{2,3},{6,4},{7,5},{5,6},{4,7},{12,8},{13,9},{15,10},{14,11},{10,12},{11,13},{9,14},{8,15}};
    std::unordered_map<int, int> g_grayDecoderAdv = {{0,0},{1,1},{3,1},{2,2},{6,2},{7,3},{5,3},{4,4},{12,4},{13,5},{15,5},{14,6},{10,6},{11,7},{9,7},{8,8},{24,8},{25,9},{27,9},{26,10},{30,10},{31,11},{29,11},{28,12},{20,12},{21,13},{23,13},{22,14},{18,14},{19,15},{17,15},{16,16}};
    // generate gray-code
    PatternGenerator::GenerateGrayCode(600,800,"0000000011111111",cv::Scalar(0,0,0),cv::Scalar(255,255,255));
    PatternGenerator::GenerateGrayCode(600,800,"0000111111110000",cv::Scalar(0,0,0),cv::Scalar(255,255,255));
    PatternGenerator::GenerateGrayCode(600,800,"0011110000111100",cv::Scalar(0,0,0),cv::Scalar(255,255,255));
    PatternGenerator::GenerateGrayCode(600,800,"0110011001100110",cv::Scalar(0,0,0),cv::Scalar(255,255,255));
    PatternGenerator::GenerateGrayCode(600,800,"01100110011001100110011001100110",cv::Scalar(0,0,0),cv::Scalar(255,255,255));

    // method 1: Use one more gray code to revise the unwrapping result
    // PatternGenerator::generate_sin(width,height,period,amplitude,stepnumber)
    PatternGenerator::GenerateSinM1(600,800,8,0.5,3);

    // read the result image, using read file to return a vector of Mat, you should use your own mat
    cv::Mat m1_basicPhase = Calculator::ComputePhaseM1(Calculator::ReadFile(3));

    // calculate the background
    cv::Mat m1_background = Calculator::ComputeNoiseM1(Calculator::ReadFile(3)); 

    // load graycode image using your own image
    cv::Mat m1_grayCode1   = cv::imread("0000000011111111.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m1_grayCode2   = cv::imread("0000111111110000.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m1_grayCode3   = cv::imread("0011110000111100.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m1_grayCode4   = cv::imread("0110011001100110.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m1_grayCode5   = cv::imread("01100110011001100110011001100110.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m1_kValueAdv   = Calculator::ComputeKAdvM1(m1_grayCode1,m1_grayCode2,m1_grayCode3,m1_grayCode4,m1_grayCode5,g_grayDecoderAdv,m1_background);
    cv::Mat m1_kValue      = Calculator::ComputeKValue(m1_grayCode1,m1_grayCode2,m1_grayCode3,m1_grayCode4,g_grayDecoderBasic,m1_background);
    cv::Mat m1_unwrap      = Calculator::UnwrapPhaseM1(m1_basicPhase,m1_kValue,m1_kValueAdv);
    m1_unwrap = (m1_unwrap + 3) *(255/103);
    m1_unwrap.convertTo(m1_unwrap, CV_8UC1);
    cv::imwrite("m1_result.png",m1_unwrap);

    // method 2: Move the phase image to revise the unwrapping result
    PatternGenerator::GenerateSinM2(600, 800, 8, 0.5, 0, "zero_sin");
    PatternGenerator::GenerateSinM2(600, 800, 8, 0.5, 2*M_PI/3, "right_sin");
    PatternGenerator::GenerateSinM2(600, 800, 8, 0.5, 4*M_PI/3, "left_sin");
    cv::Mat m2_sin1         = cv::imread("im01.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_sin2         = cv::imread("im03.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_sin3         = cv::imread("im02.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_grayCode1    = cv::imread("im04.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_grayCode2    = cv::imread("im05.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_grayCode3    = cv::imread("im06.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_grayCode4    = cv::imread("im07.png",cv::IMREAD_GRAYSCALE);
    cv::Mat m2_background   = Calculator::ComputeNoiseM2(m2_sin1,m2_sin2,m2_sin3);
    cv::Mat m2_kValue       = Calculator::ComputeKValue(m2_grayCode1,m2_grayCode2,m2_grayCode3,m2_grayCode4,g_grayDecoderBasic,m2_background);
    cv::Mat m2_basicPhase   = Calculator::ComputePhaseM2(m2_sin3,m2_sin1,m2_sin2); //zero
    cv::Mat m2_leftPhase    = Calculator::ComputeLeftPhaseM2(m2_basicPhase);
    cv::Mat m2_rightPhase   = Calculator::ComputeRightPhaseM2(m2_basicPhase); 
    cv::Mat m2_unwrap       = Calculator::UnwrapPhaseM2(m2_basicPhase, m2_rightPhase,m2_leftPhase,m2_kValue);
    m2_unwrap = (m2_unwrap + 3) *(255/103);
    m2_unwrap.convertTo(m2_unwrap, CV_8UC1);
    cv::imwrite("m2_result.png",m2_unwrap);
    return 0;
}