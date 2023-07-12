#include <iostream>
#include <bitset>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <boost/math/tools/polynomial.hpp>
#include <boost/math/tools/roots.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <mgl2/mgl.h>

class PatternGenerator{
    PatternGenerator(){}
public:
    static void GenerateGrayCode(int m, int n, std::string pattern, cv::Scalar color1, cv::Scalar color2){
        assert(n % pattern.length() == 0);
        int stripe_width = n / pattern.length();
        cv::Mat image(m, n, CV_8UC3, color1);
        for(int i = 0; i < pattern.length(); ++i) {
            if (pattern[i] == '1') {
                image(cv::Rect(i * stripe_width, 0, stripe_width, m)).setTo(color2);
            }
        }
        cv::imwrite(pattern + ".png", image);
    }

    static void GenerateSinM2(int m, int n, float k, float a, float t ,std::string name) {
        cv::Mat img(m, n, CV_32F);
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                float x = static_cast<float>(j);
                float y = 60 * std::cos(2 * M_PI * k / img.cols * x + t - M_PI/2) + 100;
                img.at<float>(i, j) = y;
            }
        }
        img.convertTo(img, CV_8UC1);  
        cv::imwrite(name+".png", img);
    }

    static void GenerateSinM1(int m, int n, float k, float a, int step){
        for(int i = 0; i < step; i++){
            cv::Mat img(m, n, CV_32F);
            for (int m = 0; m < img.rows; ++m) {
                for (int j = 0; j < img.cols; ++j) {
                    float x = static_cast<float>(j); 
                    float y = 60 * std::cos(2 * M_PI * k / img.cols * x + i * 2 * M_PI / step - M_PI/2) + 100;
                    img.at<float>(m, j) = y;
                }
            }
            img.convertTo(img, CV_8UC1);
            cv::imwrite(std::to_string(step) + "step-" + std::to_string(i)+".png", img);
        }
    }
};

class Calculator{
    Calculator(){}
public:
    static cv::Mat eigen2cv(const Eigen::MatrixXd &eigen_mat){
        cv::Mat cv_mat(eigen_mat.rows(), eigen_mat.cols(), CV_64F);
        for(int i = 0; i < cv_mat.rows; ++i)
            for(int j = 0; j < cv_mat.cols; ++j)
                cv_mat.at<double>(i, j) = eigen_mat(i, j);
        return cv_mat;
    }

    static std::vector<cv::Mat> ReadFile(int step){
        std::vector<cv::Mat> result;
        for(int i = 0; i < step; i++){
            cv::Mat img = cv::imread(std::to_string(step) + "step-" + std::to_string(i)+".png", cv::IMREAD_GRAYSCALE);
            img.convertTo(img, CV_64F);
            result.push_back(img);
        }
        return result;
    }

    static cv::Mat ComputePhaseM2(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& image3){
        if (image1.size() != image2.size() || image1.size() != image3.size()) {
            throw std::invalid_argument("Images do not have the same size.");
        }
        cv::Mat img1, img2, img3;
        image1.convertTo(img1, CV_64F);
        image2.convertTo(img2, CV_64F);
        image3.convertTo(img3, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        Eigen::MatrixXd emat2(img2.rows, img2.cols);
        Eigen::MatrixXd emat3(img3.rows, img3.cols);

        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
                emat2(i, j) = img2.at<double>(i, j);
                emat3(i, j) = img3.at<double>(i, j);
            }
        }

        Eigen::MatrixXd numerator = emat2 + emat2 - emat1 - emat3; 
        Eigen::MatrixXd denominator = emat1 - emat3;
        Eigen::MatrixXd phase = denominator.array() / numerator.array();
        phase = phase.unaryExpr([](double val) {return std::isinf(val) && val > 0 ? -std::numeric_limits<double>::infinity() : val;});
        phase = phase * sqrt(3);
        phase = phase.unaryExpr([](double elem) { return atan(elem); });
        phase = phase.unaryExpr([](double elem) { return 2*elem; });
        cv::Mat result = eigen2cv(phase);
        return result;
    }

    static cv::Mat ComputePhaseM1(const std::vector<cv::Mat> images){
        std::vector<cv::Mat> images_copy;
        for(int i = 0; i < images.size(); ++i){
            cv::Mat img;
            images[i].convertTo(img, CV_64F);
            images_copy.push_back(img);
        }
        std::vector<Eigen::MatrixXd> emat(images.size(), Eigen::MatrixXd(images[0].rows, images[0].cols));
        for(int i = 0; i < images.size(); ++i){
            for(int j = 0; j < images[i].rows; ++j){
                for(int k = 0; k < images[i].cols; ++k){
                    emat[i](j, k) = images_copy[i].at<double>(j, k);
                }
            }
        }

        Eigen::MatrixXd denominator =  Eigen::MatrixXd::Zero(images[0].rows, images[0].cols);
        for(int i = 0; i < images.size(); ++i){
            Eigen::MatrixXd tmp(images[0].rows, images[0].cols);
            tmp = -1 * emat[i] * std::sin(2 * M_PI * i / images.size());
            denominator += tmp;
        }

        Eigen::MatrixXd numerator = Eigen::MatrixXd::Zero(images[0].rows, images[0].cols);
        for(int i = 0; i < images.size(); ++i){
            Eigen::MatrixXd tmp(images[0].rows, images[0].cols);
            tmp = emat[i] * std::cos(2 * M_PI * i / images.size());
            numerator += tmp;
        }

        Eigen::MatrixXd phase = denominator.array() / numerator.array();
        phase = phase.unaryExpr([](double val) {return std::isinf(val) && val > 0 ? -std::numeric_limits<double>::infinity() : val;});
        phase = phase.unaryExpr([](double elem) { return atan(elem); });
        phase = phase.unaryExpr([](double elem) { return 2*elem; });
        cv::Mat result = eigen2cv(phase);
        return result;
    }

    static cv::Mat ComputeKValue(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat image3, const cv::Mat image4, const std::unordered_map<int, int>& gray2binary, const cv::Mat background){
        if (image1.size() != image2.size() || image1.size() != image3.size() || image1.size() != image4.size()) {
            throw std::invalid_argument("Images do not have the same size.");
        }

        cv::Mat img1, img2, img3, img4;
        image1.convertTo(img1, CV_64F);
        image2.convertTo(img2, CV_64F);
        image3.convertTo(img3, CV_64F);
        image4.convertTo(img4, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        Eigen::MatrixXd emat2(img2.rows, img2.cols);
        Eigen::MatrixXd emat3(img3.rows, img3.cols);
        Eigen::MatrixXd emat4(img4.rows, img4.cols);
        Eigen::MatrixXd ematback(img4.rows, img4.cols);

        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
                emat2(i, j) = img2.at<double>(i, j);
                emat3(i, j) = img3.at<double>(i, j);
                emat4(i, j) = img4.at<double>(i, j);
                ematback(i, j) = background.at<double>(i, j);
            }
        }

        Eigen::MatrixXd k_s = Eigen::MatrixXd::Zero(img1.rows, img1.cols);

        Eigen::MatrixXi emat1_i = (emat1.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat2_i = (emat2.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat3_i = (emat3.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat4_i = (emat4.array() > ematback.array()).cast<int>();

        Eigen::MatrixXi result(emat1.rows(), emat1.cols());

        for (int i = 0; i < emat1.rows(); ++i) {
            for (int j = 0; j < emat1.cols(); ++j) {
                std::bitset<4> binary(((emat1_i(i,j)) << 3) | ((emat2_i(i,j)) << 2) | ((emat3_i(i,j)) << 1) | ((emat4_i(i,j))));
                result(i,j) = static_cast<int>(binary.to_ulong());
            }
        }
        
        result = result.unaryExpr([&gray2binary](int elem) {
            auto iter = gray2binary.find(elem);
            if (iter != gray2binary.end()) {
                return iter->second;
            } else {
                return elem;  
            }
        });

        return eigen2cv(result.cast<double>());
    }

    static cv::Mat ComputeKAdvM1(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat image3, const cv::Mat image4, const cv::Mat image5, const std::unordered_map<int, int>& gray2binary, const cv::Mat background){
        if (image1.size() != image2.size() || image1.size() != image3.size() || image1.size() != image4.size() || image1.size()!= image5.size()) {
            throw std::invalid_argument("Images do not have the same size.");
        }

        cv::Mat img1, img2, img3, img4,img5;
        image1.convertTo(img1, CV_64F);
        image2.convertTo(img2, CV_64F);
        image3.convertTo(img3, CV_64F);
        image4.convertTo(img4, CV_64F);
        image5.convertTo(img5, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        Eigen::MatrixXd emat2(img2.rows, img2.cols);
        Eigen::MatrixXd emat3(img3.rows, img3.cols);
        Eigen::MatrixXd emat4(img4.rows, img4.cols);
        Eigen::MatrixXd emat5(img5.rows, img5.cols);
        Eigen::MatrixXd ematback(img4.rows, img4.cols);

        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
                emat2(i, j) = img2.at<double>(i, j);
                emat3(i, j) = img3.at<double>(i, j);
                emat4(i, j) = img4.at<double>(i, j);
                emat5(i, j) = img5.at<double>(i, j);
                ematback(i, j) = background.at<double>(i, j);
            }
        }

        Eigen::MatrixXd k_s = Eigen::MatrixXd::Zero(img1.rows, img1.cols);

        Eigen::MatrixXi emat1_i = (emat1.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat2_i = (emat2.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat3_i = (emat3.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat4_i = (emat4.array() > ematback.array()).cast<int>();
        Eigen::MatrixXi emat5_i = (emat5.array() > ematback.array()).cast<int>();

        Eigen::MatrixXi result(emat1.rows(), emat1.cols());

        for (int i = 0; i < emat1.rows(); ++i) {
            for (int j = 0; j < emat1.cols(); ++j) {
                std::bitset<5> binary(((emat1_i(i,j)) << 4) | ((emat2_i(i,j)) << 3) | ((emat3_i(i,j)) << 2) | ((emat4_i(i,j)) << 1) | (emat5_i(i,j)));
                result(i,j) = static_cast<int>(binary.to_ulong());
            }
        }

        
        result = result.unaryExpr([&gray2binary](int elem) {
            auto iter = gray2binary.find(elem);
            if (iter != gray2binary.end()) {
                return iter->second;
            } else {
                return elem; 
            }
        });

        return eigen2cv(result.cast<double>());
    }

    static cv::Mat ComputeNoiseM1(const std::vector<cv::Mat> images){
        std::vector<cv::Mat> images_copy;
        for(int i = 0; i < images.size(); ++i){
            cv::Mat img;
            images[i].convertTo(img, CV_64F);
            images_copy.push_back(img);
        }

        std::vector<Eigen::MatrixXd> emat(images.size(), Eigen::MatrixXd(images[0].rows, images[0].cols));
        for(int i = 0; i < images.size(); ++i){
            for(int j = 0; j < images[i].rows; ++j){
                for(int k = 0; k < images[i].cols; ++k){
                    emat[i](j, k) = images_copy[i].at<double>(j, k);
                }
            }
        }

        Eigen::MatrixXd background = Eigen::MatrixXd::Zero(images[0].rows, images[0].cols);

        int mat_size = emat.size();
        for(int i = 0; i < emat.size(); ++i){
            background = background.array() + emat[i].array();
        }
        background = background.unaryExpr([&mat_size](double elem) { return elem/mat_size;});
        return eigen2cv(background);
    }

    static cv::Mat ComputeNoiseM2(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat image3){
        if (image1.size() != image2.size() || image1.size() != image3.size()) {
            throw std::invalid_argument("Images do not have the same size.");
        }
        
        cv::Mat img1, img2, img3;
        image1.convertTo(img1, CV_64F);
        image2.convertTo(img2, CV_64F);
        image3.convertTo(img3, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        Eigen::MatrixXd emat2(img2.rows, img2.cols);
        Eigen::MatrixXd emat3(img3.rows, img3.cols);

        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
                emat2(i, j) = img2.at<double>(i, j);
                emat3(i, j) = img3.at<double>(i, j);
            }
        }

        Eigen::MatrixXd back = emat1.array() + emat2.array() + emat3.array();
        back = back.unaryExpr([](double elem) { return elem/3;});
        return eigen2cv(back);
    }

    static cv::Mat ComputeLeftPhaseM2(const cv::Mat& image1){
        cv::Mat img1;
        image1.convertTo(img1, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
            }
        }
        
        Eigen::MatrixXd phase1 = emat1.unaryExpr([](double elem){ 
            if((elem + 2 * M_PI / 3)> M_PI){
                elem = elem + 2 * M_PI / 3 - 2* M_PI;
            }
            else elem = elem + 2 * M_PI / 3;
            return elem;
        });

        return eigen2cv(phase1);
    }

    static cv::Mat ComputeRightPhaseM2(const cv::Mat& image1){
        cv::Mat img1;
        image1.convertTo(img1, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
            }
        }

        Eigen::MatrixXd phase3 = emat1.unaryExpr([](double elem) { 
            if((elem - 2 * M_PI / 3) <= -1 * M_PI){
                elem = elem - 2 * M_PI / 3 + 2 * M_PI;
            }
            return elem = elem - 2 * M_PI / 3;
        });
        return eigen2cv(phase3);
    }

    static cv::Mat UnwrapPhaseM2(const cv::Mat& image1, const cv::Mat image2, const cv::Mat image3, const cv::Mat& k){
        if (image1.size() != image2.size() || image1.size() != image3.size() || image1.size() != k.size()) {
            throw std::invalid_argument("Images do not have the same size.");
        }
        
        cv::Mat img1, img2, img3, imgk;
        image1.convertTo(img1, CV_64F);
        image2.convertTo(img2, CV_64F);
        image3.convertTo(img3, CV_64F);
        k.convertTo(imgk, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        Eigen::MatrixXd emat2(img2.rows, img2.cols);
        Eigen::MatrixXd emat3(img3.rows, img3.cols);
        Eigen::MatrixXd ematk(imgk.rows, imgk.cols);
        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
                emat2(i, j) = img2.at<double>(i, j);
                emat3(i, j) = img3.at<double>(i, j);
                ematk(i, j) = imgk.at<double>(i, j);
            }
        }

        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(img1.rows, img1.cols);

        for(int i = 0; i < emat1.rows(); ++i) {
            for(int j = 0; j < emat1.cols(); ++j) {
                double val = emat1(i, j);
                if (val < M_PI/3 && val >= -M_PI/3) {
                    result(i, j) = emat1(i,j) + 2 * M_PI * ematk(i,j);
                } 
                else if (val >= M_PI / 3) {
                    result(i, j) = emat2(i,j) + 2 * M_PI * ematk(i,j) + 2 * M_PI / 3;
                } 
                else {
                    result(i, j) = emat3(i,j) + 2 * M_PI * ematk(i,j) - 2 * M_PI / 3;
                }
            }
        }
        return eigen2cv(result);
    }

    static cv::Mat UnwrapPhaseM1(const cv::Mat& image1, const cv::Mat& k1, const cv::Mat& k2){
        if (image1.size() != k1.size() || image1.size() != k2.size()) {
            throw std::invalid_argument("Images do not have the same size.");
        }
        
        cv::Mat img1, imgk1, imgk2;
        image1.convertTo(img1, CV_64F);
        k1.convertTo(imgk1, CV_64F);
        k2.convertTo(imgk2, CV_64F);

        Eigen::MatrixXd emat1(img1.rows, img1.cols);
        Eigen::MatrixXd ematk1(imgk1.rows, imgk1.cols);
        Eigen::MatrixXd ematk2(imgk2.rows, imgk2.cols);

        for(int i = 0; i < img1.rows; ++i){
            for(int j = 0; j < img1.cols; ++j){
                emat1(i, j) = img1.at<double>(i, j);
                ematk1(i, j) = imgk1.at<double>(i, j);
                ematk2(i, j) = imgk2.at<double>(i, j);
            }
        }

        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(img1.rows, img1.cols);

        for(int i = 0; i < emat1.rows(); ++i) {
            for(int j = 0; j < emat1.cols(); ++j) {
                double val = emat1(i, j);
                if (val < M_PI/3 && val > -M_PI/3) {
                    result(i, j) = emat1(i,j) + 2 * M_PI * ematk1(i,j);
                } else if (val >= M_PI / 3) {
                    result(i, j) = emat1(i,j) + 2 * M_PI * ematk2(i,j) - 2 * M_PI;
                } else {
                    result(i, j) = emat1(i,j) + 2 * M_PI * ematk2(i,j);
                }
                if(i == 0 && j % 50 == 0 && (j / 50)%2 != 0){
                }
            }
        }
        return eigen2cv(result);
    }
};

