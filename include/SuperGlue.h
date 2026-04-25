#ifndef SUPERGLUE_H
#define SUPERGLUE_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <mutex>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

namespace ORB_SLAM3
{

struct SuperGlueMatch {
    int idx0;       // keypoint index in image 0
    int idx1;       // keypoint index in image 1
    float score;    // match confidence
};

class SuperGlue {
public:
    SuperGlue(const std::string &model_path, bool use_cuda = true, bool use_fp16 = false);

    // Match two sets of keypoints + descriptors
    // keypoints: [N,2] (x,y) pixel coordinates
    // descriptors: [N,256] float descriptors
    // image_size: (width, height) for coordinate normalization
    std::vector<SuperGlueMatch> match(
        const std::vector<cv::KeyPoint> &kpts0,
        const cv::Mat &desc0,
        const std::vector<cv::KeyPoint> &kpts1,
        const cv::Mat &desc1,
        const cv::Size &image_size);

    bool isLoaded() const { return mbLoaded; }

    // Thread-safe GPU inference mutex (shared across all SuperGlue instances)
    static std::mutex& getInferenceMutex() {
        static std::mutex mtx;
        return mtx;
    }

private:
    // Normalize keypoints using SuperGlue formula: (kpt - center) / (max(w,h) * 0.7)
    torch::Tensor normalizeKeypoints(const std::vector<cv::KeyPoint> &kpts,
                                     const cv::Size &image_size);

    // Convert cv::Mat descriptors to Tensor
    torch::Tensor descriptorsToTensor(const cv::Mat &desc);

    // Convert keypoint scores to Tensor
    torch::Tensor scoresToTensor(const std::vector<cv::KeyPoint> &kpts);

    torch::jit::script::Module mModel;
    torch::Device mDevice;
    bool mbLoaded;
    bool mbFP16;
};

}  // namespace ORB_SLAM3

#endif // SUPERGLUE_H
