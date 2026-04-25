#include "SuperGlue.h"
#include <iostream>
#include <chrono>

namespace ORB_SLAM3
{

SuperGlue::SuperGlue(const std::string &model_path, bool use_cuda, bool use_fp16)
    : mDevice(torch::kCPU), mbLoaded(false), mbFP16(use_fp16)
{
    std::cout << "[SuperGlue] DEBUG: Constructor called" << std::endl;
    std::cout << "[SuperGlue] DEBUG: model_path = " << model_path << std::endl;
    std::cout << "[SuperGlue] DEBUG: use_cuda = " << (use_cuda ? "true" : "false") << std::endl;
    std::cout << "[SuperGlue] DEBUG: use_fp16 = " << (use_fp16 ? "true" : "false") << std::endl;

    if (use_cuda && torch::cuda::is_available()) {
        mDevice = torch::Device(torch::kCUDA);
        std::cout << "[SuperGlue] DEBUG: CUDA is available, using GPU" << std::endl;
    } else {
        std::cout << "[SuperGlue] DEBUG: Using CPU (cuda_available="
                  << torch::cuda::is_available() << ")" << std::endl;
    }

    try {
        std::cout << "[SuperGlue] DEBUG: Loading TorchScript model..." << std::endl;
        mModel = torch::jit::load(model_path, mDevice);
        mModel.eval();
        std::cout << "[SuperGlue] DEBUG: Model loaded and set to eval mode" << std::endl;

        if (mbFP16 && mDevice.is_cuda()) {
            mModel.to(torch::kFloat16);
            std::cout << "[SuperGlue] DEBUG: Model converted to FP16" << std::endl;
        }

        mbLoaded = true;
        std::cout << "[SuperGlue] Model loaded from: " << model_path
                  << " (device: " << (mDevice.is_cuda() ? "CUDA" : "CPU")
                  << ", FP16: " << (mbFP16 && mDevice.is_cuda() ? "ON" : "OFF")
                  << ")" << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "[SuperGlue] ERROR: Failed to load model: " << e.what() << std::endl;
        mbLoaded = false;
    }
}

torch::Tensor SuperGlue::normalizeKeypoints(const std::vector<cv::KeyPoint> &kpts,
                                              const cv::Size &image_size)
{
    // SuperGlue normalize_keypoints formulu (superglue.py'den):
    //   size = [width, height]
    //   center = size / 2
    //   scaling = max(size) * 0.7
    //   normalized = (kpts - center) / scaling

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::zeros({(long)kpts.size(), 2}, opts);
    auto accessor = tensor.accessor<float, 2>();

    float w = (float)image_size.width;
    float h = (float)image_size.height;
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    float scaling = std::max(w, h) * 0.7f;

    for (size_t i = 0; i < kpts.size(); i++) {
        accessor[i][0] = (kpts[i].pt.x - cx) / scaling;
        accessor[i][1] = (kpts[i].pt.y - cy) / scaling;
    }

    return tensor;
}

torch::Tensor SuperGlue::descriptorsToTensor(const cv::Mat &desc)
{
    auto tensor = torch::from_blob(
        (void*)desc.data,
        {desc.rows, desc.cols},
        torch::kFloat32
    ).clone();

    return tensor;
}

torch::Tensor SuperGlue::scoresToTensor(const std::vector<cv::KeyPoint> &kpts)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::zeros({(long)kpts.size()}, opts);
    auto accessor = tensor.accessor<float, 1>();

    for (size_t i = 0; i < kpts.size(); i++) {
        accessor[i] = kpts[i].response;  // SuperPoint confidence score
    }

    return tensor;
}

std::vector<SuperGlueMatch> SuperGlue::match(
    const std::vector<cv::KeyPoint> &kpts0,
    const cv::Mat &desc0,
    const std::vector<cv::KeyPoint> &kpts1,
    const cv::Mat &desc1,
    const cv::Size &image_size)
{
    std::vector<SuperGlueMatch> matches;

    std::cout << "[SuperGlue] DEBUG match(): kpts0=" << kpts0.size()
              << " kpts1=" << kpts1.size()
              << " desc0=" << desc0.rows << "x" << desc0.cols
              << " desc1=" << desc1.rows << "x" << desc1.cols
              << " imgSize=" << image_size.width << "x" << image_size.height
              << std::endl;

    if (!mbLoaded) {
        std::cerr << "[SuperGlue] ERROR: Model not loaded, returning empty matches" << std::endl;
        return matches;
    }
    if (kpts0.empty() || kpts1.empty()) {
        std::cout << "[SuperGlue] DEBUG: Empty keypoints, returning empty matches" << std::endl;
        return matches;
    }

    // Serialize all GPU inference calls across threads
    std::lock_guard<std::mutex> lock(getInferenceMutex());
    std::cout << "[SuperGlue] DEBUG: Acquired inference mutex" << std::endl;

    torch::NoGradGuard no_grad;

    auto t_start = std::chrono::high_resolution_clock::now();

    // === Normalization is done HERE in C++ using SuperGlue's formula ===
    // kpts normalized with: (kpt - center) / (max(w,h) * 0.7)
    auto kp0_tensor = normalizeKeypoints(kpts0, image_size).unsqueeze(0).to(mDevice);  // [1, N, 2]
    auto kp1_tensor = normalizeKeypoints(kpts1, image_size).unsqueeze(0).to(mDevice);  // [1, M, 2]

    // Descriptors: [1, 256, N] channel-first
    auto desc0_tensor = descriptorsToTensor(desc0).t().unsqueeze(0).to(mDevice);        // [1, 256, N]
    auto desc1_tensor = descriptorsToTensor(desc1).t().unsqueeze(0).to(mDevice);        // [1, 256, M]

    // Keypoint scores
    auto scores0_tensor = scoresToTensor(kpts0).unsqueeze(0).to(mDevice);               // [1, N]
    auto scores1_tensor = scoresToTensor(kpts1).unsqueeze(0).to(mDevice);               // [1, M]

    std::cout << "[SuperGlue] DEBUG: Tensor shapes:" << std::endl;
    std::cout << "[SuperGlue] DEBUG:   kp0=" << kp0_tensor.sizes() << " (normalized)" << std::endl;
    std::cout << "[SuperGlue] DEBUG:   kp1=" << kp1_tensor.sizes() << " (normalized)" << std::endl;
    std::cout << "[SuperGlue] DEBUG:   desc0=" << desc0_tensor.sizes() << std::endl;
    std::cout << "[SuperGlue] DEBUG:   desc1=" << desc1_tensor.sizes() << std::endl;
    std::cout << "[SuperGlue] DEBUG:   scores0=" << scores0_tensor.sizes() << std::endl;
    std::cout << "[SuperGlue] DEBUG:   scores1=" << scores1_tensor.sizes() << std::endl;

    // Print normalization stats for first few keypoints
    if (kpts0.size() > 0) {
        auto kp0_cpu = kp0_tensor.squeeze(0).to(torch::kCPU);
        auto kp0_acc = kp0_cpu.accessor<float, 2>();
        std::cout << "[SuperGlue] DEBUG:   kp0[0] normalized=(" << kp0_acc[0][0] << ", " << kp0_acc[0][1]
                  << ") raw=(" << kpts0[0].pt.x << ", " << kpts0[0].pt.y << ")" << std::endl;
    }

    if (mbFP16 && mDevice.is_cuda()) {
        kp0_tensor = kp0_tensor.to(torch::kFloat16);
        kp1_tensor = kp1_tensor.to(torch::kFloat16);
        desc0_tensor = desc0_tensor.to(torch::kFloat16);
        desc1_tensor = desc1_tensor.to(torch::kFloat16);
        scores0_tensor = scores0_tensor.to(torch::kFloat16);
        scores1_tensor = scores1_tensor.to(torch::kFloat16);
        std::cout << "[SuperGlue] DEBUG: Converted all tensors to FP16" << std::endl;
    }

    // Forward pass: 6 inputs (keypoints already normalized by C++)
    //   forward(kpts0, kpts1, desc0, desc1, scores0, scores1) -> (matches [K,2], scores [K])
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(kp0_tensor);
    inputs.push_back(kp1_tensor);
    inputs.push_back(desc0_tensor);
    inputs.push_back(desc1_tensor);
    inputs.push_back(scores0_tensor);
    inputs.push_back(scores1_tensor);

    std::cout << "[SuperGlue] DEBUG: Running forward pass with " << inputs.size() << " inputs..." << std::endl;

    try {
        auto output = mModel.forward(inputs);

        auto t_inference = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(t_inference - t_start).count();
        std::cout << "[SuperGlue] DEBUG: Forward pass completed in " << inference_ms << " ms" << std::endl;

        // Parse output: tuple of (match_indices [K, 2], match_scores [K])
        auto output_tuple = output.toTuple();
        std::cout << "[SuperGlue] DEBUG: Output tuple has " << output_tuple->elements().size() << " elements" << std::endl;

        auto match_indices = output_tuple->elements()[0].toTensor().to(torch::kCPU).to(torch::kInt32);
        auto match_scores = output_tuple->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat32);

        std::cout << "[SuperGlue] DEBUG: match_indices shape=" << match_indices.sizes() << std::endl;
        std::cout << "[SuperGlue] DEBUG: match_scores shape=" << match_scores.sizes() << std::endl;

        int num_matches = match_indices.size(0);
        matches.reserve(num_matches);

        auto idx_acc = match_indices.accessor<int, 2>();
        auto score_acc = match_scores.accessor<float, 1>();

        for (int i = 0; i < num_matches; i++) {
            SuperGlueMatch m;
            m.idx0 = idx_acc[i][0];
            m.idx1 = idx_acc[i][1];
            m.score = score_acc[i];
            matches.push_back(m);
        }

        std::cout << "[SuperGlue] DEBUG: Extracted " << matches.size() << " matches" << std::endl;
        if (!matches.empty()) {
            float min_score = matches[0].score, max_score = matches[0].score;
            for (auto &m : matches) {
                min_score = std::min(min_score, m.score);
                max_score = std::max(max_score, m.score);
            }
            std::cout << "[SuperGlue] DEBUG: Score range [" << min_score << ", " << max_score << "]" << std::endl;
        }

    } catch (const c10::Error &e) {
        std::cerr << "[SuperGlue] ERROR: Inference failed: " << e.what() << std::endl;
    }

    // Ensure all GPU operations complete before releasing mutex
    if (mDevice.is_cuda()) {
        torch::cuda::synchronize();
        std::cout << "[SuperGlue] DEBUG: CUDA synchronized" << std::endl;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "[SuperGlue] DEBUG: Total match() time: " << total_ms << " ms" << std::endl;

    return matches;
}

}  // namespace ORB_SLAM3
