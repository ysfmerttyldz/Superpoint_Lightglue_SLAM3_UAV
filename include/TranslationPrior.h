/**
 * TranslationPrior.h
 * 
 * KRİTİK FARK: Monoküler SLAM'in koordinat sistemi ile GT farklı.
 * SLAM: bilinmeyen scale, kendi iç koordinatları
 * GT: metre cinsinden gerçek dünya
 * 
 * Çözüm: Online Sim(3) alignment
 *   1. İlk N frame: SLAM pose + GT biriktir
 *   2. Sim(3) hesapla: GT->SLAM dönüşümü (scale, R, t)
 *   3. GT'leri SLAM koordinatlarına dönüştürüp prior olarak kullan
 */

#ifndef TRANSLATIONPRIOR_H
#define TRANSLATIONPRIOR_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <mutex>

namespace ORB_SLAM3
{

class TranslationPrior
{
public:
    TranslationPrior() : mbLoaded(false), mnPriorFrames(0), mbEnabled(true),
                         mfWeightXY(1.0), mfWeightZ(1.0),
                         mbAligned(false), mfScale(1.0),
                         mMinFramesForAlignment(40),
                         mAlignmentInterval(30),
                         mnSlamObservations(0) 
    {
        mR_gt2slam = Eigen::Matrix3d::Identity();
        mt_gt2slam = Eigen::Vector3d::Zero();
    }

    // ============================================================
    // CSV LOADING
    // ============================================================
    bool LoadCSV(const std::string& csvPath, int maxFrames = 450)
    {
        std::unique_lock<std::mutex> lock(mMutexPrior);
        
        std::ifstream fin(csvPath);
        if (!fin.is_open()) {
            std::cerr << "[TranslationPrior] ERROR: Cannot open " << csvPath << std::endl;
            return false;
        }
        std::cout << "[TranslationPrior] Loading GT from: " << csvPath << std::endl;

        mvGTPoses.clear();
        mmFrameToGT.clear();

        std::string line;
        int dataCount = 0;
        bool headerSkipped = false;

        while (std::getline(fin, line) && dataCount < maxFrames)
        {
            if (line.empty() || line[0] == '#') continue;

            if (!headerSkipped) {
                char first = line[0];
                if (first == 't' || first == 'T' || first == 'x' || first == 'X' || first == 'f' || first == 'F') {
                    headerSkipped = true;
                    continue;
                }
                headerSkipped = true;
            }

            // Virgülle ayır
            std::vector<std::string> fields;
            std::istringstream iss(line);
            std::string field;
            while (std::getline(iss, field, ',')) {
                size_t s = field.find_first_not_of(" \t\r\n");
                size_t e = field.find_last_not_of(" \t\r\n");
                fields.push_back(s != std::string::npos ? field.substr(s, e - s + 1) : "");
            }

            if (fields.size() < 3) continue;

            double x, y, z;
            int frameId = dataCount;
            try {
                x = std::stod(fields[0]);
                y = std::stod(fields[1]);
                z = std::stod(fields[2]);
                if (fields.size() >= 4) {
                    std::string f4 = fields[3];
                    size_t upos = f4.find('_');
                    if (upos != std::string::npos)
                        try { frameId = std::stoi(f4.substr(upos + 1)); } catch (...) {}
                    else
                        try { frameId = std::stoi(f4); } catch (...) {}
                }
            } catch (...) { continue; }

            Eigen::Vector3d gt(x, y, z);
            mvGTPoses.push_back(gt);
            mmFrameToGT[frameId] = gt;
            dataCount++;

            if (dataCount <= 3 || dataCount % 200 == 0)
                std::cout << "[TranslationPrior] Frame " << frameId 
                          << " GT=(" << x << ", " << y << ", " << z << ")" << std::endl;
        }
        fin.close();
        mnPriorFrames = dataCount;
        mbLoaded = (dataCount > 0);
        std::cout << "[TranslationPrior] Loaded " << dataCount << " GT poses" << std::endl;
        return mbLoaded;
    }

    // ============================================================
    // ONLINE SIM(3) — SLAM her frame'de çağırır
    // ============================================================
    
    /**
     * SLAM her frame'de pose hesapladığında bunu çağırır.
     * slamWorldPos = kameranın SLAM koordinat sistemindeki world pozisyonu
     */
    void AddSlamObservation(unsigned long frameId, const Eigen::Vector3d& slamWorldPos)
    {
        std::unique_lock<std::mutex> lock(mMutexPrior);
        if (!mbLoaded || !mbEnabled) return;

        // GT var mı bu frame için?
        Eigen::Vector3d gtPos;
        auto it = mmFrameToGT.find(frameId);
        if (it != mmFrameToGT.end())
            gtPos = it->second;
        else if (frameId < (unsigned long)mnPriorFrames)
            gtPos = mvGTPoses[frameId];
        else
            return;

        mvSlamPositions.push_back(slamWorldPos);
        mvGTPositions.push_back(gtPos);
        mnSlamObservations++;

        // Alignment zamanı?
        bool shouldAlign = (!mbAligned && mnSlamObservations >= mMinFramesForAlignment)
                        || (mbAligned && mnSlamObservations % mAlignmentInterval == 0);
        
        if (shouldAlign)
            ComputeAlignment();
    }

    /**
     * Optimizer'dan çağrılır.
     * GT'yi SLAM koordinat sistemine dönüştürülmüş şekilde döndürür.
     * Alignment yoksa false döner — edge eklenmez.
     */
    bool GetAlignedPrior(unsigned long frameId, Eigen::Vector3d& slamCoordPrior) const
    {
        if (!mbLoaded || !mbEnabled || !mbAligned) return false;
        std::unique_lock<std::mutex> lock(mMutexPrior);

        Eigen::Vector3d gtPos;
        auto it = mmFrameToGT.find(frameId);
        if (it != mmFrameToGT.end())
            gtPos = it->second;
        else if (frameId < (unsigned long)mnPriorFrames)
            gtPos = mvGTPoses[frameId];
        else
            return false;

        // GT -> SLAM dönüşümü: slam = s * R * gt + t
        slamCoordPrior = mfScale * (mR_gt2slam * gtPos) + mt_gt2slam;
        return true;
    }

    bool HasAlignedPrior(unsigned long frameId) const
    {
        if (!mbLoaded || !mbEnabled || !mbAligned) return false;
        std::unique_lock<std::mutex> lock(mMutexPrior);
        return mmFrameToGT.count(frameId) > 0 || frameId < (unsigned long)mnPriorFrames;
    }

    // Getters/Setters
    bool IsLoaded() const { return mbLoaded; }
    bool IsEnabled() const { return mbEnabled; }
    bool IsAligned() const { return mbAligned; }
    int GetNumPriors() const { return mnPriorFrames; }
    int GetNumObservations() const { return mnSlamObservations; }
    double GetScale() const { return mfScale; }
    void SetEnabled(bool e) { mbEnabled = e; }
    void SetWeightXY(double w) { mfWeightXY = w; }
    void SetWeightZ(double w) { mfWeightZ = w; }
    double GetWeightXY() const { return mfWeightXY; }
    double GetWeightZ() const { return mfWeightZ; }
    void SetMinFramesForAlignment(int n) { mMinFramesForAlignment = n; }

    Eigen::Matrix3d GetInformationMatrix() const
    {
        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        info(0,0) = mfWeightXY;
        info(1,1) = mfWeightXY;
        info(2,2) = mfWeightZ;
        return info;
    }

private:

    /**
     * Umeyama Sim(3): GT(source) -> SLAM(target)
     * target = s * R * source + t
     */
    void ComputeAlignment()
    {
        int n = mvSlamPositions.size();
        if (n < 10) return;

        Eigen::Matrix3Xd source(3, n), target(3, n);
        for (int i = 0; i < n; i++) {
            source.col(i) = mvGTPositions[i];
            target.col(i) = mvSlamPositions[i];
        }

        Eigen::Vector3d mu_s = source.rowwise().mean();
        Eigen::Vector3d mu_t = target.rowwise().mean();
        Eigen::Matrix3Xd sc = source.colwise() - mu_s;
        Eigen::Matrix3Xd tc = target.colwise() - mu_t;

        double var_s = sc.squaredNorm() / n;
        if (var_s < 1e-10) {
            std::cerr << "[TranslationPrior] WARN: variance ~0" << std::endl;
            return;
        }

        Eigen::Matrix3d sigma = (tc * sc.transpose()) / n;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Vector3d D = svd.singularValues();

        Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
        if (U.determinant() * V.determinant() < 0) S(2,2) = -1;

        Eigen::Matrix3d R = U * S * V.transpose();
        double s = (S * D.asDiagonal()).trace() / var_s;
        Eigen::Vector3d t = mu_t - s * R * mu_s;

        // RMSE
        Eigen::Matrix3Xd aligned = (s * R * source).colwise() + t;
        double rmse = std::sqrt((aligned - target).squaredNorm() / n);
        double rmse_z = std::sqrt((aligned - target).row(2).squaredNorm() / n);
        double yaw = std::atan2(R(1,0), R(0,0)) * 180.0 / M_PI;

        std::cout << "[TranslationPrior] === Sim(3) Alignment ===" << std::endl;
        std::cout << "  N=" << n << " scale=" << s << " yaw=" << yaw << "deg" << std::endl;
        std::cout << "  t=(" << t.transpose() << ")" << std::endl;
        std::cout << "  RMSE_3D=" << rmse << " RMSE_Z=" << rmse_z << std::endl;

        if (s < 1e-6 || s > 1e6 || std::isnan(s)) {
            std::cerr << "[TranslationPrior] WARN: scale=" << s << " rejected" << std::endl;
            return;
        }

        mfScale = s;
        mR_gt2slam = R;
        mt_gt2slam = t;
        mbAligned = true;
        std::cout << "[TranslationPrior] Alignment ACTIVE ✓" << std::endl;
    }

    bool mbLoaded, mbEnabled, mbAligned;
    int mnPriorFrames;
    double mfWeightXY, mfWeightZ, mfScale;
    int mMinFramesForAlignment, mAlignmentInterval, mnSlamObservations;

    std::vector<Eigen::Vector3d> mvGTPoses;
    std::map<unsigned long, Eigen::Vector3d> mmFrameToGT;

    Eigen::Matrix3d mR_gt2slam;
    Eigen::Vector3d mt_gt2slam;

    std::vector<Eigen::Vector3d> mvSlamPositions, mvGTPositions;
    mutable std::mutex mMutexPrior;
};

} // namespace ORB_SLAM3
#endif // TRANSLATIONPRIOR_H
