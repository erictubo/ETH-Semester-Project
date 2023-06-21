#include <iostream>
#include <cmath>
#include <limits>
#include <vector>

#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


struct CameraProjectionCostFunctor {

    CameraProjectionCostFunctor(const Eigen::Vector2d observed_2D_point,
                                const Eigen::Vector3d gps_3D_point,
                                const double* camera_intrinsics)
        : observed_2D_point(observed_2D_point),
          gps_3D_point(gps_3D_point),
          camera_intrinsics(camera_intrinsics) {}

    template <typename T>
    bool operator()(const T* const camera_pose, T* residuals) const {

        // Extract translation vector & quaternion vector from camera pose
        const T t[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};
        const T q[4] = {camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]};

        T gps[3] = {T(gps_3D_point[0]), T(gps_3D_point[1]), T(gps_3D_point[2])};

        T cam[3];
        ceres::QuaternionRotatePoint(q, gps, cam);
        
        cam[0] += t[0];
        cam[1] += t[1];
        cam[2] += t[2];

        // Project 3D point to 2D image coordinates using camera intrinsics (fx, fy, cx, cy)
        const T projected_x = camera_intrinsics[0] * (cam[0] / cam[2]) + camera_intrinsics[2];
        const T projected_y = camera_intrinsics[1] * (cam[1] / cam[2]) + camera_intrinsics[3];

        // Compute residuals
        residuals[0] = projected_x - T(observed_2D_point[0]);
        residuals[1] = projected_y - T(observed_2D_point[1]);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d observed_2D_point,
                                       const Eigen::Vector3d gps_3D_point,
                                       const double* camera_intrinsics){
        CameraProjectionCostFunctor* functor = new CameraProjectionCostFunctor(observed_2D_point, gps_3D_point, camera_intrinsics);
        return new ceres::AutoDiffCostFunction<CameraProjectionCostFunctor, 2, 7>(functor);
    }

    const Eigen::Vector2d observed_2D_point;
    const Eigen::Vector3d gps_3D_point;
    const double* camera_intrinsics;
};


Eigen::Vector2d ReprojectPoint(const Eigen::Vector3d& gps_3D_point,
                               const double camera_pose[7],
                               const double camera_intrinsics[4]) {

    const double fx = camera_intrinsics[0];
    const double fy = camera_intrinsics[1];
    const double cx = camera_intrinsics[2];
    const double cy = camera_intrinsics[3];

    const Eigen::Vector3d t(camera_pose[0], camera_pose[1], camera_pose[2]);
    const Eigen::Vector4d q(camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]);

    Eigen::Vector3d cam_3D_point;
    ceres::QuaternionRotatePoint(q.data(), gps_3D_point.data(), cam_3D_point.data());

    cam_3D_point += t;

    // ignore point if behind camera (z < 0), fill in with NAN
    if (cam_3D_point[2] < 0) {
        return Eigen::Vector2d(NAN, NAN);
    }
    Eigen::Vector2d reprojected_2d_point(fx * (cam_3D_point[0] / cam_3D_point[2]) + cx,
                                        fy * (cam_3D_point[1] / cam_3D_point[2]) + cy);

    // IDEA: pass on image dimensions as part of intrinsics vector (last two elements)
    // or simply check size of image

    // ignore point if outside of image dimensions, fill in with NAN
    if (reprojected_2d_point[0] < 0 || reprojected_2d_point[0] >= 1920 ||
        reprojected_2d_point[1] < 0 || reprojected_2d_point[1] >= 1080) {
        return Eigen::Vector2d(NAN, NAN);
    }
    return reprojected_2d_point;
};


void ReprojectPoints(const double camera_intrinsics[4],
                     const double camera_pose[7],
                     const std::vector<Eigen::Vector3d>& gps_3D_points,
                     std::vector<Eigen::Vector2d>& reprojected_2D_points) {

    assert(gps_3D_points.size() == reprojected_2D_points.size());
    int num_points = gps_3D_points.size();

    for (size_t i = 0; i < num_points; ++i) {
        reprojected_2D_points[i] = ReprojectPoint(gps_3D_points[i], camera_pose, camera_intrinsics);
    }
}


int FindCorrespondenceIndex(const Eigen::Vector2d& observed_2D_point,
                            const std::vector<Eigen::Vector2d>& reprojected_2D_points) {

    int closest_index = 0;
    double closest_distance = std::numeric_limits<double>::max();
    const size_t num_points = reprojected_2D_points.size();

    for (size_t i = 0; i < num_points; ++i) {
        // if point is not NAN
        if (std::isnan(reprojected_2D_points[i][0]) || std::isnan(reprojected_2D_points[i][1])) {
            continue;
        }
        else {
            // Calculate distance to observed point
            double dx = reprojected_2D_points[i][0] - observed_2D_point[0];
            double dy = reprojected_2D_points[i][1] - observed_2D_point[1];
            double distance = std::sqrt(dx * dx + dy * dy);

            // Update closest point if distance is smaller
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_index = static_cast<int>(i);
            }
        }
    }
    return closest_index;
}


void VisualiseCorrespondences(const std::vector<int>& correspondence_indices,
                              const std::vector<Eigen::Vector2d>& observed_2D_points,
                              const std::vector<Eigen::Vector2d>& reprojected_2D_points,
                              cv::Mat& visualisation) {
    
    for (int i = 0; i < correspondence_indices.size(); ++i) {
        int j = correspondence_indices[i];

        cv::line(visualisation,
                 cv::Point2d(observed_2D_points[i][0], observed_2D_points[i][1]),
                 cv::Point2d(reprojected_2D_points[j][0], reprojected_2D_points[j][1]),
                 cv::Scalar(0, 255, 0));

        cv::circle(visualisation, cv::Point2d(observed_2D_points[i][0], observed_2D_points[i][1]), 3, cv::Scalar(255, 0, 0), -1);
        //cv::circle(visualisation, cv::Point2d(reprojected_2D_points[j][0], reprojected_2D_points[j][1]), 3, cv::Scalar(0, 0, 255), -1);
        }

    for (int i = 0; i < reprojected_2D_points.size(); ++i) {
        cv::circle(visualisation, cv::Point2d(reprojected_2D_points[i][0], reprojected_2D_points[i][1]), 3, cv::Scalar(0, 0, 255), -1);
    }
}


void SaveVisualisation(const cv::Mat& visualisation,
                       const std::string filename,
                       const int iteration) {
    
    std::string path = "/Users/eric/Developer/Cam2GPS/visualisation/optimization/" + filename + "_it_" + std::to_string(iteration) + ".png";
    cv::imwrite(path, visualisation);
}


void cpp_optimize_camera_pose(const std::string filename,
                              const cv::Mat& image,
                              std::vector<Eigen::Vector2d>& observed_2D_points,
                              const std::vector<Eigen::Vector3d>& gps_3D_points,
                              const double camera_intrinsics[4],
                              double camera_pose[7]) {

    // check that all observed points are inside image, go backwards to avoid index errors
    // for (int i = observed_2D_points.size() - 1; i >= 0; --i) {
    //     if (observed_2D_points[i][0] < 50 || observed_2D_points[i][0] > 1920 - 50 ||
    //         observed_2D_points[i][1] < 50 || observed_2D_points[i][1] > 1200 - 50) {
    //         std::cout << "point " << observed_2D_points[i].transpose() << " is outside image, removing\n";
    //         observed_2D_points.erase(observed_2D_points.begin() + i);
    //     }
    // }


    // // print camera intrinsics
    // std::cout << "camera intrinsics [fx, fy, cx, cy]:\n";
    // for (int i = 0; i < 4; ++i) {
    //     std::cout << camera_intrinsics[i] << " ";
    // }

    // // Tests to see data is being passed & processed correctly
    // const Eigen::Vector3d t(camera_pose[0], camera_pose[1], camera_pose[2]);
    // std::cout << "translation:\n" << t.transpose() << std::endl;

    // const Eigen::Vector4d q(camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]);
    // std::cout << "quaternion:\n" << q.transpose() << std::endl;

    // // rotate points (1, 0, 0), (0, 1, 0), (0, 0, 1) by quaternion
    // for (int i = 0; i < 3; ++i) {
    //     Eigen::Vector3d point(i == 0, i == 1, i == 2);
    //     point += t;
    //     Eigen::Vector3d rotated_point;
    //     ceres::QuaternionRotatePoint(q.data(), point.data(), rotated_point.data());
    //     std::cout << "point gps: " << point.transpose() << " -> point cam: " << rotated_point.transpose() << std::endl;
    // }

    // // print first 2 gps 3d points
    // std::cout << "gps 3d points C++:\n";
    // for (int i = 0; i < 2; ++i) {
    //     std::cout << gps_3D_points[i].transpose() << std::endl;
    // }


    const size_t num_reprojected_points = gps_3D_points.size();
    std::vector<Eigen::Vector2d> reprojected_2D_points(num_reprojected_points);

    // multiple ICP iterations with correspondence index updates
    for (int iteration = 0; iteration < 10; ++iteration) {

        ceres::Problem problem;

        ReprojectPoints(camera_intrinsics, camera_pose, gps_3D_points, reprojected_2D_points);

        const size_t num_observed_points = observed_2D_points.size();

        std::vector<int> correspondence_indices(num_reprojected_points);
        std::vector<int> correspondence_count(num_observed_points, 0);

        // Find correspondence for each observed point
        for (size_t i = 0; i < num_observed_points; ++i) {

            correspondence_indices[i] = FindCorrespondenceIndex(observed_2D_points[i], reprojected_2D_points);
            correspondence_count[correspondence_indices[i]] += 1;

            ceres::CostFunction* cost_function = CameraProjectionCostFunctor::Create(
                observed_2D_points[i],
                gps_3D_points[correspondence_indices[i]],
                camera_intrinsics);

            problem.AddResidualBlock(cost_function, nullptr, camera_pose);
        }

        // print observed points with too many correspondences
        for (int i = 0; i < num_observed_points; ++i) {
            if (correspondence_count[i] > 10) {
                std::cout << "observed point: " << observed_2D_points[i].transpose() << " has " << correspondence_count[i] << " correspondences\n";
            }
        }

        std::cout << "A" << std::endl;
        
        cv::Mat visualisation = image.clone();
        VisualiseCorrespondences(correspondence_indices, observed_2D_points, reprojected_2D_points, visualisation);
        SaveVisualisation(visualisation, filename, iteration);

        std::cout << "B" << std::endl;
        
        // Configure the solver
        ceres::Solver::Options options;
        options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
            // idea: solver type Suite Sparse
        // options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        // options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
            // sparse normal Koleski
        // options.minimizer_progress_to_stdout = true;
        options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;

        std::cout << "C" << std::endl;

        // Solve the problem
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "D" << std::endl;

        // Print the results
        // std::cout << summary.FullReport() << "\n";
        // std::cout << "Camera pose after optimization:\n";
        // for (size_t i = 0; i < 7; ++i) {
        //     std::cout << "camera_pose[" << i << "] = " << camera_pose[i] << "\n";

        // Visualize results on image using openCV

    }
}


py::array py_optimize_camera_pose(py::str filename,
                                  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> observed_2D_points,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> gps_3D_points,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> camera_intrinsics,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> camera_pose) {
    
    // check that filename is a string
    // if (!filename.check(py::isinstance<py::str>())) {
    //     throw std::runtime_error("filename must be a string");
    // }

    // check that image is a cv2::Mat
    // if (!image.check(py::isinstance<py::cv2::Mat>())) {
    //     throw std::runtime_error("image must be a cv2::Mat");
    // }

    if (camera_intrinsics.shape()[0] != 4) {
        throw std::runtime_error("camera_intrinsics must have 4 elements: [fx, fy, cx, cy]");
    }
    if (camera_pose.shape()[0] != 7) {
        throw std::runtime_error("camera_pose must have 7 elements: [x, y, z, qx, qy, qz, qw]");
    }
    if (observed_2D_points.shape()[1] != 2) {
        throw std::runtime_error("observed_2D_points must have 2 columns: [u, v]");
    }
    if (gps_3D_points.shape()[1] != 3) {
        throw std::runtime_error("gps_3D_points must have 3 columns: [x, y, z]");
    }

    std::string filename_cpp = filename;

    cv::Mat image_cpp(image.shape(0), image.shape(1), CV_MAKETYPE(CV_8U, image.shape(2)),
                      const_cast<uint8_t*>(image.data()), image.strides(0));

    const size_t num_observed_points = observed_2D_points.shape(0);
    std::vector<Eigen::Vector2d> observed_2D_points_cpp(num_observed_points);
    for (size_t i = 0; i < num_observed_points; ++i) {
        Eigen::Map<Eigen::Vector2d> point(observed_2D_points.mutable_data(i, 0));
        observed_2D_points_cpp[i] = point;
    }

    const size_t num_gps_points = gps_3D_points.shape(0);
    std::vector<Eigen::Vector3d> gps_3D_points_cpp(num_gps_points);
    for (size_t i = 0; i < num_gps_points; ++i) {
        Eigen::Map<Eigen::Vector3d> point(gps_3D_points.mutable_data(i, 0));
        gps_3D_points_cpp[i] = point;
    }

    double camera_intrinsics_cpp[4];
    std::memcpy(camera_intrinsics_cpp, camera_intrinsics.data(), 4 * sizeof(double));

    double camera_pose_cpp[7];
    std::memcpy(camera_pose_cpp, camera_pose.data(), 7 * sizeof(double));


    // Optimise the camera pose
    cpp_optimize_camera_pose(filename_cpp, image_cpp, observed_2D_points_cpp, gps_3D_points_cpp, camera_intrinsics_cpp, camera_pose_cpp);


    py::array_t<double> camera_pose_out(7);

    // convert back to numpy arrays
    std::memcpy(camera_pose_out.mutable_data(), camera_pose_cpp, 7 * sizeof(double)); 

    return camera_pose_out;
}


PYBIND11_MODULE(optimization, m) {
    m.doc() = "Optimize camera pose using Ceres";
    m.def("optimize_camera_pose", &py_optimize_camera_pose, "Optimize camera pose");
}