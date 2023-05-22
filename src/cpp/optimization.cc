#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <cmath>
#include <limits>
#include <vector>
#include <eigen3/Eigen/Core>

// #include "pybind11/include/pybind11/pybind11.h"
// #include "pybind11/include/pybind11/numpy.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


struct CameraProjectionCostFunctor {

    CameraProjectionCostFunctor(const Eigen::Vector2d observed_2D_point, const Eigen::Vector3d gps_3D_point)
        : observed_2D_point(observed_2D_point), gps_3D_point(gps_3D_point) {}

    template <typename T>
    bool operator()(const T* const camera_pose, const T* const camera_intrinsics, T* residuals) const {

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

    static ceres::CostFunction* Create(const Eigen::Vector2d observed_2D_point, const Eigen::Vector3d gps_3D_point) {
        CameraProjectionCostFunctor* functor = new CameraProjectionCostFunctor(observed_2D_point, gps_3D_point);
        return new ceres::AutoDiffCostFunction<CameraProjectionCostFunctor, 2, 7, 4>(functor);
    }

    const Eigen::Vector2d observed_2D_point;
    const Eigen::Vector3d gps_3D_point;
};


void ReprojectPoints(const double camera_intrinsics[4],
                     const double camera_pose[7],
                     const std::vector<Eigen::Vector3d>& gps_3D_points,
                     std::vector<Eigen::Vector2d>& reprojected_2D_points) {

    const double t[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};
    const double q[4] = {camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]};

    assert(gps_3D_points.size() == reprojected_2D_points.size());
    int num_points = gps_3D_points.size();

    for (size_t i = 0; i < num_points; ++i) {
        // Transform GPS point to image coordinates
        const Eigen::Vector3d& gps = gps_3D_points[i];
        Eigen::Vector3d cam;

        ceres::QuaternionRotatePoint(q, gps.data(), cam.data());

        cam[0] += t[0];
        cam[1] += t[1];
        cam[2] += t[2];

        reprojected_2D_points[i][0] = camera_intrinsics[0] * (cam[0] / cam[2]) + camera_intrinsics[2];
        reprojected_2D_points[i][1] = camera_intrinsics[1] * (cam[1] / cam[2]) + camera_intrinsics[3];
    }
}


int FindCorrespondenceIndex(const Eigen::Vector2d& observed_2D_point,
                            const std::vector<Eigen::Vector2d>& reprojected_2D_points) {

    int closest_index = 0;
    double closest_distance = std::numeric_limits<double>::max();
    const size_t num_points = reprojected_2D_points.size();

    for (size_t i = 0; i < num_points; ++i) {
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
    return closest_index;
}


void cpp_optimize_camera_pose(double camera_intrinsics[4],
                        double camera_pose[7],
                        const std::vector<Eigen::Vector2d>& observed_2D_points,
                        const std::vector<Eigen::Vector3d>& gps_3D_points) {

    const size_t num_reprojected_points = gps_3D_points.size();
    std::vector<Eigen::Vector2d> reprojected_2D_points(num_reprojected_points);

    ReprojectPoints(camera_intrinsics, camera_pose, gps_3D_points, reprojected_2D_points);

    ceres::Problem problem;

    const size_t num_observed_points = observed_2D_points.size();

    for (size_t i = 0; i < num_observed_points; ++i) {

        int correspondence_index = FindCorrespondenceIndex(observed_2D_points[i], reprojected_2D_points);

        ceres::CostFunction* cost_function = CameraProjectionCostFunctor::Create(
            observed_2D_points[i],
            gps_3D_points[correspondence_index]);

        problem.AddResidualBlock(cost_function, nullptr, camera_pose, camera_intrinsics);
    }
    
    // Configure the solver
    ceres::Solver::Options options;
    options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
        // idea: solver type Suite Sparse
    // options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
    // options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
        // sparse normal Koleski
    options.minimizer_progress_to_stdout = true;
    options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Print the results
    std::cout << summary.FullReport() << "\n";
    std::cout << "Camera pose after optimization:\n";
    for (size_t i = 0; i < 7; ++i) {
        std::cout << "camera_pose[" << i << "] = " << camera_pose[i] << "\n";
    }

}


// int main() {

//     double camera_intrinsics[4] = {1400.0, 1400.0, 1000.0, 600.0};
//     double camera_pose[7] = {0.1, 0.1, 0.1, 0.5, -0.5, 0.5, -0.5};

//     std::vector<Eigen::Vector2d> observed_2D_points = {
//         Eigen::Vector2d(1.0, 4.0),
//         Eigen::Vector2d(2.0, 5.0),
//         Eigen::Vector2d(3.0, 6.0)
//     };

//     std::vector<Eigen::Vector3d> gps_3D_points = {
//         Eigen::Vector3d(7.0, 10.0, 13.0),
//         Eigen::Vector3d(8.0, 11.0, 14.0),
//         Eigen::Vector3d(9.0, 12.0, 15.0)
//     };

//     return 0;
// }


py::array py_optimize_camera_pose(py::array_t<double, py::array::c_style | py::array::forcecast> camera_intrinsics,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> camera_pose,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> observed_2D_points,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> gps_3D_points) {

    if (camera_intrinsics.shape(0) != 4) {
        throw std::runtime_error("camera_intrinsics must have 4 elements: fx, fy, cx, cy");
    }
    if (camera_pose.shape(0) != 7) {
        throw std::runtime_error("camera_pose must have 7 elements: x, y, z, qx, qy, qz, qw");
    }
    if (observed_2D_points.shape(1) != 2) {
        throw std::runtime_error("observed_2D_points must have 2 columns: x, y");
    }
    if (gps_3D_points.shape(1) != 3) {
        throw std::runtime_error("gps_3D_points must have 3 columns: x, y, z");
    }

    double camera_intrinsics_arr[4];
    double camera_pose_arr[7];

    std::memcpy(camera_intrinsics_arr, camera_intrinsics.data(), 4 * sizeof(double));
    std::memcpy(camera_pose_arr, camera_pose.data(), 7 * sizeof(double));

    const size_t num_observed_points = observed_2D_points.shape(0);
    const size_t num_gps_points = gps_3D_points.shape(0);

    std::vector<Eigen::Vector2d> observed_2D_points_vec(num_observed_points);
    std::vector<Eigen::Vector3d> gps_3D_points_vec(num_gps_points);

    for (size_t i = 0; i < num_observed_points; ++i) {
        Eigen::Map<Eigen::Vector2d> point(observed_2D_points.mutable_data(i, 0));
        observed_2D_points_vec[i] = point;
    }

    for (size_t i = 0; i < num_gps_points; ++i) {
        Eigen::Map<Eigen::Vector3d> point(gps_3D_points.mutable_data(i, 0));
        gps_3D_points_vec[i] = point;
    }

    // Optimise the camera pose
    cpp_optimize_camera_pose(camera_intrinsics_arr, camera_pose_arr, observed_2D_points_vec, gps_3D_points_vec);


    py::array_t<double> camera_intrinsics_out(4);
    py::array_t<double> camera_pose_out(7);

    // convert back to numpy arrays
    std::memcpy(camera_intrinsics_out.mutable_data(), camera_intrinsics_arr, 4 * sizeof(double));
    std::memcpy(camera_pose_out.mutable_data(), camera_pose_arr, 7 * sizeof(double)); 

    return py::make_tuple(camera_intrinsics_out, camera_pose_out);
}


PYBIND11_MODULE(camera_pose_optimisation, m) {
    m.doc() = "Optimize camera pose using Ceres";
    m.def("optimize_camera_pose", &py_optimize_camera_pose, "Optimize camera pose");
}