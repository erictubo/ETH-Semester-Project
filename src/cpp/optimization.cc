/**
 * @file optimization.cc
 * @brief Implements camera pose optimization and projection routines using Ceres Solver.
 *
 * Contains cost functors, projection utilities, and correspondence finding for camera localization.
 * Depends on Eigen, Ceres, OpenCV, and Pybind11 for Python bindings.
 */

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


/**
 * @brief Cost functor for camera projection residuals.
 *
 * Computes the reprojection error between a 3D GPS point and its observed 2D image location,
 * given camera intrinsics and pose (translation + quaternion).
 */
struct CameraProjectionCostFunctor {

    /**
     * @brief Constructor for CameraProjectionCostFunctor.
     * @param observed_2D_point The observed 2D image point.
     * @param gps_3D_point The corresponding 3D GPS point in world coordinates.
     * @param camera_intrinsics Pointer to camera intrinsics array [fx, fy, cx, cy].
     */
    CameraProjectionCostFunctor(
        const Eigen::Vector2d observed_2D_point,
        const Eigen::Vector3d gps_3D_point,
        const double* camera_intrinsics)
        :
        observed_2D_point(observed_2D_point),
        gps_3D_point(gps_3D_point),
        camera_intrinsics(camera_intrinsics) {}

    /**
     * @brief Operator overload for Ceres automatic differentiation.
     *
     * Computes the residuals for the reprojection error.
     *
     * @tparam T Scalar type for automatic differentiation.
     * @param camera_pose Camera pose parameters [tx, ty, tz, qw, qx, qy, qz].
     * @param residuals Output residuals (2).
     * @return true on success.
     */
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

    /**
     * @brief Factory method to create a Ceres cost function for this functor.
     * @param observed_2D_point The observed 2D image point.
     * @param gps_3D_point The corresponding 3D GPS point in world coordinates.
     * @param camera_intrinsics Pointer to camera intrinsics array [fx, fy, cx, cy].
     * @return Pointer to a new ceres::CostFunction.
     */
    static ceres::CostFunction* Create(
        const Eigen::Vector2d observed_2D_point,
        const Eigen::Vector3d gps_3D_point,
        const double* camera_intrinsics) {
        try {
            CameraProjectionCostFunctor* functor = new CameraProjectionCostFunctor(observed_2D_point, gps_3D_point, camera_intrinsics);
            return new ceres::AutoDiffCostFunction<CameraProjectionCostFunctor, 2, 7>(functor);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << '\n';
            return nullptr;
        }
    }

    const Eigen::Vector2d observed_2D_point; ///< Observed 2D image point
    const Eigen::Vector3d gps_3D_point;      ///< Corresponding 3D GPS point
    const double* camera_intrinsics;         ///< Camera intrinsics [fx, fy, cx, cy]
};


/**
 * @brief Reprojects a 3D GPS point to 2D image coordinates.
 *
 * @param image Input image (for bounds checking).
 * @param camera_intrinsics Camera intrinsics array [fx, fy, cx, cy].
 * @param camera_pose Camera pose as [tx, ty, tz, qw, qx, qy, qz].
 * @param gps_3D_point 3D point in world coordinates.
 * @param reprojected_2D_point Output: 2D point in image coordinates.
 */
void ReprojectPoint(
    const cv::Mat& image,
    const double camera_intrinsics[4],
    const double camera_pose[7],
    const Eigen::Vector3d& gps_3D_point,
    Eigen::Vector2d& reprojected_2D_point) {

    const double t[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};
    const double q[4] = {camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]};

    Eigen::Vector3d cam;
    ceres::QuaternionRotatePoint(q, gps_3D_point.data(), cam.data());

    cam[0] += t[0];
    cam[1] += t[1];
    cam[2] += t[2];

    // ignore points behind camera
    if (cam[2] < 0) {
        reprojected_2D_point = Eigen::Vector2d(NAN, NAN);
    } else {
        reprojected_2D_point[0] = std::round(camera_intrinsics[0] * (cam[0] / cam[2]) + camera_intrinsics[2]);
        reprojected_2D_point[1] = std::round(camera_intrinsics[1] * (cam[1] / cam[2]) + camera_intrinsics[3]);

        // ignore points outside of image
        if (reprojected_2D_point[0] < 0 || reprojected_2D_point[0] >= image.cols ||
            reprojected_2D_point[1] < 0 || reprojected_2D_point[1] >= image.rows) {
            reprojected_2D_point = Eigen::Vector2d(NAN, NAN);
        }
    }
}


/**
 * @brief Reprojects multiple 3D GPS points to 2D image coordinates.
 *
 * @param image Input image (for bounds checking).
 * @param camera_intrinsics Camera intrinsics array [fx, fy, cx, cy].
 * @param camera_pose Camera pose as [tx, ty, tz, qw, qx, qy, qz].
 * @param gps_3D_points Vector of 3D points in world coordinates.
 * @param reprojected_2D_points Output: Vector of 2D points in image coordinates.
 */
void ReprojectPoints(
    const cv::Mat& image,
    const double camera_intrinsics[4],
    const double camera_pose[7],
    const std::vector<Eigen::Vector3d>& gps_3D_points,
    std::vector<Eigen::Vector2d>& reprojected_2D_points) {

    assert(gps_3D_points.size() == reprojected_2D_points.size());

    int num_gps_points = gps_3D_points.size();
    for (size_t i = 0; i < num_gps_points; ++i) {
        ReprojectPoint(image, camera_intrinsics, camera_pose, gps_3D_points[i], reprojected_2D_points[i]);
    }
}


/**
 * @brief Finds the closest reprojected 2D point to an observed 2D point.
 *
 * @param observed_2D_point The observed 2D image point.
 * @param reprojected_2D_points List of candidate reprojected points.
 * @param correspondence_index Output: Index of the closest reprojected point.
 */
void FindCorrespondence(
    const Eigen::Vector2d& observed_2D_point,
    const std::vector<Eigen::Vector2d>& reprojected_2D_points,
    int& correspondence_index) {

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
    correspondence_index = closest_index;
}


// void FindCorrespondences(
//     const std::vector<Eigen::Vector2d>& observed_2D_points,
//     const std::vector<Eigen::Vector2d>& reprojected_2D_points,
//     std::vector<int>& correspondence_indices) {
    
//     assert(correspondence_indices.size() == observed_2D_points.size());

//     const size_t num_observed_points = observed_2D_points.size();

//     for (size_t i = 0; i < num_observed_points; ++i) {
//         FindCorrespondence(observed_2D_points[i], reprojected_2D_points, correspondence_indices[i]);
//     }
// }


/**
 * @brief Finds correspondences between observed and reprojected 2D points.
 *
 * For each observed point, finds the closest reprojected point. If multiple observed points are assigned to the same reprojected point, only the closest is kept.
 *
 * @param observed_2D_points Vector of observed 2D image points.
 * @param reprojected_2D_points Vector of reprojected 2D points.
 * @param correspondence_indices Output: Vector of indices of corresponding reprojected points (or -1 if no correspondence).
 */
void FindCorrespondences(
    const std::vector<Eigen::Vector2d>& observed_2D_points,
    const std::vector<Eigen::Vector2d>& reprojected_2D_points,
    std::vector<int>& correspondence_indices) {
    
    assert(correspondence_indices.size() == observed_2D_points.size());

    const size_t num_observed_points = observed_2D_points.size();
    const size_t num_reprojected_points = reprojected_2D_points.size();

    std::vector<int> correspondence_count(num_reprojected_points, 0);

    // dictionary to track which observed points have been associated with a reprojected point
    std::map<int, std::vector<int>> associated_observed_points;

    for (size_t i = 0; i < num_observed_points; ++i) {

        FindCorrespondence(observed_2D_points[i], reprojected_2D_points, correspondence_indices[i]);

        int j = correspondence_indices[i];
        correspondence_count[j] += 1;

        associated_observed_points[j].push_back(i);
    }

    // Find reprojected points associated with more than one observed point and keep only closest correspondence
    for (size_t j = 0; j < num_reprojected_points; ++j) {
        if (correspondence_count[j] > 1) {
            // find closest observed point of associated observed points
            int closest_index = 0;
            double closest_distance = std::numeric_limits<double>::max();
            for (size_t k = 0; k < associated_observed_points[j].size(); ++k) {
                // Calculate distance to observed point
                double dx = reprojected_2D_points[j][0] - observed_2D_points[associated_observed_points[j][k]][0];
                double dy = reprojected_2D_points[j][1] - observed_2D_points[associated_observed_points[j][k]][1];
                double distance = std::sqrt(dx * dx + dy * dy);

                // Update closest point if distance is smaller
                if (distance < closest_distance) {
                    closest_distance = distance;
                    closest_index = associated_observed_points[j][k];
                }
            }
            // Remove correspondence_index of non-closest points
            for (size_t k = 0; k < associated_observed_points[j].size(); ++k) {
                int i = associated_observed_points[j][k];
                if (i != closest_index) {
                    correspondence_indices[i] = -1;
                }
            }
        }
    }
}


/**
 * @brief Draws correspondences between observed and reprojected 2D points on an image.
 *
 * @param correspondence_indices Indices of corresponding reprojected points for each observed point.
 * @param observed_2D_points Vector of observed 2D image points.
 * @param reprojected_2D_points Vector of reprojected 2D points.
 * @param visualization Output image for visualization.
 */
void DrawCorrespondences(
    const std::vector<int>& correspondence_indices,
    const std::vector<Eigen::Vector2d>& observed_2D_points,
    const std::vector<Eigen::Vector2d>& reprojected_2D_points,
    cv::Mat& visualization) {

    assert(correspondence_indices.size() == observed_2D_points.size());

    // Draw correspondences as lines
    for (int i = 0; i < correspondence_indices.size(); ++i) {
        if (correspondence_indices[i] != -1) {
            cv::line(visualization,
                    cv::Point2d(observed_2D_points[i][0], observed_2D_points[i][1]),
                    cv::Point2d(reprojected_2D_points[correspondence_indices[i]][0], reprojected_2D_points[correspondence_indices[i]][1]),
                    cv::Scalar(0, 255, 0));
        }
    }

    // Draw observed points
    for (int i = 0; i < observed_2D_points.size(); ++i) {
        cv::circle(visualization, cv::Point2d(observed_2D_points[i][0], observed_2D_points[i][1]), 3, cv::Scalar(255, 0, 0), -1);
    }

    // Draw reprojected points (if inside image)
    for (int i = 0; i < reprojected_2D_points.size(); ++i) {
        if (std::isnan(reprojected_2D_points[i][0]) || std::isnan(reprojected_2D_points[i][1])) {
            continue;
        } else {
            cv::circle(visualization, cv::Point2d(reprojected_2D_points[i][0], reprojected_2D_points[i][1]), 3, cv::Scalar(0, 0, 255), -1);
        }
    }
}


/**
 * @brief Saves a visualization image to disk.
 *
 * @param visualization The image to save.
 * @param filename The base filename.
 * @param camera_id The camera identifier.
 * @param iteration The optimization iteration number.
 * @param visualization_path The directory to save the image in.
 */
void SaveVisualization(
    const cv::Mat& visualization,
    const std::string& filename,
    const std::string& camera_id,
    const int& iteration,
    const std::string& visualization_path) {
    
    std::string path = visualization_path + filename + "_cam_" + camera_id + "_it_" + std::to_string(iteration) + ".png";
    cv::imwrite(path, visualization);
}


/**
 * @brief Struct representing a keyframe for optimization.
 *
 * Contains image, observed 2D points, and corresponding 3D GPS points.
 */
struct Keyframe {
    /**
     * @brief Constructor for Keyframe.
     * @param filename The image filename.
     * @param camera_id The camera identifier.
     * @param image The image data.
     * @param observed_2D_points Observed 2D image points.
     * @param gps_3D_points Corresponding 3D GPS points.
     */
    Keyframe(
        const std::string filename,
        const std::string camera_id,
        const cv::Mat& image,
        const std::vector<Eigen::Vector2d>& observed_2D_points,
        const std::vector<Eigen::Vector3d>& gps_3D_points)
        :
        filename(filename),
        image(image),
        observed_2D_points(observed_2D_points),
        gps_3D_points(gps_3D_points) {}

    std::string filename; ///< Image filename
    std::string camera_id; ///< Camera identifier
    cv::Mat image; ///< Image data
    std::vector<Eigen::Vector2d> observed_2D_points; ///< Observed 2D image points
    std::vector<Eigen::Vector3d> gps_3D_points; ///< Corresponding 3D GPS points
};

std::vector<Keyframe> keyframes;

/**
 * @brief Adds a keyframe to the global keyframe list.
 *
 * @param filename The image filename.
 * @param camera_id The camera identifier.
 * @param image The image data.
 * @param observed_2D_points Observed 2D image points.
 * @param gps_3D_points Corresponding 3D GPS points.
 */
void cpp_add_keyframe(
    const std::string filename,
    const std::string camera_id,
    const cv::Mat& image,
    const std::vector<Eigen::Vector2d>& observed_2D_points,
    const std::vector<Eigen::Vector3d>& gps_3D_points) {

    struct Keyframe keyframe(filename, camera_id, image, observed_2D_points, gps_3D_points);
    keyframes.push_back(keyframe);
}

/**
 * @brief Clears all keyframes from the global keyframe list.
 */
void cpp_reset_keyframes() {
    keyframes.clear();
}


/**
 * @brief Runs camera pose optimization over all keyframes.
 *
 * @param camera_pose Input/output: Camera pose [tx, ty, tz, qw, qx, qy, qz].
 * @param camera_intrinsics Camera intrinsics [fx, fy, cx, cy].
 * @param iterations Number of optimization iterations.
 * @param visualization_path Directory to save visualizations.
 */
void cpp_update_camera_pose(
    double camera_pose[7],
    const double camera_intrinsics[4],
    int iterations,
    const std::string& visualization_path) {

    int num_keyframes = keyframes.size();
    std::cout << "num_keyframes: " << num_keyframes << std::endl;

    for (int iteration = 0; iteration < iterations; ++iteration) {

        ceres::Problem problem;

        for (const auto& keyframe : keyframes) {

            const std::string& filename = keyframe.filename;
            const std::string& camera_id = keyframe.camera_id;
            const cv::Mat& image = keyframe.image;
            const std::vector<Eigen::Vector2d>& observed_2D_points = keyframe.observed_2D_points;
            const std::vector<Eigen::Vector3d>& gps_3D_points = keyframe.gps_3D_points;

            int num_observed_points = observed_2D_points.size();
            int num_reprojected_points = gps_3D_points.size();

            // Reproject GPS points
            std::vector<Eigen::Vector2d> reprojected_2D_points(num_reprojected_points);
            ReprojectPoints(image, camera_intrinsics, camera_pose, gps_3D_points, reprojected_2D_points);

            // Find correspondences per observed point: which reprojected point is closest?
            std::vector<int> correspondence_indices(num_observed_points);
            FindCorrespondences(observed_2D_points, reprojected_2D_points, correspondence_indices);

            // Visualise and save
            cv::Mat visualization = image.clone();
            DrawCorrespondences(correspondence_indices, observed_2D_points, reprojected_2D_points, visualization);
            SaveVisualization(visualization, filename, camera_id, iteration, visualization_path);

            // Add residuals to problem
            for (size_t i = 0; i < num_observed_points; ++i) {

                if (correspondence_indices[i] != -1) {
                    ceres::CostFunction* cost_function = CameraProjectionCostFunctor::Create(
                        observed_2D_points[i],
                        gps_3D_points[correspondence_indices[i]],
                        camera_intrinsics);

                    problem.AddResidualBlock(cost_function, nullptr, camera_pose);
                }
            }
        }
        // Configure the solver
        ceres::Solver::Options options;
        options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
            // idea: solver type Suite Sparse
        // options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
            // sparse normal Koleski
        // options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;

        // Solve the problem
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Print the results
        // std::cout << summary.FullReport() << "\n";
        // std::cout << "Camera pose after optimization:\n";
        // for (size_t i = 0; i < 7; ++i) {
        //     std::cout << "camera_pose[" << i << "] = " << camera_pose[i] << "\n";
        // }
    }
}


/**
 * @brief Python binding: Adds a keyframe from Python.
 *
 * Converts Python types to C++ types and calls cpp_add_keyframe.
 *
 * @param filename The image filename (Python string).
 * @param camera_id The camera identifier (Python string).
 * @param image The image data (numpy array).
 * @param observed_2D_points Observed 2D image points (numpy array).
 * @param gps_3D_points Corresponding 3D GPS points (numpy array).
 */
void py_add_keyframe(
    py::str filename,
    py::str camera_id,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
    py::array_t<double, py::array::c_style | py::array::forcecast> observed_2D_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> gps_3D_points) {

    // Convert Python input types to C++ types
    std::string filename_cpp = filename;

    std::string camera_id_cpp = camera_id;

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

    // Call the C++ function
    cpp_add_keyframe(filename_cpp, camera_id_cpp, image_cpp, observed_2D_points_cpp, gps_3D_points_cpp);
}


/**
 * @brief Python binding: Clears all keyframes from Python.
 */
void py_reset_keyframes() {
    cpp_reset_keyframes();
}


/**
 * @brief Python binding: Runs camera pose optimization from Python.
 *
 * Converts Python types to C++ types, runs optimization, and returns the final camera pose as a numpy array.
 *
 * @param camera_pose Input/output: Camera pose (numpy array).
 * @param camera_intrinsics Camera intrinsics (numpy array).
 * @param iterations Number of optimization iterations.
 * @param visualization_path Directory to save visualizations.
 * @return Final camera pose as a numpy array.
 */
py::array py_update_camera_pose(
    py::array_t<double, py::array::c_style | py::array::forcecast> camera_pose,
    py::array_t<double, py::array::c_style | py::array::forcecast> camera_intrinsics,
    py::int_ iterations,
    py::str visualization_path) {

    if (camera_pose.shape()[0] != 7) {
        throw std::runtime_error("camera_pose must have 7 elements: [x, y, z, qx, qy, qz, qw]");
    }
    if (camera_intrinsics.shape()[0] != 4) {
        throw std::runtime_error("camera_intrinsics must have 4 elements: [fx, fy, cx, cy]");
    }

    double camera_intrinsics_cpp[4];
    std::memcpy(camera_intrinsics_cpp, camera_intrinsics.data(), 4 * sizeof(double));

    double camera_pose_cpp[7];
    std::memcpy(camera_pose_cpp, camera_pose.data(), 7 * sizeof(double));

    int iterations_cpp;
    iterations_cpp = iterations;

    std::string visualization_path_cpp = visualization_path;

    cpp_update_camera_pose(
        camera_pose_cpp,
        camera_intrinsics_cpp,
        iterations_cpp,
        visualization_path_cpp);

    // Convert back to numpy array
    py::array_t<double> final_camera_pose(7);
    std::memcpy(final_camera_pose.mutable_data(), camera_pose_cpp, 7 * sizeof(double)); 

    return final_camera_pose;
}


/**
 * @brief Pybind11 module definition for optimization.
 */
PYBIND11_MODULE(optimization, m) {
    m.doc() = "C++ camera pose optimization of combined keyframes using Ceres";

    m.def("add_keyframe", &py_add_keyframe, "Add a keyframe to the optimization problem");
    m.def("reset_keyframes", &cpp_reset_keyframes, "Reset the optimization problem");
    m.def("update_camera_pose", &py_update_camera_pose, "Optimize camera pose");
}