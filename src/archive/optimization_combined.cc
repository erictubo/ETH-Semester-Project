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

    CameraProjectionCostFunctor(
        const Eigen::Vector2d observed_2D_point,
        const Eigen::Vector3d gps_3D_point,
        const double* camera_intrinsics)
        :
        observed_2D_point(observed_2D_point),
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

    static ceres::CostFunction* Create(
        const Eigen::Vector2d observed_2D_point,
        const Eigen::Vector3d gps_3D_point,
        const double* camera_intrinsics) {
        CameraProjectionCostFunctor* functor = new CameraProjectionCostFunctor(observed_2D_point, gps_3D_point, camera_intrinsics);
        return new ceres::AutoDiffCostFunction<CameraProjectionCostFunctor, 2, 7>(functor);
    }

    const Eigen::Vector2d observed_2D_point;
    const Eigen::Vector3d gps_3D_point;
    const double* camera_intrinsics;
};


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


void DrawCorrespondences(
    const std::vector<int>& correspondence_indices,
    const std::vector<Eigen::Vector2d>& observed_2D_points,
    const std::vector<Eigen::Vector2d>& reprojected_2D_points,
    cv::Mat& visualisation) {

    assert(correspondence_indices.size() == observed_2D_points.size());

    // Draw correspondences as lines
    for (int i = 0; i < correspondence_indices.size(); ++i) {
        if (correspondence_indices[i] != -1) {
            cv::line(visualisation,
                    cv::Point2d(observed_2D_points[i][0], observed_2D_points[i][1]),
                    cv::Point2d(reprojected_2D_points[correspondence_indices[i]][0], reprojected_2D_points[correspondence_indices[i]][1]),
                    cv::Scalar(0, 255, 0));
        }
    }

    // Draw observed points
    for (int i = 0; i < observed_2D_points.size(); ++i) {
        cv::circle(visualisation, cv::Point2d(observed_2D_points[i][0], observed_2D_points[i][1]), 3, cv::Scalar(255, 0, 0), -1);
    }

    // Draw reprojected points (if inside image)
    for (int i = 0; i < reprojected_2D_points.size(); ++i) {
        if (std::isnan(reprojected_2D_points[i][0]) || std::isnan(reprojected_2D_points[i][1])) {
            continue;
        } else {
            cv::circle(visualisation, cv::Point2d(reprojected_2D_points[i][0], reprojected_2D_points[i][1]), 3, cv::Scalar(0, 0, 255), -1);
        }
    }
}


void SaveVisualisation(
    const cv::Mat& visualisation,
    const std::string& filename,
    const int& iteration) {
    
    std::string path = "/Users/eric/Developer/Cam2GPS/visualisation/optimization/" + filename + "_it_" + std::to_string(iteration) + ".png";
    cv::imwrite(path, visualisation);
}





// Above: same as before
// Below: new for combined optimization of multiple keyframes




void cpp_optimize_camera_pose(
    double camera_pose[7],
    const double camera_intrinsics[4],
    const std::vector<std::string> filenames,
    const std::vector<cv::Mat> images,
    const std::vector<std::vector<Eigen::Vector2d>> global_observed_2D_points,
    const std::vector<std::vector<Eigen::Vector3d>> global_gps_3D_points) {

    int num_keyframes = filenames.size();


    for (int iteration = 0; iteration < 10; ++iteration) {

        ceres::Problem problem;

        // Combine all keyframes
        for (int keyframe = 0; keyframe < num_keyframes; ++keyframe) {

            // Get references to data for current keyframe
            const std::string& filename = filenames[keyframe];
            const cv::Mat& image = images[keyframe];
            const std::vector<Eigen::Vector2d>& observed_2D_points = global_observed_2D_points[keyframe];
            const std::vector<Eigen::Vector3d>& gps_3D_points = global_gps_3D_points[keyframe];

            int num_observed_points = observed_2D_points.size();
            int num_reprojected_points = gps_3D_points.size();

            // Reproject GPS points
            std::vector<Eigen::Vector2d> reprojected_2D_points(num_reprojected_points);
            ReprojectPoints(image, camera_intrinsics, camera_pose, gps_3D_points, reprojected_2D_points);

            // Find correspondences per observed point: which reprojected point is closest?
            std::vector<int> correspondence_indices(num_observed_points);
            FindCorrespondences(observed_2D_points, reprojected_2D_points, correspondence_indices);

            // Visualise and save
            cv::Mat visualisation = image.clone();
            DrawCorrespondences(correspondence_indices, observed_2D_points, reprojected_2D_points, visualisation);
            SaveVisualisation(visualisation, filename, iteration);

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
        // options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
            // sparse normal Koleski
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


py::array py_optimize_camera_pose(
    py::array_t<double, py::array::c_style | py::array::forcecast> camera_pose,
    py::array_t<double, py::array::c_style | py::array::forcecast> camera_intrinsics,

    const py::list filenames,
    const py::list images,
    const py::list global_observed_2D_points,
    const py::list global_gps_3D_points) {

    // py::list<py::str> filenames,
    // py::list<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>> images,
    // py::list<py::array_t<double, py::array::c_style | py::array::forcecast>> global_observed_2D_points,
    // py::list<py::array_t<double, py::array::c_style | py::array::forcecast>> global_gps_3D_points) {


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



    // // Iterate over all keyframes
    // l.attr("pop")();
    // std::cout << "List has length " << l.size() << std::endl;
    // for (py::handle obj : l) {  // iterators!
    //     std::cout << "  - " << obj.attr("__str__")().cast<std::string>() << std::endl;
    // }


    std::vector<std::string> filenames_cpp = filenames.cast<std::vector<std::string>>();
    // std::vector<cv::Mat> images_cpp = images.cast<std::vector<cv::Mat>>();
    // std::vector<std::vector<Eigen::Vector2d>> global_observed_2D_points_cpp = global_observed_2D_points.cast<std::vector<std::vector<Eigen::Vector2d>>>();
    // std::vector<std::vector<Eigen::Vector3d>> global_gps_3D_points_cpp = global_gps_3D_points.cast<std::vector<std::vector<Eigen::Vector3d>>>();

    // int num_keyframes = filenames.size();
    // for (int keyframe = 0; keyframe < num_keyframes; ++keyframe) {
    // }

    // cpp_optimize_camera_pose(
    //     camera_pose_cpp,
    //     camera_intrinsics_cpp,
    //     filenames_cpp,
    //     images_cpp,
    //     global_observed_2D_points_cpp,
    //     global_gps_3D_points_cpp);

    // Convert back to numpy array
    py::array_t<double> final_camera_pose(7);
    std::memcpy(final_camera_pose.mutable_data(), camera_pose_cpp, 7 * sizeof(double)); 

    return final_camera_pose;
}





struct Keyframe {
    Keyframe(
        const std::string filename,
        const cv::Mat& image,
        const std::vector<Eigen::Vector2d>& observed_2D_points,
        const std::vector<Eigen::Vector3d>& gps_3D_points)
        :
        filename(filename),
        image(image),
        observed_2D_points(observed_2D_points),
        gps_3D_points(gps_3D_points) {}

    std::string filename;
    cv::Mat image;
    std::vector<Eigen::Vector2d> observed_2D_points;
    std::vector<Eigen::Vector3d> gps_3D_points;
};


struct Optimization {

    // Initialization
    Optimization(
        std::vector<double> camera_pose,
        std::vector<double> camera_intrinsics)
        :
        camera_pose(camera_pose),
        camera_intrinsics(camera_intrinsics) {}


    // Method: add keyframe to struct
    void add_keyframe(struct Keyframe keyframe) {
        keyframes.push_back(keyframe);
    }

    // Struct members
    std::vector<double> camera_pose;
    const std::vector<double> camera_intrinsics;
    std::vector<Keyframe> keyframes;
};







PYBIND11_MODULE(optimization, m) {
    m.doc() = "Optimize camera pose using Ceres";
    m.def("optimize_camera_pose", &py_optimize_camera_pose, "Optimize camera pose");


}