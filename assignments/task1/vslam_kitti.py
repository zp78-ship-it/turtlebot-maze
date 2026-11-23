#!/usr/bin/env python3

"""
KITTI Sequence 00 Monocular Visual Odometry

Pipeline:
    KITTI images
        |
        v
    ORB feature detection
        |
        v
    BFMatcher + ratio test
        |
        v
    Essential matrix + RANSAC
        |
        v
    Recover relative R,t
        |
        v
    Triangulate 3-D landmarks
        |
        v
    Accumulate camera poses
        |
        v
    Compare against KITTI ground truth
        |
        v
    2-D similarity alignment for evaluation
        |
        v
    Trajectory plots + error metrics

Important:
Monocular visual odometry has an unknown global scale and
accumulated drift. The similarity alignment below is used
ONLY for evaluation/visualization against KITTI ground truth.
The raw estimated trajectory is also saved separately.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.expanduser("~/vslam-task1")

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "sequences",
    "00",
    "image_0"
)

POSE_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "poses",
    "00.txt"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

NUM_FRAMES = 300

# KITTI Sequence 00 camera calibration.
# P0 = left grayscale camera.
FX = 718.856
FY = 718.856
CX = 607.1928
CY = 185.2157

K = np.array(
    [
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# ORB parameters
ORB_FEATURES = 3000

# Lowe ratio threshold
RATIO_TEST = 0.75

# Minimum number of matches required
MIN_MATCHES = 20

# RANSAC threshold in pixels
ESSENTIAL_RANSAC_THRESHOLD = 1.0

# RANSAC confidence
ESSENTIAL_CONFIDENCE = 0.999

# Reject extremely large frame-to-frame rotation.
# This helps prevent one catastrophic frame from destroying
# the accumulated trajectory.
MAX_ROTATION_DEG = 20.0

# Reject extremely large normalized translation jumps.
MAX_TRANSLATION_NORM = 2.0


# ============================================================
# Utility functions
# ============================================================

def ensure_results_directory():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_images(image_dir, num_frames):
    """
    Load KITTI grayscale images.
    """

    files = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No PNG images found in:\n{image_dir}"
        )

    files = files[:num_frames]

    print(f"Found {len(files)} KITTI images.")

    images = []

    for i, filename in enumerate(files):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(
                f"Could not read image:\n{filename}"
            )

        images.append(img)

        if (i + 1) % 50 == 0:
            print(f"Loaded {i + 1}/{len(files)} images")

    return images


def load_ground_truth(pose_file, num_frames):
    """
    Load KITTI ground-truth poses.

    Each KITTI pose is a 3x4 matrix:

        [ R | t ]

    where R is the camera rotation and t is the
    camera position expressed in the world coordinate system.
    """

    poses = np.loadtxt(pose_file)

    poses = poses.reshape(-1, 3, 4)

    poses = poses[:num_frames]

    print(f"Loaded {len(poses)} ground-truth poses.")

    return poses


def pose_matrix_from_kitti(P):
    """
    Convert a KITTI 3x4 camera pose into a 4x4 homogeneous matrix.
    """

    T = np.eye(4, dtype=np.float64)

    T[:3, :4] = P

    return T


def rotation_angle_degrees(R):
    """
    Calculate the rotation angle represented by a rotation matrix.
    """

    trace = np.trace(R)

    value = (trace - 1.0) / 2.0

    value = np.clip(value, -1.0, 1.0)

    angle = np.arccos(value)

    return np.degrees(angle)


def match_features(des1, des2):
    """
    Match ORB descriptors using BFMatcher and Lowe ratio test.
    """

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    return good_matches


def triangulate_points(
    pts1,
    pts2,
    R,
    t
):
    """
    Triangulate 3-D landmarks from two camera views.

    Camera 1:
        P1 = K [I | 0]

    Camera 2:
        P2 = K [R | t]

    Returns Nx3 3-D points in camera-1 coordinates.
    """

    P1 = K @ np.hstack(
        [
            np.eye(3),
            np.zeros((3, 1))
        ]
    )

    P2 = K @ np.hstack(
        [
            R,
            t.reshape(3, 1)
        ]
    )

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        pts1.T,
        pts2.T
    )

    points_3d = (
        points_4d[:3]
        / points_4d[3]
    ).T

    return points_3d


def count_valid_triangulated_points(points_3d, R):
    """
    Count points that have finite coordinates and positive depth
    in both cameras.
    """

    if len(points_3d) == 0:
        return 0

    valid = np.isfinite(points_3d).all(axis=1)

    # Depth in camera 1
    valid &= points_3d[:, 2] > 0

    # Transform points into camera 2
    points_cam2 = (
        (R @ points_3d.T)
        + np.zeros((3, 1))
    ).T

    valid &= points_cam2[:, 2] > 0

    return int(np.sum(valid))


def estimate_motion(img1, img2, orb):
    """
    Estimate relative motion between two consecutive images.

    Returns:

        R       : relative rotation
        t       : relative translation direction
        inliers : number of essential-matrix inliers
        points3d: triangulated 3-D landmarks
        good_matches

    Important:
    recoverPose gives translation direction, not metric scale.
    """

    kp1, des1 = orb.detectAndCompute(img1, None)

    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    matches = match_features(des1, des2)

    if len(matches) < MIN_MATCHES:
        return None

    pts1 = np.float64(
        [kp1[m.queryIdx].pt for m in matches]
    )

    pts2 = np.float64(
        [kp2[m.trainIdx].pt for m in matches]
    )

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=ESSENTIAL_CONFIDENCE,
        threshold=ESSENTIAL_RANSAC_THRESHOLD,
    )

    if E is None or mask is None:
        return None

    # OpenCV can theoretically return multiple 3x3 solutions.
    if E.shape[0] > 3:
        E = E[:3, :3]

    mask = mask.ravel().astype(bool)

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    if len(pts1_in) < MIN_MATCHES:
        return None

    retval, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1_in,
        pts2_in,
        K
    )

    pose_mask = pose_mask.ravel().astype(bool)

    pts1_pose = pts1_in[pose_mask]
    pts2_pose = pts2_in[pose_mask]

    if len(pts1_pose) < MIN_MATCHES:
        return None

    rotation_deg = rotation_angle_degrees(R)

    if rotation_deg > MAX_ROTATION_DEG:
        print(
            f"  Rejected: rotation={rotation_deg:.2f} deg"
        )
        return None

    translation_norm = np.linalg.norm(t)

    if translation_norm < 1e-8:
        return None

    translation_norm = float(translation_norm)

    if translation_norm > MAX_TRANSLATION_NORM:
        print(
            f"  Rejected: translation norm={translation_norm:.2f}"
        )
        return None

    # Normalize because monocular translation has unknown scale.
    t = t.reshape(3)

    t = t / np.linalg.norm(t)

    # Triangulate landmarks for the assignment requirement.
    points_3d = triangulate_points(
        pts1_pose,
        pts2_pose,
        R,
        t
    )

    valid_3d = count_valid_triangulated_points(
        points_3d,
        R
    )

    return {
        "R": R,
        "t": t,
        "inliers": int(retval),
        "points3d": points_3d,
        "valid_3d": valid_3d,
        "matches": len(matches),
    }


def accumulate_pose(T_world_camera, R, t):
    """
    Accumulate camera pose.

    recoverPose provides:

        x_current = R * x_previous + t

    Therefore:

        T_current_previous =
            [ R  t ]
            [ 0  1 ]

    is the transform from previous camera coordinates
    into current camera coordinates.

    We want:

        T_world_current

    so:

        T_world_current =
            T_world_previous *
            inverse(T_current_previous)

    """

    T_current_previous = np.eye(4)

    T_current_previous[:3, :3] = R

    T_current_previous[:3, 3] = t

    T_previous_current = np.linalg.inv(
        T_current_previous
    )

    T_world_camera_new = (
        T_world_camera
        @ T_previous_current
    )

    return T_world_camera_new


def extract_xz(poses):
    """
    Extract KITTI camera trajectory as X,Z.
    """

    result = np.zeros(
        (len(poses), 2),
        dtype=np.float64
    )

    for i, T in enumerate(poses):

        result[i, 0] = T[0, 3]
        result[i, 1] = T[2, 3]

    return result


def normalize_origin(points):
    """
    Translate a trajectory so that its first point is (0,0).
    """

    return points - points[0]


# ============================================================
# Similarity alignment
# ============================================================

def similarity_alignment_2d(
    estimated,
    ground_truth
):
    """
    Find:

        aligned = scale * R * estimated + translation

    using the Umeyama method in 2-D.

    This corrects global rotation, translation and scale
    for evaluation.

    It does NOT change the actual visual odometry result.
    """

    src = np.asarray(
        estimated,
        dtype=np.float64
    )

    dst = np.asarray(
        ground_truth,
        dtype=np.float64
    )

    src_mean = np.mean(src, axis=0)

    dst_mean = np.mean(dst, axis=0)

    src_centered = src - src_mean

    dst_centered = dst - dst_mean

    covariance = (
        dst_centered.T
        @ src_centered
        / len(src)
    )

    U, D, Vt = np.linalg.svd(
        covariance
    )

    S = np.eye(2)

    if np.linalg.det(
        U @ Vt
    ) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    variance_src = np.sum(
        src_centered ** 2
    ) / len(src)

    scale = (
        np.trace(
            np.diag(D) @ S
        )
        / variance_src
    )

    translation = (
        dst_mean
        - scale * (R @ src_mean)
    )

    aligned = (
        scale
        * (R @ src.T)
    ).T + translation

    return (
        aligned,
        scale,
        R,
        translation
    )


def calculate_errors(
    estimated,
    ground_truth
):
    """
    Calculate Euclidean position errors.
    """

    errors = np.linalg.norm(
        estimated - ground_truth,
        axis=1
    )

    mean_error = np.mean(errors)

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    max_error = np.max(errors)

    return (
        errors,
        mean_error,
        rmse,
        max_error
    )


# ============================================================
# Save trajectory
# ============================================================

def save_trajectory(filename, trajectory):
    """
    Save trajectory as:

        x z
    """

    np.savetxt(
        filename,
        trajectory,
        fmt="%.8f"
    )


# ============================================================
# Plot
# ============================================================

def plot_trajectory(
    estimated_raw,
    estimated_aligned,
    ground_truth,
    filename,
    mean_error,
    rmse
):
    """
    Create final trajectory comparison.
    """

    plt.figure(
        figsize=(12, 9)
    )

    plt.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        label="KITTI Ground Truth",
        linewidth=2.0
    )

    plt.plot(
        estimated_aligned[:, 0],
        estimated_aligned[:, 1],
        label="Visual Odometry (aligned)",
        linewidth=2.0
    )

    plt.plot(
        estimated_raw[:, 0],
        estimated_raw[:, 1],
        linestyle="--",
        alpha=0.5,
        label="Visual Odometry (raw)"
    )

    plt.scatter(
        [ground_truth[0, 0]],
        [ground_truth[0, 1]],
        s=50,
        label="Start"
    )

    plt.xlabel(
        "X (meters)"
    )

    plt.ylabel(
        "Z (meters)"
    )

    plt.title(
        "KITTI Sequence 00: Monocular Visual Odometry vs Ground Truth"
    )

    plt.legend()

    plt.grid(True)

    plt.axis("equal")

    plt.text(
        0.02,
        0.02,
        f"Mean error: {mean_error:.2f} m\n"
        f"RMSE: {rmse:.2f} m",
        transform=plt.gca().transAxes,
        bbox=dict(
            facecolor="white",
            alpha=0.8
        )
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("KITTI Sequence 00 Monocular Visual Odometry")
    print("=" * 70)

    print()
    print(f"Image directory:")
    print(f"  {IMAGE_DIR}")

    print()
    print(f"Ground truth:")
    print(f"  {POSE_FILE}")

    ensure_results_directory()

    # --------------------------------------------------------
    # Verify input files
    # --------------------------------------------------------

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image directory does not exist:\n{IMAGE_DIR}"
        )

    if not os.path.isfile(POSE_FILE):
        raise FileNotFoundError(
            f"Ground-truth file does not exist:\n{POSE_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    images = load_images(
        IMAGE_DIR,
        NUM_FRAMES
    )

    gt_poses = load_ground_truth(
        POSE_FILE,
        NUM_FRAMES
    )

    n_frames = min(
        len(images),
        len(gt_poses)
    )

    images = images[:n_frames]

    gt_poses = gt_poses[:n_frames]

    # --------------------------------------------------------
    # Initialize ORB
    # --------------------------------------------------------

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )

    print()
    print(
        f"ORB features: {ORB_FEATURES}"
    )

    print(
        f"Processing {n_frames} frames..."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Initial camera pose
    # --------------------------------------------------------

    T_world_camera = np.eye(4)

    estimated_poses = [
        T_world_camera.copy()
    ]

    successful_motions = 0

    total_matches = 0

    total_inliers = 0

    total_valid_3d = 0

    # --------------------------------------------------------
    # Frame-to-frame visual odometry
    # --------------------------------------------------------

    for i in range(
        n_frames - 1
    ):

        img1 = images[i]

        img2 = images[i + 1]

        result = estimate_motion(
            img1,
            img2,
            orb
        )

        if result is None:

            print(
                f"Frame {i + 1:03d} -> "
                f"{i + 2:03d}: motion estimation FAILED"
            )

            # Hold previous pose if estimation fails.
            estimated_poses.append(
                T_world_camera.copy()
            )

            continue

        R = result["R"]

        t = result["t"]

        inliers = result["inliers"]

        valid_3d = result["valid_3d"]

        matches = result["matches"]

        total_matches += matches

        total_inliers += inliers

        total_valid_3d += valid_3d

        T_world_camera = accumulate_pose(
            T_world_camera,
            R,
            t
        )

        estimated_poses.append(
            T_world_camera.copy()
        )

        successful_motions += 1

        if (
            (i + 1) % 10 == 0
            or i < 5
        ):

            x = T_world_camera[0, 3]

            y = T_world_camera[1, 3]

            z = T_world_camera[2, 3]

            print(
                f"Frame {i + 1:03d} -> "
                f"{i + 2:03d} | "
                f"matches={matches:4d} | "
                f"inliers={inliers:4d} | "
                f"3D={valid_3d:4d} | "
                f"position=("
                f"{x:8.2f}, "
                f"{y:8.2f}, "
                f"{z:8.2f})"
            )

    estimated_poses = np.array(
        estimated_poses
    )

    # --------------------------------------------------------
    # Extract trajectories
    # --------------------------------------------------------

    estimated_xz = extract_xz(
        estimated_poses
    )

    ground_truth_xz = extract_xz(
        gt_poses
    )

    # Make both trajectories start at origin.
    estimated_raw = normalize_origin(
        estimated_xz
    )

    ground_truth = normalize_origin(
        ground_truth_xz
    )

    # --------------------------------------------------------
    # Raw trajectory error
    # --------------------------------------------------------

    (
        raw_errors,
        raw_mean_error,
        raw_rmse,
        raw_max_error
    ) = calculate_errors(
        estimated_raw,
        ground_truth
    )

    # --------------------------------------------------------
    # Similarity alignment
    # --------------------------------------------------------

    (
        estimated_aligned,
        scale,
        R_align,
        translation
    ) = similarity_alignment_2d(
        estimated_raw,
        ground_truth
    )

    (
        aligned_errors,
        aligned_mean_error,
        aligned_rmse,
        aligned_max_error
    ) = calculate_errors(
        estimated_aligned,
        ground_truth
    )

    # --------------------------------------------------------
    # Save trajectories
    # --------------------------------------------------------

    save_trajectory(
        os.path.join(
            RESULTS_DIR,
            "estimated_trajectory.txt"
        ),
        estimated_raw
    )

    save_trajectory(
        os.path.join(
            RESULTS_DIR,
            "ground_truth_trajectory.txt"
        ),
        ground_truth
    )

    save_trajectory(
        os.path.join(
            RESULTS_DIR,
            "estimated_trajectory_aligned.txt"
        ),
        estimated_aligned
    )

    # --------------------------------------------------------
    # Save alignment information
    # --------------------------------------------------------

    alignment_file = os.path.join(
        RESULTS_DIR,
        "alignment_final.txt"
    )

    with open(
        alignment_file,
        "w"
    ) as f:

        f.write(
            "2-D Similarity Alignment\n"
        )

        f.write(
            "========================\n\n"
        )

        f.write(
            f"Scale: {scale:.12f}\n\n"
        )

        f.write(
            "Rotation:\n"
        )

        f.write(
            np.array2string(
                R_align,
                precision=12
            )
        )

        f.write(
            "\n\nTranslation:\n"
        )

        f.write(
            np.array2string(
                translation,
                precision=12
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            f"Raw mean error: "
            f"{raw_mean_error:.6f} m\n"
        )

        f.write(
            f"Raw RMSE: "
            f"{raw_rmse:.6f} m\n"
        )

        f.write(
            f"Raw maximum error: "
            f"{raw_max_error:.6f} m\n\n"
        )

        f.write(
            f"Aligned mean error: "
            f"{aligned_mean_error:.6f} m\n"
        )

        f.write(
            f"Aligned RMSE: "
            f"{aligned_rmse:.6f} m\n"
        )

        f.write(
            f"Aligned maximum error: "
            f"{aligned_max_error:.6f} m\n"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_file = os.path.join(
        RESULTS_DIR,
        "trajectory_kitti.png"
    )

    plot_trajectory(
        estimated_raw,
        estimated_aligned,
        ground_truth,
        plot_file,
        aligned_mean_error,
        aligned_rmse
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()
    print(
        f"Frames processed: "
        f"{n_frames}"
    )

    print(
        f"Successful motion estimates: "
        f"{successful_motions}"
    )

    if total_matches > 0:

        average_matches = (
            total_matches
            / max(successful_motions, 1)
        )

        average_inliers = (
            total_inliers
            / max(successful_motions, 1)
        )

        average_3d = (
            total_valid_3d
            / max(successful_motions, 1)
        )

        print(
            f"Average matches/frame: "
            f"{average_matches:.1f}"
        )

        print(
            f"Average essential inliers/frame: "
            f"{average_inliers:.1f}"
        )

        print(
            f"Average valid 3-D points/frame: "
            f"{average_3d:.1f}"
        )

    print()
    print("RAW VISUAL ODOMETRY")
    print(
        f"Mean position error: "
        f"{raw_mean_error:.4f} m"
    )

    print(
        f"RMSE: "
        f"{raw_rmse:.4f} m"
    )

    print(
        f"Maximum position error: "
        f"{raw_max_error:.4f} m"
    )

    print()
    print("2-D SIMILARITY-ALIGNED RESULT")
    print(
        f"Global scale factor: "
        f"{scale:.6f}"
    )

    print(
        f"Mean position error: "
        f"{aligned_mean_error:.4f} m"
    )

    print(
        f"RMSE: "
        f"{aligned_rmse:.4f} m"
    )

    print(
        f"Maximum position error: "
        f"{aligned_max_error:.4f} m"
    )

    print()
    print("Saved files:")
    print(
        f"  {RESULTS_DIR}/estimated_trajectory.txt"
    )

    print(
        f"  {RESULTS_DIR}/ground_truth_trajectory.txt"
    )

    print(
        f"  {RESULTS_DIR}/estimated_trajectory_aligned.txt"
    )

    print(
        f"  {RESULTS_DIR}/alignment_final.txt"
    )

    print(
        f"  {RESULTS_DIR}/trajectory_kitti.png"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
