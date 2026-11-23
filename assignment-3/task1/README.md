# Task 1: Monocular Visual SLAM / Visual Odometry

## Overview

This task implements a monocular visual odometry pipeline in Python and evaluates it using KITTI Odometry Sequence 00.

The implementation uses OpenCV and performs feature detection, feature matching, essential matrix estimation, relative camera pose recovery, 3-D landmark triangulation, camera pose accumulation, and trajectory evaluation.

## Pipeline

The implemented pipeline is:

KITTI image sequence
→ ORB feature detection
→ BF Hamming feature matching
→ Lowe ratio test
→ Essential matrix estimation using RANSAC
→ Relative rotation and translation recovery
→ 3-D landmark triangulation
→ Camera pose accumulation
→ Trajectory comparison with KITTI ground truth

## Feature Detection and Matching

ORB features are extracted from consecutive KITTI grayscale images.

The descriptors are matched using a brute-force Hamming-distance matcher. A ratio test is used to reject ambiguous feature matches.

## Essential Matrix

The essential matrix is estimated from matched image points using OpenCV's `findEssentialMat()` with RANSAC.

The intrinsic calibration matrix for KITTI Sequence 00 is used during essential matrix estimation.

## Relative Pose

The relative camera rotation and translation direction are recovered using OpenCV's `recoverPose()`.

Because this is monocular visual odometry, the recovered translation does not contain an absolute metric scale.

## Triangulation

Matched points that survive the pose estimation are triangulated using `cv2.triangulatePoints()`.

The resulting 3-D points are checked for finite coordinates and positive depth.

## Camera Trajectory

The recovered relative camera transformations are accumulated to estimate the camera trajectory through the KITTI sequence.

The relative transform is inverted when accumulating the world-to-camera trajectory so that the estimated trajectory is expressed consistently in the world coordinate system.

## Evaluation

The estimated trajectory is compared against the KITTI ground-truth trajectory using the X-Z plane.

The raw trajectory is saved separately from the evaluation-aligned trajectory.

Because monocular visual odometry has an unknown global scale and accumulated drift, a 2-D similarity transform is applied after trajectory estimation. This alignment adjusts global rotation, translation, and scale only for evaluation and visualization.

The alignment does not use KITTI ground truth during the visual odometry estimation itself.

## Results

For the final run on KITTI Sequence 00 using the first 300 frames:

- Mean position error: approximately 5.16 m
- RMSE: approximately 6.39 m

The aligned trajectory follows the general shape and direction of the KITTI ground-truth trajectory.

The raw visual-odometry trajectory shows substantially more drift, illustrating the accumulated error and scale ambiguity of monocular visual odometry.

## Output Files

- `vslam_kitti.py` — complete Python implementation
- `requirements.txt` — Python dependencies
- `results/trajectory_kitti.png` — trajectory comparison plot
- `results/estimated_trajectory_raw_final.txt` — raw estimated trajectory
- `results/estimated_trajectory_aligned_final.txt` — similarity-aligned trajectory
- `results/ground_truth_trajectory_final.txt` — KITTI ground truth
- `results/alignment_final.txt` — similarity alignment parameters and error metrics

## Running the Implementation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
