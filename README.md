<p align="center">
  <img src="docs/assets/farlio-banner.png" alt="FAR-LIO" width="660">
</p>

<p align="center">
  A highly efficient, CUDA-accelerated framework for Fast, Accurate, and Robust LiDAR-Inertial Odometry.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.26010"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.26010-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://github.com/TUMFTM/FAR-LIO/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://github.com/TUMFTM/FAR-LIO/actions/workflows/docker.yml"><img alt="Docker build" src="https://github.com/TUMFTM/FAR-LIO/actions/workflows/docker.yml/badge.svg"></a>
  <a href="https://github.com/TUMFTM/FAR-LIO/actions/workflows/docs.yml"><img alt="Docs build" src="https://github.com/TUMFTM/FAR-LIO/actions/workflows/docs.yml/badge.svg"></a>
  <br />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white">
  <img alt="ROS 2 Jazzy" src="https://img.shields.io/badge/ROS%202-Jazzy-22314E?logo=ros&logoColor=white">
  <img alt="C++20" src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white">
</p>

<h2 align="center"><a href="https://tumftm.github.io/FAR-LIO">📖 Documentation</a></h2>

## Installation

Pull the pre-built CUDA image from the GitHub Container Registry:

```bash
docker pull ghcr.io/tumftm/far-lio:latest
```

Or build it yourself:

```bash
git clone --recursive https://github.com/TUMFTM/FAR-LIO.git
cd FAR-LIO
docker build -f docker/Dockerfile -t ghcr.io/tumftm/far-lio:latest .
```

See the [installation guide](https://tumftm.github.io/FAR-LIO/installation.html) for prerequisites and options.

## Usage

Run FAR-LIO on its own (e.g. with live sensors — needs a GPU):

```bash
docker compose --profile far-lio up
```

Or replay a ROS 2 bag through it — the helper script mounts the bag and starts both services:

```bash
./run.sh /path/to/rosbag        # bag directory (metadata.yaml) or an .mcap file
```

`ROS_DOMAIN_ID` is taken from your environment if set, otherwise `0`. See the
[usage guide](https://tumftm.github.io/FAR-LIO/usage.html) for details.

## Racetrack Deployment (A2RL)

An excerpt of FAR-LIO running on an autonomous race car on a racetrack as part of the
[Abu Dhabi Autonomous Racing League (A2RL)](https://a2rl.io/).

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img height="240" alt="A2RL onboard footage" src="https://github.com/user-attachments/assets/76415cfc-4ba3-4187-afee-672ece0ad899" />
      <br /><sub>Onboard camera footage</sub>
    </td>
    <td align="center" width="50%">
      <img height="240" alt="A2RL FAR-LIO ego view" src="https://github.com/user-attachments/assets/6504064d-b9d0-4411-96c4-ed91ed6b8f72" />
      <br /><sub>FAR-LIO's odometry</sub>
    </td>
  </tr>
</table>

See the [documentation](https://tumftm.github.io/FAR-LIO/) for the architecture and further
application domains.

## 2D lidar (F1TENTH)

A single-layer scan (a 2D lidar on a small car) can be registered in SE(2). The mode is a parameter,
`pipeline.lidar_2d` (default `false`; with it off every 3D path runs the original code, held by the
bit-exact golden tests FARLIO-1010..1030). With it on:

- **Covariance extrusion.** A planar wall's neighbourhood has two near-zero eigenvalues, across the wall and
  along z, so the smallest eigenvector (the normal) can be z and constrain nothing in the plane. Before the
  eigen-decomposition the z row and column of the covariance are replaced by an analytic vertical extrusion
  (`planarize_covariance`), so the normal is always in-plane. CPU and CUDA share the helper.
- **SE(2) solve.** Gauss-Newton and Levenberg-Marquardt solve the reduced 3x3 system in (t_x, t_y, w_z) and
  embed it back with zeros in t_z, w_x, w_y; the initial guess is first projected to SE(2) (`project_se2`).
  A parametrization rather than a prior: z, roll and pitch cannot move (tests assert them bit-equal to the
  guess), and the float32 6x6 system is never made ill-conditioned by a large prior weight.
- **LaserScan input.** `node.input_scan:=/scan` makes the nodes subscribe `sensor_msgs/LaserScan` (same
  best-effort QoS as the point-cloud input) and convert it with `laser_geometry`; beams are placed at
  `angle_min + i*angle_increment`, and non-finite or out-of-range ranges are dropped. Empty (the default) keeps
  the PointCloud2 input.
- **`config/f1tenth.yml`** is `config/far-lio.yml` with F1TENTH values, each changed value commented with its
  source. It runs `tam_gicp_ext` with `ekf_2d_kin` (IMU only; the vehicle-model values are recorded for
  `ekf_2d_nh`).

- **Lidar yaw in the 2D EKF.** The upstream node read `mask[0..2]` (roll, pitch, yaw) of a one-entry
  `orientation_rad` mask in 2D, so the lidar yaw was never fused and the filter started at yaw 0. It now reads the
  yaw flag only (`mask[0]`), and a 2D `orientation_rad` whose `R_init` is not one entry (three in 3D) is refused at
  startup (FARLIO-1890, 1900, 1910).

### Initial pose (`node.input_initial_pose`)

By default the first scan is registered at the identity, which in localization mode is wherever the mapping run
started. For the `EXTERNALGUESS` executables (`tam_gicp_ext`, `tam_icp_ext`, `tam_cuda_*_ext`):

- `node.input_initial_pose` (default `""`: no subscription; `config/f1tenth.yml`: `/core/state/initial_pose`) is a
  `geometry_msgs/msg/PoseWithCovarianceStamped` topic, **RELIABLE, TRANSIENT_LOCAL, depth 1** (a latched
  publisher; a VOLATILE one, such as RViz's `SetInitialPose` tool on `/initialpose`, is never heard). The
  `header.frame_id` must be `node.odom_frame` (`far_lio_odom`); any other frame is refused with a WARN. The pose is
  that of `node.child_frame` (`base_link`); roll and pitch are dropped in 2D; z is kept (`project_se2`, FARLIO-1200), though the 2D EKF has no z state. The stamp and covariance are
  ignored.
- The latest valid pose received before the first registered scan is that scan's guess, ahead of the tf lookup.
  **One shot:** a pose after the first registered scan is ignored with a WARN ("relocalization not supported").
- `node.wait_initial_pose` (default `false`): scans are dropped, and nothing is registered or published, until a
  valid pose has arrived. The EKF publishes nothing either. Not with `node.wait_tf: true` or an empty
  `input_initial_pose` (the node refuses to start), and a non-empty `input_initial_pose` on `tam_icp_cv` (constant velocity, no guess) is refused too.
- **The told pose must already be accurate to about 0.1 m / 2 deg.** The input does not correct a rough pose.
  The matcher's acceptance check (`pipeline.diagnostic.ellipsis_size_s` 0.15 m x `ellipsis_size_d` 0.075 m around
  the guess, `config/f1tenth.yml:135-136`) marks a registration further than that from its guess "converged | too
  far from guess" (level 2), and the EKF then never initialises. Measured on the synthetic oval: offsets of 0, 0.05 m
  / 0.02 m / 1 deg and 0.1 m / 0.05 m / 2 deg start the EKF; 0.3 m / 0.2 m / 5 deg does not. The matcher also
  barely corrects an offset (first odometry 0.051 m from the truth at a 0.054 m offset). Correcting a rough pose is
  not done here.
- The pose is consumed, and the one shot closes, by the first scan that is actually registered: an empty scan (all
  `inf`, or nothing left after cropping) leaves it for the next one (FARLIO-1855). A pose with an all-zero or
  non-finite orientation is refused with a WARN (FARLIO-1840).
- Tests: FARLIO-1810 (keys), 1820 (hold), 1830 (pose is the first guess), 1840 (frame, orientation), 1855 (empty scan), 1850 (one shot), 1860
  (startup refusals), 1870 (through `ekf_2d_kin` and `ekf_2d_nh`), 1880 (the 1830 case on `tam_cuda_gicp_ext`, by hand
  on a GPU).

**What the tests measure (and do not claim).** A single scan cannot observe its position along a straight
between parallel walls, so FARLIO-1210 (one registration from a 0.10 m / -0.08 m / 2 deg offset) asserts only
lateral error and heading on straights and records the along-track error (0.09-0.13 m at 64 beams, i.e. the
offset left uncorrected). It asserts 1 cm / 0.2 deg where that is met (GICP at 1080 beams on straights, and one
curve position); elsewhere each case has a bound of twice its measured worst (GICP on curves 0.05-0.18 m,
ICP on curves 0.16-0.20 m). FARLIO-1610 (one synthetic lap at 2 m/s, 2 s ramp, ekf_2d_kin or ekf_2d_nh, 64
or 1080 beams) records drift of 1.6-11.5 m worst position per case (kin1080 once 9.035 m) and, per case, a worst yaw of 3.43 (kin1080),
3.52 (nh1080), 10.77 (kin64) and 14.81 deg (nh64) (10 runs per case, with the EKF fusing the lidar yaw), against bounds of twice
that (position: 2 x the worst of all runs); it does not meet 0.10 m / 2 deg in any case. The yaw is
GICP's: it is tens of degrees wrong on this lap, and before the 2D yaw fix the EKF's yaw was the integral of the
test's exact simulated gyro instead (`research/2026-11-02-far-lio-1610-yaw-measured.md` in GR-roboracer's scratch). Neither test is an accuracy claim: localization on a map built by
FAR-LIO itself is measured in the GR-roboracer stack.

**64-beam density caveat.** Fraction of frame points that get a covariance (FARLIO-1100, synthetic stadium
track, mean over 12 poses): 64 beams 0.000 at 0.25 m voxels, 0.605 at 0.5 m, 0.697 at 1.0 m; 1080 beams 0.921,
0.944, 0.950. `config/f1tenth.yml` uses 0.5 m. A 64-beam scan is sparse: point-to-point ICP (`tam_icp_ext`)
needs no covariance, and the node executable (not a yml key) chooses between the two.

**CPU latency** (FARLIO-1710, `bench_planar_example`, registration only, planar GaussNewton, one thread,
200 scans per case, a 2 cm guess error): GICP_EXT 0.102 ms mean / 0.271 ms p99 at 64 beams and 0.394 / 1.081
ms at 1080; ICP_EXT 0.054 / 0.135 ms at 64 and 0.569 / 1.126 ms at 1080. Reported, not gated; it excludes
preprocessing, map update and ROS.

Build and test (CPU, in the `deps` image; the CUDA build is compiled in CI and its tests run by hand on a GPU):

```bash
docker build -f docker/Dockerfile --target deps -t gr-far-lio-deps .
docker run --rm -v "$PWD":/dev_ws/src/far_lio -v gr-far-lio-build:/dev_ws/out -w /dev_ws gr-far-lio-deps bash -lc '
  . /opt/ros/jazzy/setup.bash &&
  colcon --log-base out/log build --build-base out/build-cpu --install-base out/install-cpu \
    --packages-up-to tam_odometry tam_state_estimation_node \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=OFF -DBUILD_TESTING=ON &&
  colcon --log-base out/log test --build-base out/build-cpu --install-base out/install-cpu \
    --packages-select tam_odometry tam_state_estimation_node &&
  colcon test-result --test-result-base out/build-cpu --verbose'
```

## References

If you use FAR-LIO in your research, please cite our paper:

```bibtex
@misc{2026far-lio,
      title={FAR-LIO: Enabling High-Speed Autonomy through Fast, Accurate, and Robust LiDAR-Inertial Odometry}, 
      author={Maximilian Leitenstern and Marcel Weinmann and Patrick Haft and Tobias Lasser and Dominik Kulmer and Markus Lienkamp},
      year={2026},
      eprint={2606.26010},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.26010},
}
```

### Core Developers

[Marcel Weinmann](mailto:marcel.weinmann@tum.de)  
[Maximilian Leitenstern](mailto:maxi.leitenstern@tum.de)  
Institute of Automotive Technology, School of Engineering and Design, Technical University of Munich, 85748 Garching, Germany

### Acknowledgements

We thank Patrick Haft and Tobias Lasser ([NVIDIA Corporation](https://www.nvidia.com/)) for their assistance during the CUDA development.
