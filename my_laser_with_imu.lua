include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,
  use_pose_extrapolator = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.5,
  submap_publish_period_sec = 0.2,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
  
}

MAP_BUILDER.use_trajectory_builder_2d = true

--TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
--TRAJECTORY_BUILDER_2D.use_imu_data = false  --是否使用imu数据
--TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.15
--TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(35.)
--POSE_GRAPH.optimization_problem.huber_scale = 1e2

-- 调整 IMU 在扫描匹配中的权重
--TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 1e2  -- 原默认1e2（增大旋转权重）
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 1e0  -- 平移权重保持或微调

TRAJECTORY_BUILDER_2D.use_imu_data = true  --是否使用imu数据
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 2 --积累几帧激光数据作为一个标准单位scan
--TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 1   --//尽量小点  // 如果移动距离过小, 或者时间过短, 不进行地图的更新
--POSE_GRAPH.optimize_every_n_nodes = 30  --后端优化节点

return options
