#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class CartographerLaserTransfer : public rclcpp::Node
{
public:
    CartographerLaserTransfer() : Node("cartographer_laser_transfer")
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        vision_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/mavros/vision_pose/pose", 10);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "/mavros/imu/data",
    rclcpp::SensorDataQoS(),
    std::bind(&CartographerLaserTransfer::imu_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&CartographerLaserTransfer::publish_vision_pose, this));

        RCLCPP_INFO(this->get_logger(), "cartographer_laser_transfer node started.");
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        latest_imu_time_ = msg->header.stamp;
        imu_received_ = true;
    }

    void publish_vision_pose()
    {
        if (!imu_received_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "IMU data not received yet, waiting...");
            return;
        }

        geometry_msgs::msg::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Could not get transform: %s", ex.what());
            return;
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = latest_imu_time_;  // 与IMU时间同步
        pose_msg.header.frame_id = "map";

        // 仅发布水平位置信息（x, y），z 设置为 0.0
        pose_msg.pose.position.x = transformStamped.transform.translation.x;
        pose_msg.pose.position.y = transformStamped.transform.translation.y;
        pose_msg.pose.position.z = 0.0;

        // 保留原始方向（你可以只保留 yaw，如果需要）
        // 提取原始四元数
        tf2::Quaternion q_orig;
        tf2::fromMsg(transformStamped.transform.rotation, q_orig);

        // 提取 yaw 角度
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_orig).getRPY(roll, pitch, yaw);

        // 构造仅包含 yaw 的四元数（绕 Z 轴旋转）
        tf2::Quaternion q_yaw;
        q_yaw.setRPY(0, 0, yaw);
        pose_msg.pose.orientation = tf2::toMsg(q_yaw);

        // 发布消息
        vision_pose_pub_->publish(pose_msg);
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr vision_pose_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Time latest_imu_time_;
    bool imu_received_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CartographerLaserTransfer>());
    rclcpp::shutdown();
    return 0;
}
