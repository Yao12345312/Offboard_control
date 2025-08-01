#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "mavros_msgs/msg/position_target.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include "rclcpp/qos.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace std::chrono_literals;

class PID {
public:
    PID(double kp, double ki, double kd)
        : kp_(kp), ki_(ki), kd_(kd), integral_(0.0) {}

    double compute(double target_pos, double current_pos, double current_vel, double dt) {
        double error = target_pos - current_pos;
        integral_ += error * dt;
        double derivative = -current_vel;
        double output = kp_ * error + ki_ * integral_ + kd_ * derivative;
        return std::clamp(output, -2.0, 2.0);
    }

    void reset() {
        integral_ = 0;
    }

private:
    double kp_, ki_, kd_;
    double integral_;
};

class OffboardControl : public rclcpp::Node {
public:
    OffboardControl()
        : Node("offboard_control"),
          step_(0),
          flag_(0),
          finish_task2_(0),
          pid_x_(0.8, 0.0, 0.2),
          pid_y_(0.8, 0.0, 0.2),
          pid_z_(1.2, 0.0, 0.35)
    {
        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "mavros/state", 10,
            [this](const mavros_msgs::msg::State::SharedPtr msg) {
                current_state_ = *msg;
            });

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "mavros/local_position/pose",
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                current_pose_ = *msg;
            });

        vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "mavros/local_position/velocity",
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
                current_vel_ = *msg;
            });

        way_point_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/grid_waypoint",
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const std_msgs::msg::String::SharedPtr msg) {
                parse_waypoints(msg->data);
            });

        k230_class_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "/k230/class_info",
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                class_id = msg->data;
            });
        
        k230_pos_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/k230/position",
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const geometry_msgs::msg::Point::SharedPtr msg) {
                offset_x = msg->x;
                offset_y = msg->y;
            });

        k230_num_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "/k230/count",
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                k230_num = msg->data;
            });

        raw_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>(
            "mavros/setpoint_raw/local", 10);

        laser_pointer_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("control_laser_pointer", 10);

        serial_screen_pub_ = this->create_publisher<std_msgs::msg::String>("/serial_screen_command", 10);

        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");

        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("mavros/cmd/arming");

        timer_ = this->create_wall_timer(
            20ms, std::bind(&OffboardControl::timer_callback, this));

        last_request_time_ = this->now();

        start_time_ = this->now();
    }

    bool is_connected() const { return current_state_.connected; }

private:
    // 由编号自动计算坐标，不再使用字典
    std::tuple<double, double, double> get_position_from_label(const std::string& label) {
        // 格式必须为A{num}B{num}
        if (label.size() < 4 || label[0] != 'A') return std::make_tuple(0.0, 0.0, 1.2);

        size_t posB = label.find('B');
        if (posB == std::string::npos) return std::make_tuple(0.0, 0.0, 1.2);

        int a_num = std::stoi(label.substr(1, posB-1));
        int b_num = std::stoi(label.substr(posB+1));
        if (a_num < 1 || a_num > 9 || b_num < 1 || b_num > 7) return std::make_tuple(0.0, 0.0, 1.2);

        double x = (b_num - 1) * 0.5;
        double y = (9 - a_num) * 0.5;
        double z = 1.2;
        return std::make_tuple(x, y, z);
    }

    // 只在未起飞前允许设置航点，收到话题并解析后step=1
    void parse_waypoints(const std::string& msg) {
        if (step_ > 0) {
            RCLCPP_WARN(this->get_logger(), "Already started mission, ignoring new waypoints.");
            return;
        }
        waypoint_sequence_.clear();
        std::stringstream ss(msg);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                waypoint_sequence_.push_back(item);
            }
        }
        if (!waypoint_sequence_.empty()) {
            current_waypoint_index_ = 0;
            waypoints_ready_ = true;
            RCLCPP_INFO(this->get_logger(), "Received waypoint sequence, total: %zu", waypoint_sequence_.size());
        }
    }

    // 只有收到航点且无人机已连接/解锁/进入offboard才起飞
    void timer_callback() {
        if (!current_state_.connected) return;
        switch (step_) {
            case 0:
                handle_init_phase();
                break;
            case 1:
                // 起飞，直接到第一个航点
                if (waypoint_sequence_.empty()) {
                    RCLCPP_WARN(this->get_logger(), "No waypoints set, aborting mission.");
                    rclcpp::shutdown();
                    return;
                }
                // 起飞点A9B1，悬停1秒后进入航点序列
                if (!hold_position_start_) {
                    hold_pisition_start_time_ = this->now();
                    hold_position_start_ = true;
                    auto pos = get_position_from_label("A9B1");
                    publish_position(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
                    RCLCPP_INFO(this->get_logger(), "Takeoff at A9B1 (%.2f, %.2f, %.2f)",
                        std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
                } else {
                    auto pos = get_position_from_label("A9B1");
                    publish_position(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
                    auto elapsed = this->now() - hold_pisition_start_time_;
                    if (elapsed.seconds() >= 3.0) {
                        step_ = 2;
                        hold_position_start_ = false;
                        RCLCPP_INFO(this->get_logger(), "Start mission, fly to waypoints...");
                    }
                }
                break;
            case 2:
                // 遍历所有航点
                if (current_waypoint_index_ < waypoint_sequence_.size()) {
                    const std::string& label = waypoint_sequence_[current_waypoint_index_];
                    auto pos = get_position_from_label(label);
                    double px = std::get<0>(pos);
                    double py = std::get<1>(pos);
                    double pz = std::get<2>(pos);
                    if (!hold_position_start_) {
                        hold_pisition_start_time_ = this->now();
                        hold_position_start_ = true;
                        publish_position(px, py, pz);
                        RCLCPP_INFO(this->get_logger(), "Flying to waypoint: %s (%.2f, %.2f, %.2f)", label.c_str(), px, py, pz);
                    } else {
                        publish_position(px, py, pz);
                        auto elapsed = this->now() - hold_pisition_start_time_;
                        if(class_id==0 || class_id==1 ||class_id==2||class_id==3||class_id==4)
                            {
                            k230_class(class_id,label);
                            RCLCPP_INFO(this->get_logger(), "find animal");
                            }
                        else{
                            RCLCPP_INFO(this->get_logger(), "no found animal");
                        }    
                        control_laser_pointer(1,2.0);
                        if (elapsed.seconds() >= 3.5) {
                            current_waypoint_index_++;
                            hold_position_start_ = false;
                            RCLCPP_INFO(this->get_logger(), "Reached waypoint: %s, moving to next...", label.c_str());
                        }
                    }
                } else {
                    step_ = 3;
                    hold_position_start_ = false;
                }
                break;
            case 3:
                // 飞到最后一个点悬停5秒
                if (!waypoint_sequence_.empty()) {
                    auto pos = get_position_from_label(waypoint_sequence_.back());
                    double px = std::get<0>(pos);
                    double py = std::get<1>(pos);
                    double pz = std::get<2>(pos);
                    if (!hold_position_start_) {
                        hold_pisition_start_time_ = this->now();
                        hold_position_start_ = true;
                        publish_position(px, py, pz);
                        RCLCPP_INFO(this->get_logger(), "Arrived at final waypoint: %s (%.2f, %.2f, %.2f)", waypoint_sequence_.back().c_str(), px, py, pz);
                    } else {
                        publish_position(px, py, pz);
                        auto elapsed = this->now() - hold_pisition_start_time_;
                        if (elapsed.seconds() >= 3.5) {
                            step_ = 4;
                            hold_position_start_ = false;
                            RCLCPP_INFO(this->get_logger(), "Ready to land...");
                        }
                    }
                } else {
                    step_ = 4;
                }
                break;
            case 4:
                // 降落到A9B1 (起飞点)
                auto pos = get_position_from_label("A9B1");
                double px = std::get<0>(pos);
                double py = std::get<1>(pos);
                if (!hold_position_start_) {
                    hold_pisition_start_time_ = this->now();
                    hold_position_start_ = true;
                    publish_position(px, py, 0.2);
                    RCLCPP_INFO(this->get_logger(), "Landing at (%.2f, %.2f, 0.2)", px, py);
                } else {
                    publish_position(px, py, 0.2);
                    auto elapsed = this->now() - hold_pisition_start_time_;
                    if (elapsed.seconds() >= 3.0) {
                        auto arm_req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
                        arm_req->value = false;
                        arming_client_->async_send_request(arm_req);
                        rclcpp::sleep_for(std::chrono::seconds(3));
                        RCLCPP_INFO(this->get_logger(), "Disarm request sent. Mission complete.");
                        rclcpp::shutdown();
                    }
                }
                break;
        }
    }

    // 只有在waypoints_ready时才允许起飞
    void handle_init_phase() {
        // 如果未收到航点则不允许起飞
        if (!waypoints_ready_) {
            auto message = mavros_msgs::msg::PositionTarget();
            message.header.stamp = this->now();
            message.header.frame_id = "map";
            message.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
            message.type_mask =
                mavros_msgs::msg::PositionTarget::IGNORE_VX |
                mavros_msgs::msg::PositionTarget::IGNORE_VY |
                mavros_msgs::msg::PositionTarget::IGNORE_VZ |
                mavros_msgs::msg::PositionTarget::IGNORE_AFX |
                mavros_msgs::msg::PositionTarget::IGNORE_AFY |
                mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
                mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE |
                mavros_msgs::msg::PositionTarget::IGNORE_YAW;
            message.position.x = 0.0;
            message.position.y = 0.0;
            message.position.z = 0.21;
            raw_pub_->publish(message);

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for waypoints before takeoff...");
            return;
        }

        // 收到航点后，进行解锁和模式切换
        if (current_state_.mode != "OFFBOARD") {
            if ((this->now() - last_request_time_).seconds() > 2.0) {
                last_request_time_ = this->now();
                RCLCPP_INFO(this->get_logger(), "Requesting OFFBOARD mode...");
                // 实飞建议遥控器手动切模式，代码可屏蔽
                // auto mode_req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
                // mode_req->custom_mode = "OFFBOARD";
                // set_mode_client_->async_send_request(mode_req);
            }
            return;
        }
        if (!current_state_.armed) {
            if ((this->now() - last_request_time_).seconds() > 1.0) {
                auto arm_req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
                arm_req->value = true;
                arming_client_->async_send_request(arm_req);
                last_request_time_ = this->now();
                RCLCPP_INFO(this->get_logger(), "Requesting arming...");
            }
            return;
        }
        // 解锁且模式切换完成、且收到航点，才允许step=1
        RCLCPP_INFO(this->get_logger(), "OFFBOARD & armed & waypoints ready, takeoff!");
        step_ = 1;
        flag_ = 0;
    }

    void publish_position(double px, double py, double pz) {
        auto message = mavros_msgs::msg::PositionTarget();
        message.header.stamp = this->now();
        message.header.frame_id = "map";
        message.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
        message.type_mask =
            mavros_msgs::msg::PositionTarget::IGNORE_VX |
            mavros_msgs::msg::PositionTarget::IGNORE_VY |
            mavros_msgs::msg::PositionTarget::IGNORE_VZ |
            mavros_msgs::msg::PositionTarget::IGNORE_AFX |
            mavros_msgs::msg::PositionTarget::IGNORE_AFY |
            mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW;
        message.position.x = px;
        message.position.y = py;
        message.position.z = pz;
        raw_pub_->publish(message);
    }

    void control_laser_pointer(int flag,int time) {
        auto message = std_msgs::msg::Float64MultiArray();

        // 激光状态 1 表示点亮，0 表示熄灭
        message.data.push_back(flag);

        // 激光点亮的时间
        message.data.push_back(time);  

        // 发布消息
        laser_pointer_pub_->publish(message);
   
        // RCLCPP_INFO(this->get_logger(), "laser pointer on");
    }

    void k230_class(int class_id,const std::string& label) {
        // 定义 class_map
        static const std::unordered_map<int, std::string> class_map = {
            {0, "elephant"},
            {1, "peacock"},
            {2, "monkey"},
            {3, "tiger"},
            {4, "wolf"}
        };
        std::string class_name = "unknown";
        // 查找 class_id 对应的类别名称
        auto it = class_map.find(class_id);
            if (it != class_map.end()) {  // 检查是否找到
            class_name = it->second;
            }
            // 创建并发布动物名称
            auto id_msg = std_msgs::msg::String();
            id_msg.data ="page1.t"+ std::to_string(send_to_screen_count) +".txt=\"" + class_name + "\"";
            serial_screen_pub_->publish(id_msg);
            send_to_screen_count++;

            auto num_msg = std_msgs::msg::String();
            num_msg.data="page1.t"+ std::to_string(send_to_screen_count) +".txt=\"" + std::to_string(k230_num) + "\"";
            serial_screen_pub_->publish(num_msg);
            send_to_screen_count++;

            auto pos_msg = std_msgs::msg::String();
            pos_msg.data="page1.t"+ std::to_string(send_to_screen_count) +".txt=\"" + label + "\"";
            serial_screen_pub_->publish(pos_msg);
            send_to_screen_count++;
    }

    bool Ex_vision_fly_to_target(double errx, double erry, double pz) {
        double tx = current_pose_.pose.position.x - errx; // 你可能需要根据实际需要来计算 tx
        double ty = current_pose_.pose.position.y - erry; 
        publish_position(tx, ty, pz);
        double dist_x = std::fabs(errx);
        double dist_y = std::fabs(erry);
        // ring_centre_x 可以用errx 最大值来存 缺点是误识别时会有较大问题
        
        if (dist_x <= 0.008 && dist_y <= 0.008 && errx != 0) 
        {
            return 1;
        } 
        else 
        { 
	        return 0;
        }

    }


    int step_;
    int flag_;
    int finish_task2_;
    float offset_x;
    float offset_y;
    int class_id;
    int send_to_screen_count=1;
    int k230_num;

    rclcpp::Time last_request_time_;
    rclcpp::Time start_time_;
    rclcpp::Time hold_pisition_start_time_;
    bool hold_position_start_ = false;
    bool waypoints_ready_ = false;
    mavros_msgs::msg::State current_state_;
    geometry_msgs::msg::PoseStamped current_pose_;
    geometry_msgs::msg::TwistStamped current_vel_;
    PID pid_x_, pid_y_, pid_z_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr way_point_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr k230_class_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr k230_pos_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr k230_num_subscription_;
    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr raw_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr laser_pointer_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr serial_screen_pub_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> waypoint_sequence_;
    size_t current_waypoint_index_ = 0;
    std::string qr_data_;
   
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OffboardControl>();

    while (rclcpp::ok() && !node->is_connected()) {
        rclcpp::spin_some(node);
        std::this_thread::sleep_for(100ms);
        RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
                             "Waiting for FCU connection...");
    }

    RCLCPP_INFO(node->get_logger(), "FCU connected!");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
