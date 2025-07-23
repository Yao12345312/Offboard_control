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
#include <sstream>
#include "rclcpp/qos.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std::chrono_literals;

class PID {
public:
    PID(double kp, double ki, double kd)
        : kp_(kp), ki_(ki), kd_(kd), integral_(0.0) {}

    double compute(double target_pos, double current_pos, double current_vel, double dt) {
        double error = target_pos - current_pos;
        integral_ += error * dt;            //飞机位置误差做积分项
        double derivative = -current_vel;  // 使用飞控反馈速度作为微分项
        double output = kp_ * error + ki_ * integral_ + kd_ * derivative;
        //限幅
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
          step_(0), flag_(0),
          pid_x_(0.8, 0.0, 0.2), //分别设置三轴PID
          pid_y_(0.8, 0.0, 0.2),
          pid_z_(1.2, 0.0, 0.35)
    {
        //状态接收器初始化
        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "mavros/state", 10,
            [this](const mavros_msgs::msg::State::SharedPtr msg) {
                current_state_ = *msg;
            });
        //获取飞机位姿数据
       // MAVROS 的 /pose 话题订阅使用 BestEffort
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

        qr_data_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "qr_data",  
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const std_msgs::msg::String::SharedPtr msg) {
                // 获取cv识别的二维码数据
                qr_data_ = msg->data;
            }); 

        ring_offset_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "ring/offset",  
            rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)).best_effort(),
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                // 获取cv识别的圆环与中心的偏差
                offset_msg_ = *msg;
            });   

        raw_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>(
            "mavros/setpoint_raw/local", 10);
        //舵机控制初始化
        servo1_pub_ = this->create_publisher<std_msgs::msg::Int32>("servo1_cmd", 10);

        servo2_pub_ = this->create_publisher<std_msgs::msg::Int32>("servo2_cmd", 10);
 
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");

        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("mavros/cmd/arming");

        timer_ = this->create_wall_timer(
            20ms, std::bind(&OffboardControl::timer_callback, this));

        last_request_time_ = this->now();

        start_time_ = this->now();

        
    }

    bool is_connected() const { return current_state_.connected; }


private:
    void timer_callback() {
    	auto t1 = this->now();
        if (!current_state_.connected) return;

        // double dt = 0.02;
        switch (step_) {
            case 0:
                handle_init_phase();
                break;
                
             case 1: // 穿圆环终点
            
                if (!hold_position_start_) {
                    hold_pisition_start_time_= this->now();
                    hold_position_start_ = true;
                    publish_position(0.0, 0.0, 1.4);
                    RCLCPP_INFO(this->get_logger(), "reach step 1");
                } 
                else 
                {
                    publish_position(0.0, 0.0, 1.4); 
                    auto elapsed = this->now() - hold_pisition_start_time_;
                    if (elapsed.seconds() >= 10.0) {
                    RCLCPP_INFO(this->get_logger(), "go to step 2");
                    step_ = 2;
                    hold_position_start_ = false;  // 清除状态
                    }
                }
                
            break;    

            case 2: // 穿圆环起点
            
                if (!hold_position_start_) {
                    hold_pisition_start_time_= this->now();
                    hold_position_start_ = true;
                    publish_position(0.00, 0.00, 1.4);
                    RCLCPP_INFO(this->get_logger(), "reach step 2");
                } 
                else 
                {
                    double errx = offset_msg_.data[1];
                    // 还未确定相机x和实际x的关系
                    if (Ex_vision_fly_to_target(errx, 0.0, 1.4)) {
                        RCLCPP_INFO(this->get_logger(), "go to step 3");
                        step_ = 3;
                        ring_center_x = current_pose_.pose.position.x;
                        hold_position_start_ = false;  // 清除状态
                    }
                   
                }
                
            break;

            case 3: // 穿圆环终点
            
                if (!hold_position_start_) {
                    hold_pisition_start_time_= this->now();
                    hold_position_start_ = true;
                    publish_position(ring_center_x, -1.50, 1.4);
                    RCLCPP_INFO(this->get_logger(), "reach step 3");
                } 
                else 
                {
                    publish_position(ring_center_x, -1.50, 1.4); 
                    auto elapsed = this->now() - hold_pisition_start_time_;
                    if (elapsed.seconds() >= 5.0) {
                    RCLCPP_INFO(this->get_logger(), "go to step 4");
                    step_ = 4;
                    hold_position_start_ = false;  // 清除状态
                    }
                }
                
            break;

            case 4: // 穿圆环终点
            
                if (!hold_position_start_) {
                    hold_pisition_start_time_= this->now();
                    hold_position_start_ = true;
                    publish_position(ring_center_x, -3.00, 1.4);
                    RCLCPP_INFO(this->get_logger(), "reach step 4");
                } 
                else 
                {
                    publish_position(ring_center_x, -3.00, 1.4); 
                    auto elapsed = this->now() - hold_pisition_start_time_;
                    if (elapsed.seconds() >= 5.0) {
                    RCLCPP_INFO(this->get_logger(), "go to step 5");
                    step_ = 5;
                    hold_position_start_ = false;  // 清除状态
                    }
                }
                
            break;

            case 5: // 穿圆环终点
            
                if (!hold_position_start_) {
                hold_pisition_start_time_ = this->now();
                hold_position_start_ = true;
                publish_position(ring_center_x, -3.00, 0.17);
                    RCLCPP_INFO(this->get_logger(), "start step 19");
                }
                else { 

                publish_position(ring_center_x, -3.00, 0.17);
                auto elapsed = this->now() - hold_pisition_start_time_;

                if (elapsed.seconds() >= 8.0) {
                    RCLCPP_INFO(this->get_logger(), "landed.");

                    auto arm_req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();

                    arm_req->value = false;

                    arming_client_->async_send_request(arm_req);
                    // 等待 3 秒，确保 PX4 收到命令
                    rclcpp::sleep_for(std::chrono::seconds(3));

                    RCLCPP_INFO(this->get_logger(), "Disarm request sent. Shutting down...");

                    rclcpp::shutdown();   // 关闭节点

                    }
                }
            
            break;

        }        
            auto t2 = this->now();
            // RCLCPP_INFO(this->get_logger(), "%.2f ms", (t2 - t1).nanoseconds() / 1e6);
    }

    void handle_init_phase() {
        auto message = mavros_msgs::msg::PositionTarget();
        message.header.stamp = this->now();
        message.header.frame_id = "map";
        message.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
        
        // 设置掩码只使用位置控制
        message.type_mask = 
            mavros_msgs::msg::PositionTarget::IGNORE_VX |
            mavros_msgs::msg::PositionTarget::IGNORE_VY |
            mavros_msgs::msg::PositionTarget::IGNORE_VZ |
            mavros_msgs::msg::PositionTarget::IGNORE_AFX |
            mavros_msgs::msg::PositionTarget::IGNORE_AFY |
            mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE|
            mavros_msgs::msg::PositionTarget::IGNORE_YAW;

        // 设置初始位置
        message.position.x = 0.0;
        message.position.y = 0.0;
        message.position.z = 0.12;  // 设置一个小的高度作为初始目标
        

        raw_pub->publish(message);

        // 模式未切换则切换
        if (current_state_.mode != "OFFBOARD") {
            if ((this->now() - last_request_time_).seconds() > 2.0) {
                //实际飞行需要注释掉！如果通过程序切offboard，飞机失控时遥控器将无法接管，注释掉程序会一直等待遥控器切入offboard

             //auto mode_req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
             //mode_req->custom_mode = "OFFBOARD";
             //set_mode_client_->async_send_request(mode_req);

                last_request_time_ = this->now();
                RCLCPP_INFO(this->get_logger(), "Requesting OFFBOARD mode...");
            }
            return;
        }

        // 解锁
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

        RCLCPP_INFO(this->get_logger(), "OFFBOARD & armed, proceed to step 1");
        step_ = 1;
        flag_ = 0;
    }

    


    int fly_to_target(double tx, double ty, double tz, double dt) {
    	
        double ex = tx - current_pose_.pose.position.x;
        double ey = ty - current_pose_.pose.position.y;
        double ez = tz - current_pose_.pose.position.z;

        double vx = pid_x_.compute(tx, current_pose_.pose.position.x, current_vel_.twist.linear.x, dt);
        double vy = pid_y_.compute(ty, current_pose_.pose.position.y, current_vel_.twist.linear.y, dt);
        double vz = pid_z_.compute(tz, current_pose_.pose.position.z, current_vel_.twist.linear.z, dt);

        publish_velocity(vx, vy, vz);
        //输出速度
        // RCLCPP_INFO(this->get_logger(), "vx = %.2lf vy = %.2lf vz = %.2lf",vx,vy,vz);
        
        double dist_x = std::fabs(ex);
        double dist_y = std::fabs(ez);
        double dist_z = std::fabs(ey);
        
        if (dist_x <= 0.15 && dist_y <= 0.15 && dist_z <= 0.05) 
        {
            return 1;
        } 
        else 
        { 
	     return 0;
        }
    }

    void publish_velocity(double vx, double vy, double vz) {
        auto message = mavros_msgs::msg::PositionTarget();
        message.header.stamp = this->now();
        message.header.frame_id = "map";
        message.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
        
        // 设置掩码只使用速度控制
        message.type_mask = 
            mavros_msgs::msg::PositionTarget::IGNORE_PX |
            mavros_msgs::msg::PositionTarget::IGNORE_PY |
            mavros_msgs::msg::PositionTarget::IGNORE_PZ |
            mavros_msgs::msg::PositionTarget::IGNORE_AFX |
            mavros_msgs::msg::PositionTarget::IGNORE_AFY |
            mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW;

        // 速度限幅
        message.velocity.x = std::clamp(vx, -2.0, 2.0);
        message.velocity.y = std::clamp(vy, -2.0, 2.0);
        message.velocity.z = std::clamp(vz, -2.0, 2.0);
        //message.yaw = 0.0;

        // 发布速度命令
        raw_pub->publish(message);
    }

    bool Ex_vision_fly_to_target(double errx, double py, double pz) {
        double tx = current_pose_.pose.position.x - errx; // 你可能需要根据实际需要来计算 tx
        publish_position(tx, py, pz);
        double dist_x = std::fabs(errx);
        // ring_centre_x 可以用errx 最大值来存 缺点是误识别时会有较大问题
        
        if (dist_x <= 0.008 && errx != 0) 
        {
            return 1;
        } 
        else 
        { 
	    return 0;
        }

    }
    
    void publish_position(double px, double py, double pz) {
        auto message = mavros_msgs::msg::PositionTarget();
        message.header.stamp = this->now();
        message.header.frame_id = "map";
        message.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
        
        // 设置掩码只使用速度控制
        message.type_mask = 
            mavros_msgs::msg::PositionTarget::IGNORE_VX |
            mavros_msgs::msg::PositionTarget::IGNORE_VY |
            mavros_msgs::msg::PositionTarget::IGNORE_VZ |
            mavros_msgs::msg::PositionTarget::IGNORE_AFX |
            mavros_msgs::msg::PositionTarget::IGNORE_AFY |
            mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE |
            mavros_msgs::msg::PositionTarget::IGNORE_YAW;

        // 速度限幅
        message.position.x = px;
        message.position.y = py;
        message.position.z = pz;
        //message.yaw = 0.0;

        // 发布速度命令
        raw_pub->publish(message);
    }

    //舵机控制函数，直接输入要转动的角度
    void control_servo(int num,int angle) {
        std_msgs::msg::Int32 msg;
        msg.data = angle;
        if(num==1)
        {
            servo1_pub_->publish(msg);
        }
        else if(num==2)
        {
            servo2_pub_->publish(msg);
        }
        RCLCPP_INFO(this->get_logger(), "Published servo angle: %d", angle);
    }


    int step_;
    int flag_;
    
    float ring_center_x;

    float land_position_=0;
    //保存扫描到的二维码
    std::string qr_data_;
    
    std_msgs::msg::Float32MultiArray offset_msg_; 

    rclcpp::Time last_request_time_;
    rclcpp::Time start_time_;
    //舵机控制时间延时
    rclcpp::Time servo_action_start_time_;
    //悬停时间
    rclcpp::Time hold_pisition_start_time_;

    bool servo_action_started_ = false;

    bool hold_position_start_ = false;

    bool pid_enabled_ = false;

    mavros_msgs::msg::State current_state_;
    geometry_msgs::msg::PoseStamped current_pose_;
    geometry_msgs::msg::TwistStamped current_vel_;

    PID pid_x_, pid_y_, pid_z_;

    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;

    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_sub_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr qr_data_subscription_;
    
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr ring_offset_sub_;

    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr raw_pub;

    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;

    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr servo1_pub_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr servo2_pub_;

    rclcpp::TimerBase::SharedPtr timer_;
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
    //开启程序主循环
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}