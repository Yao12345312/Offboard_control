#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include <chrono>

using namespace std::chrono_literals;  // This allows the use of the 'ms' suffix

class OffboardControl : public rclcpp::Node
{
public:
    OffboardControl() : Node("offboard_control")
    {
  //[订阅] 无人机状态信息
        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "mavros/state", 10, std::bind(&OffboardControl::state_callback, this,  	std::placeholders::_1));
  // [发布] 位置和姿态信息到话题 "topic" 上，消息类型是 geometry_msgs::msg::PoseStamped          
        local_pos_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "mavros/setpoint_position/local", 10);

// 服务：启动服务用的函数为nh下的serviceClient<>()函数，<>里面是该服务的类型，（）里面是该服务的路径
        //[服务] 修改系统模式
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");
        //[服务] 解锁/上锁 
        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("mavros/cmd/arming");

//创建定时器,每0.1秒发布一次信息,同时绑定OffboardControl类里的回调函数,起到类似定时中断函数的效果
        timer_ = this->create_wall_timer(
            100ms, std::bind(&OffboardControl::timer_callback, this));
//无人机起飞至一定高度

        // pose_.pose.position.x = 0.0;
        // pose_.pose.position.y = 0.0;
        // pose_.pose.position.z = 2.0;
    }

	
private:

     void state_callback(const mavros_msgs::msg::State::SharedPtr msg)
    {
        current_state_ = *msg;
    }
    
    int step_=0;
    rclcpp::Time start_time_=this->now();

    void timer_callback()
{
        
    if (current_state_.mode != "OFFBOARD" )
        {   
            //切换至offboard模式,实际飞行可能需注释,等待使用遥控器切换,使遥控器能紧急接管无人机
            auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
            request->custom_mode = "OFFBOARD";
            set_mode_client_->async_send_request(request);
        }

        if (!current_state_.armed)
        {   
            //若未解锁,请求解锁无人机
            auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
            request->value = true;
            arming_client_->async_send_request(request);
        }
        //执行移动任务
        switch (step_)
        {
        case 0:
            RCLCPP_INFO(this->get_logger(), "Takeoff phase");
            if (this->now() - start_time_ > 5s)
            {
                pose_.pose.position.x = 0.0; 
                pose_.pose.position.y = 0.0;
                pose_.pose.position.z = 2.0;
                start_time_ = this->now();
                step_=1;
            }
            break;
        case 1:
            RCLCPP_INFO(this->get_logger(), "step_1");
            if (this->now() - start_time_ > 5s)
            {
                pose_.pose.position.x = 5.0;
                pose_.pose.position.y = 0.0;
                pose_.pose.position.z = 2.0; 
                start_time_ = this->now();
                step_=2;
            }
            break;
            case 2:
            RCLCPP_INFO(this->get_logger(), "step_2");
            if (this->now() - start_time_ > 5s)
            {
                pose_.pose.position.x = 5.0;
                pose_.pose.position.y = 5.0;
                pose_.pose.position.z = 2.0; 
                start_time_ = this->now();
                step_=3;
            }
            break;
            case 3:
            //仿真中无人机不能完全降落
            RCLCPP_INFO(this->get_logger(), "landing");
            if (this->now() - start_time_ > 5s)
            {
                pose_.pose.position.x = 5.0;
                pose_.pose.position.y = 5.0;
                pose_.pose.position.z = 0.0; //实际还要加上高度传感器离地高度，并考虑到传感器精度（最短测量距离为0.02m）
                start_time_ = this->now();
                step_=4;
            }
            break;
        case 4:
            RCLCPP_INFO(this->get_logger(), "exit_offboard_control");
            //如果飞机十秒内无法完全降落,将在空中解除off board控制
            if (this->now() - start_time_ > 10s)
            {
                auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
                request->value = false;
                arming_client_->async_send_request(request);
                rclcpp::shutdown();
            }
            break;
        }
        //发布位置信息
        local_pos_pub_->publish(pose_);
}
    
 //定义无人机姿态信息
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
//定义发布的位置信息
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_pub_;
 
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;

    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;

    rclcpp::TimerBase::SharedPtr timer_;
//定义无人机状态信息成员
    mavros_msgs::msg::State current_state_;
//定义新的姿态和位置信息
    geometry_msgs::msg::PoseStamped pose_;
    
};

int main(int argc, char **argv)
{
    //初始化ROS2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OffboardControl>();
  
     //启动事件主循环,事件循环会持续监听和处理各种事件，如定时器触发、消息发布、消息订阅等。
    rclcpp::spin(node);
    //关闭节点
    rclcpp::shutdown();
    return 0;
}
