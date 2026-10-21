// Native ROS frontend. CPython owns the reviewed planner and arm executor;
// no Python ROS node, HTTP server or second arm connection is started.
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std::chrono_literals;
using Pose = geometry_msgs::msg::PoseStamped;
using Trigger = std_srvs::srv::Trigger;
using Steady = std::chrono::steady_clock;

static double stamp_seconds(const builtin_interfaces::msg::Time &stamp) {
  return double(stamp.sec) + double(stamp.nanosec)*1e-9;
}
static bool finite(const std::vector<double> &v, size_t size) {
  return v.size() == size && std::all_of(v.begin(), v.end(), [](double x){return std::isfinite(x);});
}
static void quaternion_ok(const std::vector<double> &q) {
  double norm = 0.; for (double v:q) norm += v*v;
  if (!finite(q,4) || std::abs(norm-1.)>1e-5)
    throw std::runtime_error("Quaternion must be finite and normalized [x,y,z,w]");
}

class ObjectFollower final : public rclcpp::Node {
 public:
  explicit ObjectFollower(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : Node("object_follow", options) {
    py::dict config;
    auto parameter = [&](const std::string &key, const auto &value) {
      config[py::str(key)] = py::cast(declare_parameter(key,value));
    };
    parameter("backend",std::string("direct")); parameter("api_url",std::string("http://127.0.0.1:8000"));
    parameter("robot_host",std::string("192.168.77.225")); parameter("robot_port",9000);
    for (const auto &key:{"auto_connect","frame_verified","scene_verified","mount_configured",
                           "mount_assumed","mount_calibrated","approach_enabled"}) parameter(key,false);
    parameter("dry_run",true); parameter("sdk_speed",70); parameter("reference_speed_rad_s",1.);
    parameter("show_rrt",true);
    parameter("approach_camera_fov_fallback",true);
    parameter("optical_frame",std::string("camera_color_optical_frame"));
    parameter("base_frame",std::string("g_base")); parameter("eef_frame",std::string("joint6_flange"));
    parameter("max_detection_age_s",.75); parameter("target_deadband_mm",30.);
    parameter("cone_distance_mm",350.); parameter("cone_half_angle_deg",30.);
    parameter("approach_final_distance_mm",150.); parameter("approach_step_mm",20.);
    parameter("approach_distance_tolerance_mm",10.);
    parameter("horizontal_fov_deg",82.); parameter("vertical_fov_deg",68.);
    parameter("angular_margin_deg",0.); parameter("near_m",.005);
    parameter("eef_to_optical_translation_m",std::vector<double>{0.,0.,0.});
    parameter("eef_to_optical_quaternion_xyzw",std::vector<double>{0.,0.,0.,1.});
    const auto root = declare_parameter("project_root", std::filesystem::current_path().string());
    auto sys = py::module_::import("sys");
    for (const auto &path:{root+"/.venv/lib/python3.12/site-packages",root+"/pymycobot",root,
                          root+"/object_tracking_ros2"}) sys.attr("path").attr("insert")(0,path);
    runtime_ = py::module_::import("object_tracking_ros2.cpp_runtime").attr("Runtime")(
        config,py::cpp_function([this](){return now().seconds();}));
    timings_ = runtime_.attr("timings");
    base_ = get_parameter("base_frame").as_string(); eef_ = get_parameter("eef_frame").as_string();
    optical_ = get_parameter("optical_frame").as_string();
    max_age_ = get_parameter("max_detection_age_s").as_double();
    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock(),tf2::durationFromSec(5.));
    listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_,this,false);
    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    target_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/object_follow/target",10);
    status_pub_ = create_publisher<std_msgs::msg::String>("~/status",10);
    timing_pub_ = create_publisher<std_msgs::msg::String>("~/timing",10);
    planning_pub_ = create_publisher<std_msgs::msg::String>("~/planning_visualization",10);
    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states",10);
    eef_pub_ = create_publisher<Pose>("~/eef_pose",10);
    pose_sub_ = create_subscription<Pose>("/object_pose/filtered",rclcpp::QoS(3).reliable(),
        [this](Pose::ConstSharedPtr message){pose(*message);});
    state_sub_ = create_subscription<std_msgs::msg::String>("/object_pose/tracking_state",10,
        [this](std_msgs::msg::String::ConstSharedPtr message){
          detector_tracking_ = message->data=="TRACKING";
          if (!detector_tracking_) queue_.clear();
          py::gil_scoped_acquire gil; runtime_.attr("set_detector_state")(message->data);
        });
    joints_sub_ = create_subscription<sensor_msgs::msg::JointState>("/arm/command_joints",1,
        [this](sensor_msgs::msg::JointState::ConstSharedPtr message){
          try {
            if (!message->velocity.empty() || !message->effort.empty())
              throw std::runtime_error("Joint goals accept positions only");
            py::gil_scoped_acquire gil;
            runtime_.attr("command_joints")(message->name,message->position,
                stamp_seconds(message->header.stamp),simulation_time());
          } catch (const std::exception &e){RCLCPP_ERROR(get_logger(),"Joint goal rejected: %s",e.what());}
        });
    command_pose_sub_ = create_subscription<Pose>("/arm/command_pose",1,
        [this](Pose::ConstSharedPtr message){
          try {
            const auto &p=message->pose.position; const auto &q=message->pose.orientation;
            py::gil_scoped_acquire gil;
            runtime_.attr("command_pose")(message->header.frame_id,std::vector<double>{p.x,p.y,p.z},
                std::vector<double>{q.x,q.y,q.z,q.w},stamp_seconds(message->header.stamp),simulation_time());
          } catch (const std::exception &e){RCLCPP_ERROR(get_logger(),"Pose goal rejected: %s",e.what());}
        });
    // Separate callback groups keep Stop, detection and feedback runnable while
    // connect or Arm waits on the checked backend. Arm and enable are exclusive.
    arm_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    stop_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    connect_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    disconnect_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    feedback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    service("~/arm",arm_group_,[this](){return call("arm",simulation_time());});
    service("/arm/enable_motion",arm_group_,[this](){return call("enable_motion",simulation_time());});
    service("~/stop",stop_group_,[this](){return call("stop");});
    service("/arm/stop",stop_group_,[this](){return call("stop");});
    service("~/connect",connect_group_,[this](){return call("connect",true);});
    service("~/disconnect",disconnect_group_,[this](){
      queue_reset_.store(true);
      auto answer=call("connect",false); last_feedback_stamp_.store(0.); return answer;
    });
    tick_timer_ = create_wall_timer(50ms,[this](){tick();});
    feedback_timer_ = create_wall_timer(50ms,[this](){feedback();},feedback_group_);
    timing_timer_ = create_wall_timer(1s,[this](){
      py::gil_scoped_acquire gil;
      std_msgs::msg::String message; message.data=runtime_.attr("timing")().cast<std::string>();
      timing_pub_->publish(message);
    });
    if (get_parameter("mount_configured").as_bool() || get_parameter("mount_calibrated").as_bool()) mount();
    auto_connect_ = get_parameter("auto_connect").as_bool() && !get_parameter("dry_run").as_bool();
    if (get_parameter("auto_connect").as_bool() && get_parameter("dry_run").as_bool())
      RCLCPP_WARN(get_logger(),"Ignoring auto_connect in dry run; ~/connect remains explicit");
    RCLCPP_INFO(get_logger(),"C++ ROS frontend ready; checked Python planner/executor retained in process");
  }

  void shutdown() {
    tick_timer_->cancel(); feedback_timer_->cancel(); timing_timer_->cancel();
    // Executor callbacks have finished before this method is called.
    runtime_.attr("shutdown")();
    timings_=py::none(); runtime_=py::none();
  }

 private:
  struct Sample { Pose pose; Steady::time_point received; };
  bool simulation_time() const {return get_parameter("use_sim_time").as_bool();}
  template<class... Args> std::string call(const char *method, Args&&... args) {
    py::gil_scoped_acquire gil;
    return runtime_.attr(method)(std::forward<Args>(args)...).template cast<std::string>();
  }
  void service(const std::string &name,rclcpp::CallbackGroup::SharedPtr group,
               std::function<std::string()> work) {
    services_.push_back(create_service<Trigger>(name,
      [this,work](const Trigger::Request::SharedPtr,Trigger::Response::SharedPtr response){
        try {response->message=work(); response->success=true;}
        catch(const std::exception &e){response->success=false;response->message=e.what();}
      },rclcpp::ServicesQoS(),group));
  }
  void metric(const std::string &name,double milliseconds) {
    py::gil_scoped_acquire gil;timings_.attr("add")(name,milliseconds);
  }
  void duration(const std::string &name,Steady::time_point began) {
    metric(name,std::chrono::duration<double,std::milli>(Steady::now()-began).count());
  }
  void error(const std::string &message,bool expired=false) {
    py::gil_scoped_acquire gil;runtime_.attr("detection_error")(message,expired);
  }
  void mount() {
    auto p=get_parameter("eef_to_optical_translation_m").as_double_array();
    auto q=get_parameter("eef_to_optical_quaternion_xyzw").as_double_array();
    quaternion_ok(q);if (!finite(p,3)) throw std::runtime_error("Invalid optical translation");
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp=now();t.header.frame_id=eef_;t.child_frame_id=optical_;
    t.transform.translation.x=p[0];t.transform.translation.y=p[1];t.transform.translation.z=p[2];
    t.transform.rotation.x=q[0];t.transform.rotation.y=q[1];t.transform.rotation.z=q[2];t.transform.rotation.w=q[3];
    static_->sendTransform(t);
  }
  void pose(const Pose &message) {
    const auto began=Steady::now();
    try {
      if (!detector_tracking_) return;
      double stamp=stamp_seconds(message.header.stamp), current=now().seconds();
      if (stamp<=last_stamp_) return;
      const auto &p=message.pose.position;
      if (message.header.frame_id!=optical_) throw std::runtime_error("Unexpected detection frame");
      if (!finite({p.x,p.y,p.z},3)) throw std::runtime_error("Object centre must contain three finite metres");
      if (p.z<=0. || p.z>5.) throw std::runtime_error("Invalid optical depth (expected metres)");
      if (!std::isfinite(stamp)||stamp<=0.||stamp<=last_received_stamp_)
        throw std::runtime_error("Missing or out-of-order capture stamp");
      if (current-stamp>max_age_ || stamp>current+.02) {
        error("Stale or future detection",current-stamp>max_age_);return;
      }
      last_received_stamp_=stamp;
      metric("detection.capture_to_callback",(current-stamp)*1000);
      if (queue_.size()==32) queue_.pop_front();
      queue_.push_back({message,Steady::now()});
    } catch (const std::exception &e){error(e.what());}
    duration("ros.detection_callback",began);
  }
  void process_pose() {
    const auto began=Steady::now();const double current=now().seconds();
    if (queue_reset_.exchange(false)) queue_.clear();
    while (!queue_.empty()) {
      double stamp=stamp_seconds(queue_.front().pose.header.stamp);
      if (stamp<=last_stamp_ || current-stamp>max_age_) queue_.pop_front();else break;
    }
    for (auto it=queue_.rbegin();it!=queue_.rend();++it) {
      const auto &message=it->pose;const double stamp=stamp_seconds(message.header.stamp);
      if (stamp>last_feedback_stamp_.load()) continue;
      geometry_msgs::msg::TransformStamped t;
      try {t=buffer_->lookupTransform(base_,message.header.frame_id,rclcpp::Time(message.header.stamp));}
      catch(const tf2::TransformException &){continue;}
      try {
        const auto &r=t.transform.rotation;quaternion_ok({r.x,r.y,r.z,r.w});
        const auto &p=message.pose.position;
        auto v=tf2::quatRotate(tf2::Quaternion(r.x,r.y,r.z,r.w),tf2::Vector3(p.x,p.y,p.z));
        std::vector<double> target{1000*(v.x()+t.transform.translation.x),
          1000*(v.y()+t.transform.translation.y),1000*(v.z()+t.transform.translation.z)};
        {py::gil_scoped_acquire gil;target=runtime_.attr("submit")(target,stamp).cast<std::vector<double>>();}
        geometry_msgs::msg::PointStamped out;out.header=message.header;out.header.frame_id=base_;
        out.point.x=target[0]/1000;out.point.y=target[1]/1000;out.point.z=target[2]/1000;
        target_pub_->publish(out);
        metric("detection.capture_to_target",(now().seconds()-stamp)*1000);
        duration("detection.callback_to_target",it->received);
        last_stamp_=stamp;
      } catch(const std::exception &e){error(e.what());}
      break;
    }
    duration("ros.process_pose",began);
  }
  void tick() {
    auto began=Steady::now();
    if (last_tick_!=Steady::time_point{}) duration("ros.tick_interval",last_tick_);
    last_tick_=began;
    process_pose();
    try {
      py::gil_scoped_acquire gil;runtime_.attr("tick")(get_parameter("show_rrt").as_bool());
      std_msgs::msg::String message;message.data=runtime_.attr("status")().cast<std::string>();
      status_pub_->publish(message);
      message.data=runtime_.attr("planning_display")().cast<std::string>();
      if (!message.data.empty()) planning_pub_->publish(message);
    } catch(const std::exception &e){RCLCPP_ERROR(get_logger(),"Tracking tick: %s",e.what());}
    duration("ros.tick",began);
  }
  void feedback() {
    auto began=Steady::now();
    try {
      py::gil_scoped_acquire gil;
      if (auto_connect_) {
        auto_connect_=false;
        try {runtime_.attr("connect")(true);}
        catch(const std::exception &e){RCLCPP_ERROR(get_logger(),"Connect: %s",e.what());}
      }
      auto samples=runtime_.attr("feedback")().cast<py::list>();
      if (samples.empty()) {duration("ros.feedback_worker",began);return;}
      const double current=now().seconds();
      const double latest=samples[samples.size()-1].cast<py::dict>()["stamp_unix_s"].cast<double>();
      if (!std::isfinite(latest)||latest>current+.02||current-latest>max_age_)
        throw std::runtime_error("Stale backend feedback pose");
      std::vector<geometry_msgs::msg::TransformStamped> transforms;
      double previous=last_feedback_stamp_.load();
      for (auto item:samples) {
        auto sample=item.cast<py::dict>();const double stamp=sample["stamp_unix_s"].cast<double>();
        if (!std::isfinite(stamp)||stamp>current+.02) throw std::runtime_error("Invalid backend feedback stamp");
        if (stamp<=previous) continue;
        if (sample["frame_id"].cast<std::string>()!=base_ || sample["child_frame_id"].cast<std::string>()!=eef_)
          throw std::runtime_error("Backend feedback frame mismatch");
        auto p=sample["translation_m"].cast<std::vector<double>>();
        auto q=sample["quaternion_xyzw"].cast<std::vector<double>>();quaternion_ok(q);
        if(!finite(p,3)) throw std::runtime_error("Invalid backend feedback translation");
        geometry_msgs::msg::TransformStamped t;t.header.stamp=rclcpp::Time(int64_t(stamp*1e9));
        t.header.frame_id=base_;t.child_frame_id=eef_;
        t.transform.translation.x=p[0];t.transform.translation.y=p[1];t.transform.translation.z=p[2];
        t.transform.rotation.x=q[0];t.transform.rotation.y=q[1];t.transform.rotation.z=q[2];t.transform.rotation.w=q[3];
        Pose e;e.header=t.header;e.pose.position.x=p[0];e.pose.position.y=p[1];e.pose.position.z=p[2];
        e.pose.orientation=t.transform.rotation;eef_pub_->publish(e);
        if (sample.contains("angles_deg") && sample.contains("joint_names")) {
          auto angles=sample["angles_deg"].cast<std::vector<double>>();
          auto names=sample["joint_names"].cast<std::vector<std::string>>();
          auto unique=names;std::sort(unique.begin(),unique.end());
          if (!finite(angles,6)||names.size()!=6||std::adjacent_find(unique.begin(),unique.end())!=unique.end())
            throw std::runtime_error("Invalid measured joint feedback");
          sensor_msgs::msg::JointState j;j.header=t.header;j.name=names;
          for(double angle:angles) j.position.push_back(angle*std::acos(-1.)/180.);
          joint_pub_->publish(j);
        }
        transforms.push_back(t);previous=stamp;
        metric("feedback.capture_to_publish",(current-stamp)*1000);
      }
      if (!transforms.empty()) {
        broadcaster_->sendTransform(transforms);
        // Avoid a DDS roundtrip before exact-stamp camera lookup can run.
        for(const auto &t:transforms) buffer_->setTransform(t,"measured-backend",false);
        last_feedback_stamp_.store(previous);
      }
      runtime_.attr("feedback_health")(previous);
    } catch(const std::exception &e){
      py::gil_scoped_acquire gil;runtime_.attr("feedback_failure")(std::string(e.what()));
    }
    duration("ros.feedback_worker",began);
  }
  py::object runtime_,timings_;
  std::string base_,eef_,optical_;
  double max_age_=0.75,last_stamp_=0.,last_received_stamp_=0.;
  std::atomic<double> last_feedback_stamp_{0.};
  std::atomic<bool> queue_reset_{false};
  bool detector_tracking_=false,auto_connect_=false;
  std::deque<Sample> queue_;Steady::time_point last_tick_{};
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_;
  rclcpp::CallbackGroup::SharedPtr arm_group_,stop_group_,connect_group_,disconnect_group_,feedback_group_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> services_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_pub_;
  rclcpp::Publisher<Pose>::SharedPtr eef_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_,timing_pub_,planning_pub_;
  rclcpp::Subscription<Pose>::SharedPtr pose_sub_,command_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joints_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::TimerBase::SharedPtr tick_timer_,feedback_timer_,timing_timer_;
};

int main(int argc,char **argv) {
  // Construct and destroy all Python objects while holding the GIL. The ROS
  // executor releases it; each callback acquires only around runtime calls.
  // rclcpp owns SIGINT/SIGTERM. Installing CPython's signal handlers would
  // leave a pending KeyboardInterrupt that interrupts delivery of Stop.
  py::scoped_interpreter interpreter{false};
  rclcpp::init(argc,argv);
  std::shared_ptr<ObjectFollower> node;
  try {
    node=std::make_shared<ObjectFollower>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),6);
    executor.add_node(node);
    {py::gil_scoped_release release;executor.spin();}
    node->shutdown();executor.remove_node(node);node.reset();
  } catch(const std::exception &e) {
    std::cerr<<"C++ tracking node: "<<e.what()<<std::endl;
    if (node) {
      try {node->shutdown();}
      catch(const std::exception &cleanup){std::cerr<<"Shutdown: "<<cleanup.what()<<std::endl;}
    }
    rclcpp::shutdown();
    // A timed-out Python worker must never run against a finalized interpreter.
    // Process exit also closes any owned robot connection.
    std::_Exit(1);
  }
  rclcpp::shutdown();return 0;
}
