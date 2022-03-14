/*
 * Copyright 2016 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#include <algorithm>
#include <fstream>
#include <mutex>
#include <thread>
#include <string>

#include <ignition/math/Pose3.hh>
#include <ignition/transport/Node.hh>
#include <ignition/transport/AdvertiseOptions.hh>

#include "HumanPlugin.hh"
#include <gazebo/common/PID.hh>
#include <gazebo/common/Time.hh>
#include "HumanPlugin.hh"

#include <ros/ros.h>

namespace gazebo
{
  class HumanPluginPrivate
  {
    /// \enum DirectionType
    /// \brief Direction selector switch type.
    public: enum DirectionType {
              /// \brief Reverse
              REVERSE = -1,
              /// \brief Neutral
              NEUTRAL = 0,
              /// \brief Forward
              FORWARD = 1
            };

    public: ros::NodeHandle nh;

    public: ros::Subscriber sub_cmd_;
    public: ros::Publisher pub_status_;
    public: ros::Publisher pub_pose_;

    public: std::string model_name;

    /// \brief Pointer to the world
    public: physics::WorldPtr world;

    /// \brief Pointer to the parent model
    public: physics::ModelPtr model;
    public: physics::ActorPtr actor;

    /// \brief Transport node
    public: transport::NodePtr gznode;

    /// \brief Ignition transport node
    public: ignition::transport::Node node;

    /// \brief Ignition transport position pub
    public: ignition::transport::Node::Publisher posePub;

    /// \brief Ignition transport console pub
    public: ignition::transport::Node::Publisher consolePub;

    /// \brief Physics update event connection
    public: event::ConnectionPtr updateConnection;

    /// \brief Chassis link
    public: physics::LinkPtr chassisLink;

    /// \brief Front left wheel joint
    public: physics::JointPtr flWheelJoint;

    /// \brief Front right wheel joint
    public: physics::JointPtr frWheelJoint;

    /// \brief Rear left wheel joint
    public: physics::JointPtr blWheelJoint;

    /// \brief Rear right wheel joint
    public: physics::JointPtr brWheelJoint;

    /// \brief Front left wheel steering joint
    public: physics::JointPtr flWheelSteeringJoint;

    /// \brief Front right wheel steering joint
    public: physics::JointPtr frWheelSteeringJoint;

    /// \brief Steering wheel joint
    public: physics::JointPtr handWheelJoint;

    /// \brief PID control for the front left wheel steering joint
    public: common::PID flWheelSteeringPID;

    /// \brief PID control for the front right wheel steering joint
    public: common::PID frWheelSteeringPID;

    /// \brief PID control for steering wheel joint
    public: common::PID handWheelPID;

    /// \brief Last pose msg time
    public: common::Time lastMsgTime;

    /// \brief Last sim time received
    public: common::Time lastSimTime;

    /// \brief Last sim time when a pedal command is received
    public: common::Time lastPedalCmdTime;

    /// \brief Last sim time when a steering command is received
    public: common::Time lastSteeringCmdTime;

    /// \brief Last sim time when a EV mode command is received
    public: common::Time lastModeCmdTime;

    /// \brief Current direction of the vehicle: FORWARD, NEUTRAL, REVERSE.
    public: DirectionType directionState;

    /// \brief Chassis aerodynamic drag force coefficient,
    /// with units of [N / (m/s)^2]
    public: double chassisAeroForceGain = 0;

    /// \brief Max torque that can be applied to the front wheels
    public: double frontTorque = 0;

    /// \brief Max torque that can be applied to the back wheels
    public: double backTorque = 0;

    /// \brief Max speed (m/s) of the car
    public: double maxSpeed = 0;

    /// \brief Max steering angle
    public: double maxSteer = 0;

    /// \brief Max torque that can be applied to the front brakes
    public: double frontBrakeTorque = 0;

    /// \brief Max torque that can be applied to the rear brakes
    public: double backBrakeTorque = 0;

    /// \brief Angle ratio between the steering wheel and the front wheels
    public: double steeringRatio = 0;

    /// \brief Max range of hand steering wheel
    public: double handWheelHigh = 0;

    /// \brief Min range of hand steering wheel
    public: double handWheelLow = 0;

    /// \brief Front left wheel desired steering angle (radians)
    public: double flWheelSteeringCmd = 0;

    /// \brief Front right wheel desired steering angle (radians)
    public: double frWheelSteeringCmd = 0;

    /// \brief Steering wheel desired angle (radians)
    public: double handWheelCmd = 0;

    /// \brief Front left wheel radius
    public: double flWheelRadius = 0;

    /// \brief Front right wheel radius
    public: double frWheelRadius = 0;

    /// \brief Rear left wheel radius
    public: double blWheelRadius = 0;

    /// \brief Rear right wheel radius
    public: double brWheelRadius = 0;

    /// \brief Front left joint friction
    public: double flJointFriction = 0;

    /// \brief Front right joint friction
    public: double frJointFriction = 0;

    /// \brief Rear left joint friction
    public: double blJointFriction = 0;

    /// \brief Rear right joint friction
    public: double brJointFriction = 0;

    /// \brief Distance distance between front and rear axles
    public: double wheelbaseLength = 0;

    /// \brief Distance distance between front left and right wheels
    public: double frontTrackWidth = 0;

    /// \brief Distance distance between rear left and right wheels
    public: double backTrackWidth = 0;

    /// \brief Gas energy density (J/gallon)
    public: const double kGasEnergyDensity = 1.29e8;

    /// \brief Battery charge capacity in Watt-hours
    public: double batteryChargeWattHours = 280;

    /// \brief Battery discharge capacity in Watt-hours
    public: double batteryDischargeWattHours = 260;

    /// \brief Gas engine efficiency
    public: double gasEfficiency = 0.37;

    /// \brief Minimum gas flow rate (gallons / sec)
    public: double minGasFlow = 1e-4;

    /// \brief Gas consumption (gallon)
    public: double gasConsumption = 0;

    /// \brief Battery state-of-charge (percent, 0.0 - 1.0)
    public: double batteryCharge = 0.75;

    /// \brief Battery charge threshold when it has to be recharged.
    public: const double batteryLowThreshold = 0.125;

    /// \brief Whether EV mode is on or off.
    public: bool evMode = false;

    /// \brief Gas pedal position in percentage. 1.0 = Fully accelerated.
    public: double gasPedalPercent = 0;

    /// \brief Power for charging a low battery (Watts).
    public: const double kLowBatteryChargePower = 2000;

    /// \brief Threshold delimiting the gas pedal (throttle) low and medium
    /// ranges.
    public: const double kGasPedalLowMedium = 0.25;

    /// \brief Threshold delimiting the gas pedal (throttle) medium and high
    /// ranges.
    public: const double kGasPedalMediumHigh = 0.5;

    /// \brief Threshold delimiting the speed (throttle) low and medium
    /// ranges in miles/h.
    public: const double speedLowMedium = 26.0;

    /// \brief Threshold delimiting the speed (throttle) medium and high
    /// ranges in miles/h.
    public: const double speedMediumHigh = 46.0;

    /// \brief Brake pedal position in percentage. 1.0 =
    public: double brakePedalPercent = 0;

    /// \brief Hand brake position in percentage.
    public: double handbrakePercent = 1.0;

    /// \brief Angle of steering wheel at last update (radians)
    public: double handWheelAngle = 0;

    /// \brief Steering angle of front left wheel at last update (radians)
    public: double flSteeringAngle = 0;

    /// \brief Steering angle of front right wheel at last update (radians)
    public: double frSteeringAngle = 0;

    /// \brief Linear velocity of chassis c.g. in world frame at last update (m/s)
    public: ignition::math::Vector3d chassisLinearVelocity;

    /// \brief Angular velocity of front left wheel at last update (rad/s)
    public: double flWheelAngularVelocity = 0;

    /// \brief Angular velocity of front right wheel at last update (rad/s)
    public: double frWheelAngularVelocity = 0;

    /// \brief Angular velocity of back left wheel at last update (rad/s)
    public: double blWheelAngularVelocity = 0;

    /// \brief Angular velocity of back right wheel at last update (rad/s)
    public: double brWheelAngularVelocity = 0;

    /// \brief Subscriber to the keyboard topic
    public: transport::SubscriberPtr keyboardSub;

    /// \brief Mutex to protect updates
    public: std::mutex mutex;

    /// \brief Odometer
    public: double odom = 0.0;

    /// \brief Keyboard control type
    public: int keyControl = 1;

    /// \brief Publisher for the world_control topic.
    public: transport::PublisherPtr worldControlPub;

    public: int status_;
    public: ignition::math::Vector3d goal_;

    public: double acc_scale_;
    public: double str_scale_;
    public: double min_acc_;
    public: double max_acc_;
    public: double max_dstr_;
    public: double tar_vel_;
    public: geometry_msgs::Point init_pose;

/// \brief Velocity of the actor
    public: ignition::math::Vector3d velocity;

    /// \brief List of connections
    public: std::vector<event::ConnectionPtr> connections;

    /// \brief Current target location
    public: ignition::math::Vector3d target;

    /// \brief Target location weight (used for vector field)
    public: double targetWeight = 1.0;

    /// \brief Obstacle weight (used for vector field)
    public: double obstacleWeight = 1.0;

    /// \brief Time scaling factor. Used to coordinate translational motion
    /// with the actor's walking animation.
    public: double animationFactor = 1.0;

    /// \brief Time of the last update.
    public: common::Time lastUpdate;

    /// \brief List of models to ignore. Used for vector field
    public: std::vector<std::string> ignoreModels;

    /// \brief Custom trajectory info.
    public: physics::TrajectoryInfoPtr trajectoryInfo;

    public: int timestep;

  };
}

using namespace gazebo;

/////////////////////////////////////////////////
HumanPlugin::HumanPlugin()
    : dataPtr(new HumanPluginPrivate)
{
  int argc = 0;
  char *argv = nullptr;
  ros::init(argc, &argv, "HumanPlugin");
  this->robot_namespace_ = "";
}


/////////////////////////////////////////////////
HumanPlugin::~HumanPlugin()
{
  this->dataPtr->updateConnection.reset();
}

/////////////////////////////////////////////////
void HumanPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  gzwarn << "HumanPlugin loading params" << std::endl;
  // shortcut to this->dataPtr
  HumanPluginPrivate *dPtr = this->dataPtr.get();

//   this->dataPtr->model = _model;
//   this->dataPtr->world = this->dataPtr->actor->GetWorld();
  
  this->dataPtr->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
  this->dataPtr->world = this->dataPtr->actor->GetWorld();
  
  auto physicsEngine = this->dataPtr->world->Physics();
  physicsEngine->SetParam("friction_model", std::string("cone_model"));

  this->dataPtr->gznode = transport::NodePtr(new transport::Node());
  this->dataPtr->gznode->Init();

  double x = 0.0;
  double y = 0.0;
  if (_sdf->HasElement("x")) x = _sdf->Get<double>("x");
  if (_sdf->HasElement("x")) y = _sdf->Get<double>("y");
  this->dataPtr->init_pose.x = x;
  this->dataPtr->init_pose.y = y;

  this->ResetActor();

  if (_sdf->HasElement("robotNamespace"))
    this->robot_namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>() + "/";
  ros::NodeHandle nh(this->robot_namespace_);
  this->dataPtr->model_name = _sdf->Get<std::string>("model_name");
  
  // this->dataPtr->controlSub = nh.subscribe("prius", 10, &HumanPlugin::OnPriusCommand, this);
  this->dataPtr->sub_cmd_ = nh.subscribe(this->dataPtr->model_name + "/cmd", 10, &HumanPlugin::CallbackCmd, this);
  this->dataPtr->pub_status_ = nh.advertise<social_navigation::Status>(this->dataPtr->model_name + "/status", 1000);
  this->dataPtr->pub_pose_ = nh.advertise<geometry_msgs::PoseStamped>(this->dataPtr->model_name + "/pose", 1000);
 
  if (_sdf->HasElement("target_weight"))
    this->dataPtr->targetWeight = _sdf->Get<double>("target_weight");
  else
    this->dataPtr->targetWeight = 1.15;

  // Read in the obstacle weight
  if (_sdf->HasElement("obstacle_weight"))
    this->dataPtr->obstacleWeight = _sdf->Get<double>("obstacle_weight");
  else
    this->dataPtr->obstacleWeight = 1.5;

  // Read in the animation factor (applied in the OnUpdate function).
  if (_sdf->HasElement("animation_factor"))
    this->dataPtr->animationFactor = _sdf->Get<double>("animation_factor");
  else
    this->dataPtr->animationFactor = 4.5;

  // Add our own name to models we should ignore when avoiding obstacles.
  this->dataPtr->ignoreModels.push_back(this->dataPtr->actor->GetName());

  // Read in the other obstacles to ignore
  if (_sdf->HasElement("ignore_obstacles"))
  {
    sdf::ElementPtr modelElem =
      _sdf->GetElement("ignore_obstacles")->GetElement("model");
    while (modelElem)
    {
      this->dataPtr->ignoreModels.push_back(modelElem->Get<std::string>());
      modelElem = modelElem->GetNextElement("model");
    }
  }

  
  

  this->dataPtr->updateConnection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&HumanPlugin::Update, this));
}

void HumanPlugin::ResetActor()
{
  this->dataPtr->timestep = 0;
  this->dataPtr->lastSimTime = 0;
  this->dataPtr->tar_vel_ = 0.0;
  this->dataPtr->lastUpdate = 0;

  this->dataPtr->status_ = WAIT;
  this->dataPtr->goal_ = ignition::math::Vector3d(this->dataPtr->init_pose.x,this->dataPtr->init_pose.y,1.2138);

  auto skelAnims = this->dataPtr->actor->SkeletonAnimations();
  if (skelAnims.find(WALKING_ANIMATION) == skelAnims.end())
  {
    gzerr << "Skeleton animation " << WALKING_ANIMATION << " not found.\n";
  }
  else
  {
    // Create custom trajectory
    this->dataPtr->trajectoryInfo.reset(new physics::TrajectoryInfo());
    this->dataPtr->trajectoryInfo->type = WALKING_ANIMATION;
    this->dataPtr->trajectoryInfo->duration = 1.0;

    this->dataPtr->actor->SetCustomTrajectory(this->dataPtr->trajectoryInfo);
  }
}

void HumanPlugin::CallbackCmd(const social_navigation::Command::ConstPtr& msg){
  std::string name = msg->name;
  // if(name.compare(this->dataPtr->model_name.c_str()) != 0){
  //   gzwarn << "Mismatch actor name!\n"; 
  //   return;
  // }
  this->dataPtr->status_ = msg->status;
  this->dataPtr->goal_ = ignition::math::Vector3d(msg->goal.position.x, msg->goal.position.y, 1.2138);
  this->dataPtr->tar_vel_ = msg->velocity;
  return;
}


void HumanPlugin::Reset()
{
}

/////////////////////////////////////////////////
void HumanPlugin::Update()
{
  this->dataPtr->timestep += 1;
  // shortcut to this->dataPtr
  HumanPluginPrivate *dPtr = this->dataPtr.get();

  std::lock_guard<std::mutex> lock(this->dataPtr->mutex);

  common::Time curTime = this->dataPtr->world->SimTime();
  double dt = (curTime - this->dataPtr->lastSimTime).Double();
  ignition::math::Pose3d pose = this->dataPtr->actor->WorldPose();

  if(this->dataPtr->status_ == INIT){
    if ((pose.Pos() - this->dataPtr->goal_).Length() < 0.1) this->dataPtr->status_ = MOVE;
    pose.Pos() = this->dataPtr->goal_;
    pose.Rot() = ignition::math::Quaterniond(1.5707, 0, 0.0);
    this->dataPtr->actor->SetWorldPose(pose, false, false);
  }
  else if(this->dataPtr->status_ == MOVE){
    ignition::math::Vector3d pos = this->dataPtr->goal_ - pose.Pos();
    ignition::math::Vector3d rpy = pose.Rot().Euler();

    double distance = pos.Length();

    pos = pos.Normalize() * this->dataPtr->targetWeight;


    // Compute the yaw orientation
    ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
    yaw.Normalize();

    pose.Pos() += pos * this->dataPtr->tar_vel_ * dt;
    pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z()+yaw.Radian());


    double distanceTraveled = (pose.Pos() - this->dataPtr->actor->WorldPose().Pos()).Length();

    this->dataPtr->actor->SetWorldPose(pose, false, false);
    this->dataPtr->actor->SetScriptTime(this->dataPtr->actor->ScriptTime() +
      (distanceTraveled * this->dataPtr->animationFactor));
  }

  this->dataPtr->lastSimTime = curTime;
  geometry_msgs::PoseStamped rt;
  rt.pose.position.x = pose.Pos().X();
  rt.pose.position.y = pose.Pos().Y();
  rt.pose.position.z = pose.Pos().Z();
  rt.pose.orientation.x = pose.Rot().X();
  rt.pose.orientation.y = pose.Rot().Y();
  rt.pose.orientation.z = pose.Rot().Z();
  rt.pose.orientation.w = pose.Rot().W();
  rt.header.frame_id = this->dataPtr->model_name;
  this->dataPtr->pub_pose_.publish(rt);

  social_navigation::Status rt_status;
  rt_status.name = this->dataPtr->model_name.c_str();
  rt_status.status = this->dataPtr->status_;
  this->dataPtr->pub_status_.publish(rt_status);

}

GZ_REGISTER_MODEL_PLUGIN(HumanPlugin)
