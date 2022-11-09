FROM osrf/ros:melodic-desktop-full
# FROM osrf/ros:kinetic-desktop-full
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,display

# add non-root user
RUN useradd -m vscode && \
    echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode && \
    chmod 0440 /etc/sudoers.d/vscode

ENV SHELL /bin/bash

# init rosdep
RUN rosdep update

RUN apt-get update && apt-get install -y apt-utils build-essential

RUN apt-get update && apt-get install -q -y python-catkin-tools
RUN apt-get update && apt-get install -q -y ros-$ROS_DISTRO-hector-gazebo-plugins

RUN apt-get install dbus libxext6 libxrender1 libxtst6 libxi6 libxxf86vm1 mesa-utils python3-pip x11-apps xterm iputils-ping -y

RUN apt-get update && apt-get install ros-$ROS_DISTRO-ros-control ros-$ROS_DISTRO-ros-controllers ros-$ROS_DISTRO-joy -y



## For Astra
RUN apt-get install ros-*-rgbd-launch ros-*-libuvc ros-*-libuvc-camera ros-*-libuvc-ros
RUN apt-get install ros-$ROS_DISTRO-openni-* ros-$ROS_DISTRO-openni2-* ros-$ROS_DISTRO-freenect-* -y
RUN apt-get install usbutils


RUN apt install wget

# RUN apt-get install ros-melodic-joy ros-melodic-teleop-twist-joy \
#   ros-melodic-teleop-twist-keyboard ros-melodic-laser-proc \
#   ros-melodic-rgbd-launch ros-melodic-depthimage-to-laserscan \
#   ros-melodic-rosserial-arduino ros-melodic-rosserial-python \
#   ros-melodic-rosserial-server ros-melodic-rosserial-client \
#   ros-melodic-rosserial-msgs ros-melodic-amcl ros-melodic-map-server \
#   ros-melodic-move-base ros-melodic-urdf ros-melodic-xacro \
#   ros-melodic-compressed-image-transport ros-melodic-rqt* \
#   ros-melodic-gmapping ros-melodic-navigation ros-melodic-interactive-markers -y

RUN apt-get install ros-$ROS_DISTRO-dynamixel-sdk -y

RUN apt-get install ros-$ROS_DISTRO-turtlebot3-msgs -y

RUN apt-get install ros-$ROS_DISTRO-turtlebot3 -y

RUN apt-get install ros-$ROS_DISTRO-realsense2-* -y

# RUN spt-get install ros-melodic-orb-slam2-ros -y

# RUN pip3 install pandas

RUN apt-get install python-pip -y

# RUN pip install pandas

# RUN pip3 install scikit-image

# RUN pip3 install scikit-learn

# RUN pip3 install matplotlib

# RUN pip3 install pyyaml

# RUN pip3 install dotmap

# RUN pip3 install opencv-python
