#! bin/bash

# Fix Qt rendering bugs
# echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc

# echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc


echo "export QT_X11_NO_MITSHM=1" >> ~/.bashrc
echo "export QT_GRAPHICSSYSTEM="native"" >> ~/.bashrc


echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
echo "source /workspaces/Turtlebot2/devel/setup.bash" >> ~/.bashrc

# Turtlebot specific specific
echo "source /workspaces/Turtlebot2/src/turtlebot/setup_create.sh" >> ~/.bashrc
echo "source /workspaces/Turtlebot2/src/turtlebot/setup_kobuki.sh" >> ~/.bashrc

# echo "export ROS_MASTER_URI=http://localhost:11311" >> ~/.bashrc
# IP=$(hostname -i) 
# echo "export ROS_HOSTNAME=$IP" >> ~/.bashrc

source ~/.bashrc

sudo cp .devcontainer/99-realsense-libusb.rules /etc/udev/rules.d


rosdep update

# echo "export ROS_MASTER_URI=http://192.168.60.112:11311/" >> ~/.bashrc # Different everytime
# echo "export ROS_IP=192.168.60.112" >> ~/.bashrc
# echo "export ROS_HOSTNAME=192.168.60.112" >> ~/.bashrc