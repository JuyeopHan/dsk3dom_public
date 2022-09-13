# DS-K3DOM

Implementation of DS-K3DOM

## How to build

Implementation for simulation dataset

```console
cd ~/catkin_ws/src
git clone https://github.com/JuyeopHan/DS-K3DOM.git
cd ~/catkin_ws && catkin_make
source devel/setup.bash
roslaunch ds_k3dom ds_k3dom_demo.launch
```

or

Implementation for indoor dataset

```console
cd ~/catkin_ws/src
git clone https://github.com/JuyeopHan/DS-K3DOM.git
cd ~/catkin_ws && catkin_make
source devel/setup.bash
roslaunch ds_k3dom ds_k3dom_exp.launch
```


<p align="center">
  <img src="./docs/simulation.gif">
</p>

<p align="center">
  <img src="./docs/experiment.gif">
</p>


## References

Juyeop Han*, Youngjae Min*, Hyeok-Joo Chae, Byeong-Min Jeong, and Han-Lim Choi, "DS-K3DOM: 3-D Dynamic Occupancy Mapping with Kernel Inference
and Dempster-Shafer Evidential Theory" submitted to 2023 IEEE International Conference on Robotics and Automation (ICRA)

This code is extented from [youngjae-min/k3dom](https://github.com/youngjae-min/k3dom).
