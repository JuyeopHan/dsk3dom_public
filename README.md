# DS-K3DOM

Implementation of DS-K3DOM : 3-D Dynamic Occupancy Mapping with Kernel Inference and Dempster-Shafer Evidential Theory

## How to build

```console
cd ~/catkin_ws/src
git clone https://github.com/JuyeopHan/DS-K3DOM.git
cd ~/catkin_ws && catkin_make
```

## Implementation for simulation dataset

```console
cd ~/catkin_ws
source devel/setup.bash
roslaunch ds_k3dom ds_k3dom_demo.launch
```


## Implementation for indoor dataset

```console
cd ~/catkin_ws
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
and Dempster-Shafer Evidential Theory", 2023 IEEE International Conference on Robotics and Automation (ICRA) \[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10160364)] \[[ArXiv](https://arxiv.org/abs/2209.07764)\] \[[youtube](https://www.youtube.com/watch?v=l_vRqDmNyWE)\] 

This code is extented from [youngjae-min/k3dom](https://github.com/youngjae-min/k3dom).

## Citation

```console
@INPROCEEDINGS{
  DSK3DOM,
  author={Han, Juyeop and Min, Youngjae and Chae, Hyeok-Joo and Jeong, Byeong-Min and Choi, Han-Lim},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={DS-K3DOM: 3-D Dynamic Occupancy Mapping with Kernel Inference and Dempster-Shafer Evidential Theory}, 
  year={2023},
  volume={},
  number={},
  pages={6217-6223},
  keywords={Atmospheric measurements;Heuristic algorithms;Sensor fusion;Approximation algorithms;Particle measurements;Set theory;Robot sensing systems},
  doi={10.1109/ICRA48891.2023.10160364}
}
```
