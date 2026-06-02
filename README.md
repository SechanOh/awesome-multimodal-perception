# Awesome Autonomous Driving and Robotics Background

A reading roadmap for skimming essential papers and lectures for autonomous driving and robotics background.  
This roadmap prioritizes conceptual coverage and system-level understanding over simply listing the latest SOTA papers.

Last updated: 2026-06-02

## How to Use This Roadmap

- **Core**: read first; useful background for most autonomous driving and robotics work.
- **Bridge**: connects classical ideas to modern deep learning systems.
- **Modern**: representative recent direction; useful for understanding current research.
- **Advanced**: optional frontier material; skim after the basics.

## Table of Contents

- [1. 2D Object Detection](#1-2d-object-detection)
- [2. Segmentation and Scene Parsing](#2-segmentation-and-scene-parsing)
- [3. Object Tracking and Multi-Object Tracking](#3-object-tracking-and-multi-object-tracking)
- [4. Depth Estimation and Stereo](#4-depth-estimation-and-stereo)
- [5. Optical Flow](#5-optical-flow)
- [6. Multiple View Geometry and 3D Vision](#6-multiple-view-geometry-and-3d-vision)
- [7. 3D Object Detection and BEV Perception](#7-3d-object-detection-and-bev-perception)
- [8. SLAM, Odometry, and Occupancy](#8-slam-odometry-and-occupancy)
- [9. Motion Forecasting](#9-motion-forecasting)
- [10. Planning and Control](#10-planning-and-control)
- [11. Robot Learning](#11-robot-learning)
- [12. Foundation Models for Vision and Robotics](#12-foundation-models-for-vision-and-robotics)
- [13. End-to-End Autonomous Driving](#13-end-to-end-autonomous-driving)
- [14. Generative Vision and World Models](#14-generative-vision-and-world-models)
- [15. Robustness, Uncertainty, and Safety](#15-robustness-uncertainty-and-safety)

## 1. 2D Object Detection

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| Faster R-CNN → YOLOv3 | <kbd>two-stage detection</kbd> → <kbd>one-stage real-time detection</kbd> |
| YOLOv3 → DETR | <kbd>multi-scale dense prediction</kbd> → <kbd>object queries</kbd>, <kbd>set prediction</kbd> |
| DETR → Deformable DETR → DINO | <kbd>NMS-free detection</kbd> → <kbd>sparse multi-scale attention</kbd>, <kbd>strong DETR training recipes</kbd> |
| GLIP → Grounding DINO | <kbd>closed-set detection</kbd> → <kbd>language-conditioned open-set detection</kbd> |

### Reading Order

1. **[Core]** [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS 2015  
   Skim for: <kbd>region proposal networks</kbd>, <kbd>two-stage detection</kbd>, <kbd>anchors</kbd>, <kbd>RoI heads</kbd>

2. **[Core]** [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767), 2018  
   Skim for: <kbd>one-stage detection</kbd>, <kbd>real-time inference</kbd>, <kbd>multi-scale prediction</kbd>

3. **[Core]** [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020  
   Skim for: <kbd>DETR</kbd>, <kbd>object queries</kbd>, <kbd>Hungarian matching</kbd>, <kbd>NMS-free detection</kbd>

4. **[Bridge]** [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159), ICLR 2021  
   Skim for: <kbd>sparse multi-scale attention</kbd>, <kbd>faster DETR convergence</kbd>

5. **[Modern]** [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605), ICLR 2023  
   Skim for: <kbd>denoising training</kbd>, <kbd>anchor design</kbd>, <kbd>strong DETR recipe</kbd>

6. **[Modern]** [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023  
   Skim for: <kbd>text-conditioned detection</kbd>, <kbd>open-set detection</kbd>, <kbd>phrase grounding</kbd>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the detection parts first.

- [DETR paper explanation search](https://www.youtube.com/results?search_query=DETR+paper+explained+object+detection+transformer)  
  Use this if Hungarian matching and object queries are unclear.

## 2. Segmentation and Scene Parsing

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| FCN → DeepLabv3+ | <kbd>dense prediction</kbd> → <kbd>atrous convolution</kbd>, <kbd>encoder-decoder segmentation</kbd> |
| DeepLabv3+ → SegFormer | <kbd>CNN segmentation</kbd> → <kbd>transformer-based scene parsing</kbd> |
| Mask R-CNN → Mask2Former | <kbd>instance masks</kbd> → <kbd>universal segmentation</kbd> |
| Mask2Former → SAM → SAM 2 | <kbd>task-specific segmentation</kbd> → <kbd>promptable image and video segmentation</kbd> |

### Reading Order

1. **[Core]** [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038), CVPR 2015  
   Skim for: <kbd>dense prediction</kbd>, <kbd>fully convolutional networks</kbd>, <kbd>pixel-level classification</kbd>

2. **[Core]** [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611), ECCV 2018  
   Skim for: <kbd>DeepLabv3+</kbd>, <kbd>atrous convolution</kbd>, <kbd>semantic segmentation</kbd>

3. **[Bridge]** [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), NeurIPS 2021  
   Skim for: <kbd>transformer encoder</kbd>, <kbd>lightweight decoder</kbd>, <kbd>scene parsing</kbd>

4. **[Core]** [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017  
   Skim for: <kbd>RoIAlign</kbd>, <kbd>instance segmentation</kbd>, <kbd>mask prediction</kbd>

5. **[Modern]** [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527), CVPR 2022  
   Skim for: <kbd>Mask2Former</kbd>, <kbd>semantic segmentation</kbd>, <kbd>instance segmentation</kbd>, <kbd>panoptic segmentation</kbd>

6. **[Modern]** [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: <kbd>promptable segmentation</kbd>, <kbd>data engine</kbd>, <kbd>ambiguity-aware masks</kbd>

7. **[Modern]** [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <kbd>video segmentation</kbd>, <kbd>memory</kbd>, <kbd>promptable tracking</kbd>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use this for detection vs segmentation task taxonomy.

- [Semantic segmentation lecture search](https://www.youtube.com/results?search_query=semantic+segmentation+computer+vision+lecture+FCN+DeepLab+SegFormer)  
  Use before reading DeepLabv3+ and SegFormer.

## 3. Object Tracking and Multi-Object Tracking

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| SORT → DeepSORT | <kbd>Kalman filtering</kbd>, <kbd>Hungarian matching</kbd> → <kbd>appearance embeddings</kbd> |
| Tracktor++ → CenterTrack | <kbd>tracking-by-detection</kbd> → <kbd>joint detection and tracking</kbd> |
| TrackFormer → ByteTrack | <kbd>transformer queries</kbd> → <kbd>associating low-confidence detections</kbd> |
| SAM 2 | <kbd>box-level MOT</kbd> → <kbd>promptable video object segmentation</kbd> |

### Reading Order

1. **[Core]** [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), ICIP 2016  
   Skim for: <kbd>Kalman filtering</kbd>, <kbd>Hungarian matching</kbd>, <kbd>tracking-by-detection</kbd>

2. **[Core]** [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402), ICIP 2017  
   Skim for: <kbd>DeepSORT</kbd>, <kbd>appearance embeddings</kbd>, <kbd>data association</kbd>

3. **[Bridge]** [Tracktor++: Tracking without Bells and Whistles](https://arxiv.org/abs/1903.05625), ICCV 2019  
   Skim for: <kbd>detector regression</kbd>, <kbd>tracking-by-detection</kbd>

4. **[Bridge]** [CenterTrack: Tracking Objects as Points](https://arxiv.org/abs/2004.01177), ECCV 2020  
   Skim for: <kbd>joint detection and tracking</kbd>, <kbd>object centers</kbd>, <kbd>motion offsets</kbd>

5. **[Modern]** [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702), CVPR 2022  
   Skim for: <kbd>tracking queries</kbd>, <kbd>DETR-style tracking</kbd>

6. **[Modern]** [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), ECCV 2022  
   Skim for: <kbd>low-confidence detections</kbd>, <kbd>association strategy</kbd>

7. **[Advanced]** [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <kbd>promptable video object tracking</kbd>, <kbd>segmentation memory</kbd>

### YouTube Skim Resources

- [Multi-object tracking lecture search](https://www.youtube.com/results?search_query=multi+object+tracking+computer+vision+lecture+SORT+DeepSORT)  
  Start here for Kalman filters and association.

- [ByteTrack paper explanation search](https://www.youtube.com/results?search_query=ByteTrack+paper+explained+multi+object+tracking)  
  Use this after SORT and DeepSORT.

## 4. Depth Estimation and Stereo

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| Monodepth2 → MiDaS | <kbd>self-supervised monocular depth</kbd> → <kbd>relative depth</kbd>, <kbd>multi-dataset training</kbd> |
| PSMNet → RAFT-Stereo | <kbd>cost volume stereo</kbd> → <kbd>iterative stereo matching</kbd> |
| MiDaS → Depth Anything → Depth Anything V2 | <kbd>robust relative depth</kbd> → <kbd>large-scale unlabeled data</kbd>, <kbd>synthetic data</kbd> |

### Reading Order

1. **[Core]** [Unsupervised Learning of Depth and Ego-Motion from Video](https://arxiv.org/abs/1704.07813), CVPR 2017  
   Skim for: <kbd>photometric loss</kbd>, <kbd>ego-motion</kbd>, <kbd>self-supervised depth</kbd>

2. **[Core]** [Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260), ICCV 2019  
   Skim for: <kbd>Monodepth2</kbd>, <kbd>auto-masking</kbd>, <kbd>minimum reprojection loss</kbd>

3. **[Core]** [Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669), CVPR 2018  
   Skim for: <kbd>stereo matching</kbd>, <kbd>cost volume</kbd>, <kbd>spatial pyramid pooling</kbd>

4. **[Bridge]** [RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/abs/2109.07547), 3DV 2021  
   Skim for: <kbd>iterative refinement</kbd>, <kbd>stereo correspondence</kbd>

5. **[Modern]** [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer](https://arxiv.org/abs/1907.01341), TPAMI 2022  
   Skim for: <kbd>MiDaS</kbd>, <kbd>relative depth</kbd>, <kbd>multi-dataset training</kbd>

6. **[Modern]** [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891), CVPR 2024  
   Skim for: <kbd>large-scale unlabeled data</kbd>, <kbd>teacher-student training</kbd>

7. **[Modern]** [Depth Anything V2](https://arxiv.org/abs/2406.09414), NeurIPS 2024  
   Skim for: <kbd>synthetic data</kbd>, <kbd>sharper predictions</kbd>, <kbd>practical model variants</kbd>

### YouTube Skim Resources

- [Monocular depth estimation lecture search](https://www.youtube.com/results?search_query=monocular+depth+estimation+computer+vision+lecture+self-supervised+depth)  
  Use this before Monodepth2 and MiDaS.

- [Stereo matching lecture search](https://www.youtube.com/results?search_query=stereo+matching+depth+estimation+computer+vision+lecture)  
  Use this before PSMNet or RAFT-Stereo.

## 5. Optical Flow

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| Lucas-Kanade / Horn-Schunck → FlowNet | <kbd>variational/classical flow</kbd> → <kbd>learned optical flow</kbd> |
| FlowNet → PWC-Net | <kbd>end-to-end flow</kbd> → <kbd>pyramid</kbd>, <kbd>warping</kbd>, <kbd>cost volume</kbd> |
| PWC-Net → RAFT → GMFlow | <kbd>coarse-to-fine matching</kbd> → <kbd>all-pairs correlation</kbd> → <kbd>global matching</kbd> |

### Reading Order

1. **[Core]** [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_2/lucas_bruce_d_1981_2.pdf), IJCAI 1981  
   Skim for: <kbd>Lucas-Kanade</kbd>, <kbd>local motion</kbd>, <kbd>image alignment</kbd>

2. **[Core]** [Determining Optical Flow](https://people.csail.mit.edu/bkph/papers/Optical_Flow_OPT.pdf), Artificial Intelligence 1981  
   Skim for: <kbd>Horn-Schunck</kbd>, <kbd>brightness constancy</kbd>, <kbd>smoothness prior</kbd>

3. **[Bridge]** [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371), CVPR 2018  
   Skim for: <kbd>pyramid</kbd>, <kbd>warping</kbd>, <kbd>cost volume</kbd>

4. **[Core]** [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039), ECCV 2020  
   Skim for: <kbd>all-pairs correlation volume</kbd>, <kbd>iterative flow refinement</kbd>

5. **[Modern]** [GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680), CVPR 2022  
   Skim for: <kbd>transformer-style global matching</kbd>, <kbd>correspondence</kbd>

### YouTube Skim Resources

- [Optical flow lecture search](https://www.youtube.com/results?search_query=optical+flow+computer+vision+lecture+Lucas+Kanade+Horn+Schunck+RAFT)  
  Watch a classical optical-flow lecture first, then RAFT-specific explanations.

- [RAFT paper explanation search](https://www.youtube.com/results?search_query=RAFT+optical+flow+paper+explained)  
  Use this to understand correlation volumes and recurrent update blocks.

## 6. Multiple View Geometry and 3D Vision

### Field Evolution

| **Papers / Topics** | **Shift** |
|---|---|
| Multiple View Geometry → COLMAP | <kbd>epipolar geometry</kbd>, <kbd>triangulation</kbd>, <kbd>bundle adjustment</kbd> → <kbd>practical SfM pipeline</kbd> |
| COLMAP → DUSt3R | <kbd>feature matching + optimization</kbd> → <kbd>feed-forward pointmap prediction</kbd> |
| DUSt3R → VGGT | <kbd>pairwise geometry foundation model</kbd> → <kbd>generalist prediction of cameras, depth, tracks, and point maps</kbd> |

### Reading Order

1. **[Core]** [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/), Hartley and Zisserman  
   Skim for: <kbd>camera models</kbd>, <kbd>epipolar geometry</kbd>, <kbd>fundamental matrix</kbd>, <kbd>triangulation</kbd>

2. **[Core]** [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf), CVPR 2016  
   Skim for: <kbd>SfM pipeline</kbd>, <kbd>COLMAP</kbd>, <kbd>bundle adjustment</kbd>

3. **[Bridge]** [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763), CVPR 2020  
   Skim for: <kbd>local features</kbd>, <kbd>matching</kbd>, <kbd>attention-based correspondence</kbd>

4. **[Modern]** [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: <kbd>pointmap prediction</kbd>, <kbd>unconstrained image pairs</kbd>, <kbd>reconstruction without calibrated cameras</kbd>

5. **[Advanced]** [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: <kbd>feed-forward prediction</kbd>, <kbd>cameras</kbd>, <kbd>depth maps</kbd>, <kbd>point maps</kbd>, <kbd>tracks</kbd>

### YouTube Skim Resources

- [Multiple view geometry lecture search](https://www.youtube.com/results?search_query=multiple+view+geometry+computer+vision+lecture+camera+pose+epipolar+geometry+triangulation)  
  Use this for camera pose, epipolar geometry, and triangulation.

- [COLMAP and SfM tutorial search](https://www.youtube.com/results?search_query=COLMAP+structure+from+motion+tutorial+bundle+adjustment)  
  Useful for practical SfM intuition.

## 7. 3D Object Detection and BEV Perception

> This section intentionally focuses on camera-centric 3D detection and BEV perception. LiDAR-specific and sensor-fusion methods can be added later as a separate expansion.

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| FCOS3D → DETR3D | <kbd>monocular 3D detection</kbd> → <kbd>query-based 3D detection</kbd> |
| DETR3D → PETR | <kbd>3D-to-2D query projection</kbd> → <kbd>position embedding for 3D detection</kbd> |
| BEVDet → BEVFormer | <kbd>camera-to-BEV lifting</kbd> → <kbd>spatiotemporal BEV representation</kbd> |
| BEVFormer → StreamPETR | <kbd>BEV queries</kbd> → <kbd>online temporal 3D object detection</kbd> |

### Reading Order

1. **[Core]** [FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection](https://arxiv.org/abs/2104.10956), ICCV Workshops 2021  
   Skim for: <kbd>monocular 3D detection</kbd>, <kbd>3D box parameterization</kbd>, <kbd>camera geometry</kbd>

2. **[Core]** [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://arxiv.org/abs/2110.06922), CoRL 2021  
   Skim for: <kbd>multi-view cameras</kbd>, <kbd>3D queries</kbd>, <kbd>3D-to-2D projection</kbd>

3. **[Bridge]** [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625), ECCV 2022  
   Skim for: <kbd>3D position embedding</kbd>, <kbd>multi-view detection</kbd>

4. **[Core]** [BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View](https://arxiv.org/abs/2112.11790), 2021  
   Skim for: <kbd>camera-to-BEV transformation</kbd>, <kbd>BEV feature learning</kbd>

5. **[Core]** [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270), ECCV 2022  
   Skim for: <kbd>BEV queries</kbd>, <kbd>spatial cross-attention</kbd>, <kbd>temporal self-attention</kbd>

6. **[Modern]** [StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2303.11926), ICCV 2023  
   Skim for: <kbd>online 3D detection</kbd>, <kbd>object-centric temporal modeling</kbd>

### YouTube Skim Resources

- [BEV perception autonomous driving lecture search](https://www.youtube.com/results?search_query=BEV+perception+autonomous+driving+lecture+BEVFormer+DETR3D)  
  Use this to understand why BEV is a common representation for driving.

- [Camera-only 3D object detection explanation search](https://www.youtube.com/results?search_query=camera-only+3D+object+detection+DETR3D+BEVFormer+explained)  
  Use this before BEVFormer if 3D queries are unfamiliar.

## 8. SLAM, Odometry, and Occupancy

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| ORB-SLAM3 → DROID-SLAM | <kbd>feature-based SLAM</kbd>, <kbd>loop closure</kbd> → <kbd>learned recurrent updates</kbd>, <kbd>dense bundle adjustment</kbd> |
| DROID-SLAM → NICE-SLAM → Co-SLAM | <kbd>visual odometry / dense BA</kbd> → <kbd>neural implicit dense mapping</kbd>, <kbd>real-time neural SLAM</kbd> |
| Scene as Occupancy → OccFormer / OccTransformer | <kbd>3D scene structure</kbd> → <kbd>occupancy prediction</kbd>, <kbd>camera-only scene representation</kbd> |

### Reading Order

1. **[Core]** [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898), T-RO 2021  
   Skim for: <kbd>feature-based SLAM</kbd>, <kbd>visual-inertial integration</kbd>, <kbd>loop closure</kbd>, <kbd>multi-map reuse</kbd>

2. **[Core]** [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869), NeurIPS 2021  
   Skim for: <kbd>learned recurrent updates</kbd>, <kbd>dense bundle adjustment</kbd>, <kbd>visual odometry</kbd>

3. **[Bridge]** [NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://arxiv.org/abs/2112.12130), CVPR 2022  
   Skim for: <kbd>neural implicit mapping</kbd>, <kbd>dense reconstruction</kbd>, <kbd>tracking</kbd>

4. **[Advanced]** [Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM](https://arxiv.org/abs/2304.14377), CVPR 2023  
   Skim for: <kbd>real-time neural SLAM</kbd>, <kbd>hybrid encodings</kbd>

5. **[Modern]** [Scene as Occupancy](https://arxiv.org/abs/2306.02851), ICCV 2023  
   Skim for: <kbd>camera-only 3D occupancy prediction</kbd>, <kbd>scene completion</kbd>

6. **[Modern]** [OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2304.05316), ICCV 2023  
   Skim for: <kbd>semantic occupancy</kbd>, <kbd>dual-path transformer</kbd>

7. **[Modern]** [OccTransformer: Improving BEVFormer for 3D Camera-Only Occupancy Prediction](https://arxiv.org/abs/2402.18140), 2024  
   Skim for: <kbd>BEV-style occupancy</kbd>, <kbd>camera-only occupancy</kbd>

### YouTube Skim Resources

- [Cyrill Stachniss SLAM Course - Introduction to Robot Mapping](https://www.youtube.com/watch?v=wVsfCnyt5jA)  
  Best starting point for SLAM vocabulary and mapping intuition.

- [Cyrill Stachniss SLAM course playlist search](https://www.youtube.com/results?search_query=Cyrill+Stachniss+SLAM+course+playlist)  
  Use this for graph-based SLAM, least squares, loop closure, and mapping.

- [Occupancy prediction autonomous driving search](https://www.youtube.com/results?search_query=occupancy+prediction+autonomous+driving+BEV+camera-only+lecture)  
  Use this for autonomous-driving occupancy framing.

## 9. Motion Forecasting

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| Social LSTM / Social GAN → VectorNet | <kbd>agent interaction modeling</kbd> → <kbd>vectorized map and trajectory representation</kbd> |
| VectorNet → LaneGCN → TNT | <kbd>polyline representation</kbd> → <kbd>lane graph reasoning</kbd>, <kbd>target-driven prediction</kbd> |
| Wayformer → MTR | <kbd>transformer forecasting</kbd> → <kbd>motion query-based multimodal prediction</kbd> |

### Reading Order

1. **[Bridge]** [Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://arxiv.org/abs/1603.06142), CVPR 2016  
   Skim for: <kbd>trajectory prediction</kbd>, <kbd>social interaction</kbd>, <kbd>sequence modeling</kbd>

2. **[Bridge]** [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/abs/1803.10892), CVPR 2018  
   Skim for: <kbd>multimodal trajectory prediction</kbd>, <kbd>social pooling</kbd>

3. **[Core]** [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259), CVPR 2020  
   Skim for: <kbd>vectorized map</kbd>, <kbd>polyline representation</kbd>, <kbd>agent-map interaction</kbd>

4. **[Core]** [Learning Lane Graph Representations for Motion Forecasting](https://arxiv.org/abs/2007.13732), ECCV 2020  
   Skim for: <kbd>LaneGCN</kbd>, <kbd>lane graph</kbd>, <kbd>agent-lane interaction</kbd>

5. **[Bridge]** [TNT: Target-driven Trajectory Prediction](https://arxiv.org/abs/2008.08294), CoRL 2020  
   Skim for: <kbd>target candidates</kbd>, <kbd>goal-conditioned prediction</kbd>

6. **[Modern]** [Wayformer: Motion Forecasting via Simple & Efficient Attention Networks](https://arxiv.org/abs/2207.05844), ICRA 2023  
   Skim for: <kbd>attention</kbd>, <kbd>map-agent tokens</kbd>, <kbd>trajectory forecasting</kbd>

7. **[Modern]** [Motion Transformer with Global Intention Localization and Local Movement Refinement](https://arxiv.org/abs/2209.13508), NeurIPS 2022  
   Skim for: <kbd>MTR</kbd>, <kbd>motion queries</kbd>, <kbd>multimodal trajectory prediction</kbd>

### YouTube Skim Resources

- [Motion forecasting autonomous driving lecture search](https://www.youtube.com/results?search_query=motion+forecasting+autonomous+driving+trajectory+prediction+lecture+VectorNet+LaneGCN)  
  Use this to understand trajectory prediction and map-conditioned forecasting.

- [Trajectory prediction multimodal forecasting search](https://www.youtube.com/results?search_query=multimodal+trajectory+prediction+autonomous+driving+paper+explained)  
  Use this after VectorNet and LaneGCN.

## 10. Planning and Control

### Field Evolution

| **Topics / Papers** | **Shift** |
|---|---|
| Classical planning → MPC | <kbd>search and trajectory optimization</kbd> → <kbd>receding-horizon control</kbd> |
| Imitation learning → ChauffeurNet | <kbd>behavior cloning</kbd> → <kbd>closed-loop driving behavior modeling</kbd> |
| Rule-based / modular planning → learning-based planning benchmarks | <kbd>hand-engineered planners</kbd> → <kbd>learned planners and closed-loop evaluation</kbd> |

### Reading Order

1. **[Core]** [Planning Algorithms](http://lavalle.pl/planning/), Steven M. LaValle  
   Skim for: <kbd>configuration space</kbd>, <kbd>sampling-based planning</kbd>, <kbd>motion planning</kbd>

2. **[Core]** [Model Predictive Control: Theory, Computation, and Design](https://sites.engineering.ucsb.edu/~jbraw/mpc/), Rawlings, Mayne, and Diehl  
   Skim for: <kbd>MPC</kbd>, <kbd>constraints</kbd>, <kbd>receding horizon</kbd>

3. **[Bridge]** [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), CVPR 2017  
   Skim for: <kbd>imitation learning</kbd>, <kbd>driving policy</kbd>, <kbd>behavior cloning</kbd>

4. **[Bridge]** [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079), 2018  
   Skim for: <kbd>imitation learning</kbd>, <kbd>synthetic perturbations</kbd>, <kbd>closed-loop behavior</kbd>

5. **[Modern]** [nuPlan: A Closed-loop ML-based Planning Benchmark for Autonomous Vehicles](https://arxiv.org/abs/2106.11810), 2021  
   Skim for: <kbd>closed-loop planning</kbd>, <kbd>planner evaluation</kbd>, <kbd>simulation</kbd>

6. **[Modern]** [GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning](https://arxiv.org/abs/2303.05760), ICCV 2023  
   Skim for: <kbd>interactive prediction</kbd>, <kbd>planning</kbd>, <kbd>game-theoretic reasoning</kbd>

### YouTube Skim Resources

- [Motion planning lecture search](https://www.youtube.com/results?search_query=motion+planning+robotics+lecture+RRT+trajectory+optimization)  
  Use this for classical planning vocabulary.

- [Model predictive control lecture search](https://www.youtube.com/results?search_query=model+predictive+control+lecture+robotics+autonomous+driving)  
  Use this before reading learning-based planning papers.

- [Autonomous driving planning lecture search](https://www.youtube.com/results?search_query=autonomous+driving+planning+and+control+lecture)  
  Use this to connect forecasting, planning, and control.

## 11. Robot Learning

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| Behavior cloning / RL background → RT-1 | <kbd>task-specific robot policies</kbd> → <kbd>large-scale robot learning</kbd> |
| RT-1 → RT-2 | <kbd>robot trajectories</kbd> → <kbd>web-scale VLM knowledge transfer</kbd> |
| Diffusion Policy → Open X-Embodiment → OpenVLA | <kbd>continuous action modeling</kbd> → <kbd>cross-embodiment data</kbd>, <kbd>open VLA models</kbd> |

### Reading Order

1. **[Core]** [A Survey of Robot Learning from Demonstration](https://www.sciencedirect.com/science/article/pii/S0921889008001772), Robotics and Autonomous Systems 2009  
   Skim for: <kbd>learning from demonstration</kbd>, <kbd>imitation learning</kbd>, <kbd>policy learning</kbd>

2. **[Bridge]** [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702), JMLR 2016  
   Skim for: <kbd>visuomotor policies</kbd>, <kbd>end-to-end robot learning</kbd>

3. **[Core]** [BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning](https://arxiv.org/abs/2202.02005), CoRL 2021  
   Skim for: <kbd>language-conditioned imitation learning</kbd>, <kbd>zero-shot task generalization</kbd>

4. **[Core]** [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817), RSS 2023  
   Skim for: <kbd>large-scale robot data</kbd>, <kbd>robotics transformer</kbd>, <kbd>language-conditioned control</kbd>

5. **[Core]** [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023 / IJRR 2024  
   Skim for: <kbd>action diffusion</kbd>, <kbd>receding-horizon control</kbd>, <kbd>multimodal action distributions</kbd>

6. **[Modern]** [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818), CoRL 2023  
   Skim for: <kbd>VLA model</kbd>, <kbd>robot actions as tokens</kbd>, <kbd>web knowledge transfer</kbd>

7. **[Modern]** [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864), 2023  
   Skim for: <kbd>cross-embodiment learning</kbd>, <kbd>robot datasets</kbd>, <kbd>RT-X</kbd>

8. **[Modern]** [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246), CoRL 2024  
   Skim for: <kbd>open VLA training</kbd>, <kbd>parameter-efficient adaptation</kbd>, <kbd>robot action generation</kbd>

### YouTube Skim Resources

- [Robot learning lecture search](https://www.youtube.com/results?search_query=robot+learning+lecture+imitation+learning+behavior+cloning+visuomotor+policy)  
  Use this for imitation learning and visuomotor policy basics.

- [Diffusion Policy paper explanation search](https://www.youtube.com/results?search_query=Diffusion+Policy+visuomotor+policy+learning+paper+explained)  
  Focus on action diffusion and receding-horizon execution.

- [RT-1 RT-2 OpenVLA explanation search](https://www.youtube.com/results?search_query=RT-1+RT-2+OpenVLA+vision+language+action+model+explained)  
  Use this for modern VLA-style robot learning.

## 12. Foundation Models for Vision and Robotics

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| ViT → CLIP | <kbd>transformer vision backbone</kbd> → <kbd>image-text contrastive learning</kbd> |
| CLIP → MAE → DINOv2 | <kbd>zero-shot transfer</kbd> → <kbd>masked image modeling</kbd>, <kbd>self-supervised dense features</kbd> |
| DINOv2 → SAM / SAM 2 | <kbd>foundation visual features</kbd> → <kbd>promptable image/video segmentation</kbd> |
| CLIP-style VLMs → VLA models | <kbd>vision-language alignment</kbd> → <kbd>robot actions and embodied policies</kbd> |

### Reading Order

1. **[Core]** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021  
   Skim for: <kbd>ViT</kbd>, <kbd>patch tokens</kbd>, <kbd>scale</kbd>

2. **[Core]** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021  
   Skim for: <kbd>CLIP</kbd>, <kbd>image-text contrastive learning</kbd>, <kbd>zero-shot transfer</kbd>

3. **[Core]** [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022  
   Skim for: <kbd>masked image modeling</kbd>, <kbd>asymmetric encoder-decoder</kbd>

4. **[Core]** [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), TMLR 2024  
   Skim for: <kbd>self-supervised features</kbd>, <kbd>curated data</kbd>, <kbd>dense transfer</kbd>

5. **[Bridge]** [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), ICCV 2023  
   Skim for: <kbd>SigLIP</kbd>, <kbd>scalable image-text loss</kbd>

6. **[Modern]** [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: <kbd>promptable segmentation</kbd>, <kbd>foundation task</kbd>

7. **[Modern]** [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <kbd>video foundation segmentation</kbd>, <kbd>memory</kbd>

8. **[Modern]** [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217), 2024  
   Skim for: <kbd>video representation learning</kbd>, <kbd>video foundation encoder</kbd>

9. **[Advanced]** [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238), CVPR 2024  
   Skim for: <kbd>VLM alignment</kbd>, <kbd>vision encoder scaling</kbd>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 8: Attention and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ)  
  Best first video before ViT, MAE, DINOv2, and SAM.

- [CLIP paper explanation search](https://www.youtube.com/results?search_query=CLIP+paper+explained+contrastive+language+image+pretraining)  
  Focus on contrastive loss and zero-shot evaluation.

- [DINOv2 paper explanation search](https://www.youtube.com/results?search_query=DINOv2+paper+explained+computer+vision+foundation+model)  
  Use this after CLIP and MAE.

## 13. End-to-End Autonomous Driving

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| End-to-end driving → ChauffeurNet | <kbd>behavior cloning</kbd> → <kbd>synthetic perturbation and closed-loop behavior</kbd> |
| ChauffeurNet → TransFuser | <kbd>camera-based imitation</kbd> → <kbd>multi-view perception-to-control learning</kbd> |
| TransFuser → UniAD | <kbd>driving policy learning</kbd> → <kbd>planning-oriented unified perception, prediction, mapping, and planning</kbd> |
| UniAD → VAD / VLA-style driving | <kbd>modular unified stack</kbd> → <kbd>vectorized scene representation</kbd>, <kbd>vision-language action framing</kbd> |

### Reading Order

1. **[Core]** [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), CVPR 2017  
   Skim for: <kbd>large-scale imitation learning</kbd>, <kbd>driving</kbd>, <kbd>behavior cloning</kbd>

2. **[Bridge]** [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079), 2018  
   Skim for: <kbd>closed-loop driving behavior</kbd>, <kbd>synthetic data</kbd>, <kbd>imitation learning</kbd>

3. **[Bridge]** [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving](https://arxiv.org/abs/2205.15997), 2022  
   Skim for: <kbd>end-to-end driving</kbd>, <kbd>transformer fusion</kbd>, <kbd>imitation learning</kbd>

4. **[Core]** [Planning-Oriented Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html), CVPR 2023  
   Skim for: <kbd>UniAD</kbd>, <kbd>perception</kbd>, <kbd>prediction</kbd>, <kbd>mapping</kbd>, <kbd>planning</kbd>

5. **[Modern]** [VAD: Vectorized Scene Representation for Efficient Autonomous Driving](https://arxiv.org/abs/2303.12077), ICCV 2023  
   Skim for: <kbd>vectorized scene representation</kbd>, <kbd>planning-oriented driving</kbd>

6. **[Advanced]** [OpenDriveVLA: Towards End-to-End Autonomous Driving with Large Vision Language Action Model](https://arxiv.org/abs/2503.23463), 2025  
   Skim for: <kbd>VLA framing</kbd>, <kbd>autonomous driving</kbd>, <kbd>action generation</kbd>

7. **[Advanced]** [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/abs/2503.19755), 2025  
   Skim for: <kbd>language-instructed driving</kbd>, <kbd>action generation</kbd>

### YouTube Skim Resources

- [UniAD paper explanation search](https://www.youtube.com/results?search_query=UniAD+Planning-Oriented+Autonomous+Driving+paper+explained)  
  Use this to understand why planning is used as the organizing objective.

- [End-to-end autonomous driving lecture search](https://www.youtube.com/results?search_query=end-to-end+autonomous+driving+imitation+learning+lecture+UniAD+VAD)  
  Use this before frontier VLA-style driving papers.

## 14. Generative Vision and World Models

### Field Evolution

| **Papers** | **Shift** |
|---|---|
| DDPM → Latent Diffusion | <kbd>diffusion objective</kbd>, <kbd>denoising process</kbd> → <kbd>latent diffusion</kbd> |
| Latent Diffusion → Video Diffusion / SVD | <kbd>image generation</kbd> → <kbd>video generation</kbd>, <kbd>image-to-video</kbd> |
| DreamerV3 → Genie | <kbd>latent imagination for control</kbd> → <kbd>action-controllable generative environments</kbd> |
| GAIA-1 → DrivingWorld / GAIA-2 | <kbd>driving video generation</kbd> → <kbd>controllable driving world models</kbd> |

### Reading Order

1. **[Core]** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020  
   Skim for: <kbd>diffusion objective</kbd>, <kbd>denoising process</kbd>

2. **[Core]** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022  
   Skim for: <kbd>latent diffusion</kbd>, <kbd>cross-attention conditioning</kbd>

3. **[Bridge]** [Video Diffusion Models](https://arxiv.org/abs/2204.03458), NeurIPS 2022  
   Skim for: <kbd>video diffusion</kbd>, <kbd>temporal generation</kbd>

4. **[Bridge]** [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127), 2023  
   Skim for: <kbd>image-to-video</kbd>, <kbd>latent video diffusion</kbd>

5. **[Core]** [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), 2023  
   Skim for: <kbd>DreamerV3</kbd>, <kbd>latent imagination</kbd>, <kbd>control</kbd>

6. **[Modern]** [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391), ICML 2024  
   Skim for: <kbd>action-controllable environments</kbd>, <kbd>unlabeled videos</kbd>

7. **[Modern]** [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080), 2023  
   Skim for: <kbd>generative driving simulation</kbd>, <kbd>driving videos</kbd>

8. **[Advanced]** [DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT](https://arxiv.org/abs/2412.19505), 2024  
   Skim for: <kbd>video-token world modeling</kbd>, <kbd>driving</kbd>

9. **[Advanced]** [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523), 2025  
   Skim for: <kbd>controllable multi-view generation</kbd>, <kbd>driving world generation</kbd>

### YouTube Skim Resources

- [DDPM basics video](https://www.youtube.com/watch?v=h6T1UU9pzOY)  
  Good first skim for diffusion mechanics.

- [Genie world model explanation search](https://www.youtube.com/results?search_query=Genie+Generative+Interactive+Environments+world+model+explained)  
  Use this for interactive world-model intuition.

- [GAIA driving world model explanation search](https://www.youtube.com/results?search_query=GAIA+generative+world+model+autonomous+driving+explained)  
  Use this for driving-specific world-model framing.

## 15. Robustness, Uncertainty, and Safety

### Field Evolution

| **Papers / Topics** | **Shift** |
|---|---|
| Adversarial examples → PGD adversarial training | <kbd>attack setup</kbd> → <kbd>robust optimization</kbd> |
| Domain adaptation → Test-time adaptation | <kbd>domain-invariant training</kbd> → <kbd>online adaptation</kbd> |
| Deep ensembles / Bayesian uncertainty → Safety-aware perception | <kbd>uncertainty estimation</kbd> → <kbd>calibration</kbd>, <kbd>OOD awareness</kbd>, <kbd>risk-sensitive systems</kbd> |
| WILDS / real-world shifts | <kbd>clean benchmarks</kbd> → <kbd>distribution shift</kbd>, <kbd>robust evaluation</kbd> |

### Reading Order

1. **[Bridge]** [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), ICLR 2015  
   Skim for: <kbd>FGSM</kbd>, <kbd>adversarial examples</kbd>

2. **[Bridge]** [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), ICLR 2018  
   Skim for: <kbd>PGD adversarial training</kbd>, <kbd>robust optimization</kbd>

3. **[Core]** [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818), JMLR 2016  
   Skim for: <kbd>gradient reversal</kbd>, <kbd>domain-invariant representation</kbd>

4. **[Core]** [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles), NeurIPS 2017  
   Skim for: <kbd>ensemble uncertainty</kbd>, <kbd>calibration</kbd>, <kbd>OOD behavior</kbd>

5. **[Core]** [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977), NeurIPS 2017  
   Skim for: <kbd>epistemic uncertainty</kbd>, <kbd>aleatoric uncertainty</kbd>, <kbd>dense prediction</kbd>

6. **[Modern]** [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726), ICLR 2021  
   Skim for: <kbd>test-time adaptation</kbd>, <kbd>entropy minimization</kbd>, <kbd>batch-norm adaptation</kbd>

7. **[Modern]** [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), ICML 2021  
   Skim for: <kbd>realistic domain shift</kbd>, <kbd>benchmark design</kbd>

8. **[Advanced]** [RobustBench: a standardized adversarial robustness benchmark](https://arxiv.org/abs/2010.09670), NeurIPS 2021  
   Skim for: <kbd>standardized robustness evaluation</kbd>, <kbd>attack pitfalls</kbd>

### YouTube Skim Resources

- [Domain adaptation lecture search](https://www.youtube.com/results?search_query=domain+adaptation+computer+vision+lecture+test+time+adaptation)  
  Use this before DANN and Tent.

- [Uncertainty in deep learning lecture search](https://www.youtube.com/results?search_query=uncertainty+in+deep+learning+deep+ensembles+lecture)  
  Use before Deep Ensembles and Bayesian uncertainty for vision.

- [Safety autonomous driving perception uncertainty search](https://www.youtube.com/results?search_query=autonomous+driving+perception+uncertainty+safety+OOD+lecture)  
  Use this to connect uncertainty with safety-critical systems.

