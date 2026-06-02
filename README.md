# Awesome Multimodal Perception

A reading roadmap for skimming essential computer vision papers across modern perception, geometry, embodied AI, generative vision, and robustness.

Last updated: 2026-06-02

## Table of Contents

- [1. Object Detection](#1-object-detection)
- [2. Segmentation](#2-segmentation)
- [3. Object Tracking](#3-object-tracking)
- [4. Depth, Geometry, and Optical Flow](#4-depth-geometry-and-optical-flow)
- [5. SLAM, Odometry, and Occupancy](#5-slam-odometry-and-occupancy)
- [6. 3D Reconstruction](#6-3d-reconstruction)
- [7. Foundation Models](#7-foundation-models)
- [8. End-to-End Autonomous Driving and Robotics Models](#8-end-to-end-autonomous-driving-and-robotics-models)
- [9. Generative Vision and World Models](#9-generative-vision-and-world-models)
- [10. Robustness, Uncertainty, and Domain Adaptation](#10-robustness-uncertainty-and-domain-adaptation)

## 1. Object Detection

### Field Evolution

| Papers | Shift |
|---|---|
| **Faster R-CNN → YOLOv3** | <kbd>two-stage detection</kbd> → <kbd>one-stage real-time detection</kbd> |
| **YOLOv3 → DETR** | <kbd>multi-scale prediction</kbd> → <kbd>object queries</kbd>, <kbd>Hungarian matching</kbd> |
| **DETR → Deformable DETR → DINO** | <kbd>NMS-free detection</kbd> → <kbd>sparse multi-scale attention</kbd>, <kbd>denoising training</kbd> |
| **GLIP → Grounding DINO** | <kbd>phrase grounding</kbd> → <kbd>text-conditioned open-set detection</kbd> |

### Reading Order

1. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS 2015  
   Skim for: <kbd>region proposal networks</kbd>, <kbd>two-stage detection</kbd>, <kbd>anchors</kbd>, <kbd>RoI heads</kbd>

2. [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767), 2018  
   Skim for: <kbd>one-stage real-time detection</kbd>, <kbd>multi-scale prediction</kbd>, <kbd>speed vs accuracy tradeoffs</kbd>

3. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020  
   Skim for: <kbd>DETR</kbd>, <kbd>object queries</kbd>, <kbd>Hungarian matching</kbd>, <kbd>NMS-free detection</kbd>

4. [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159), ICLR 2021  
   Skim for: <kbd>sparse multi-scale attention</kbd>, <kbd>faster DETR convergence</kbd>

5. [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605), ICLR 2023  
   Skim for: <kbd>denoising training</kbd>, <kbd>anchor design</kbd>, <kbd>strong DETR training recipes</kbd>

6. [Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857), CVPR 2022  
   Skim for: <kbd>phrase grounding</kbd>

7. [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023  
   Skim for: <kbd>text-conditioned open-set detection</kbd>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the detection parts first. It covers detection task setup, R-CNN-style models, and modern detector families.

- [Stanford CS231n 2017 Lecture 11: Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo)  
  Older but useful for R-CNN and Faster R-CNN foundations.

- [DETR paper explanation search](https://www.youtube.com/results?search_query=DETR+paper+explained+object+detection+transformer)  
  Use this if Hungarian matching and object queries are unclear after the paper.

## 2. Segmentation

### Field Evolution

| Papers | Shift |
|---|---|
| **Mask R-CNN → Mask2Former** | <kbd>mask prediction</kbd> → <kbd>universal segmentation</kbd> |
| **Mask2Former → Segment Anything** | <kbd>semantic segmentation</kbd>, <kbd>instance segmentation</kbd>, <kbd>panoptic segmentation</kbd> → <kbd>promptable segmentation</kbd> |
| **Segment Anything → SAM 2** | <kbd>SA-1B</kbd>, <kbd>ambiguity-aware mask prediction</kbd> → <kbd>memory-based segmentation</kbd>, <kbd>video segmentation</kbd> |

### Reading Order

1. [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017  
   Skim for: <kbd>RoIAlign</kbd>, <kbd>mask prediction</kbd>, <kbd>instance detection</kbd>

2. [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527), CVPR 2022  
   Skim for: <kbd>Mask2Former</kbd>, <kbd>universal segmentation</kbd>, <kbd>semantic segmentation</kbd>, <kbd>instance segmentation</kbd>, <kbd>panoptic segmentation</kbd>

3. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: <kbd>promptable segmentation</kbd>, <kbd>SA-1B</kbd>, <kbd>ambiguity-aware mask prediction</kbd>

4. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <kbd>memory-based segmentation</kbd>, <kbd>video segmentation</kbd>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the segmentation parts for task taxonomy, dense prediction, and mask-based outputs.

- [Stanford CS231n 2017 Lecture 11: Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo)  
  Useful for Mask R-CNN foundations.

- [Segment Anything paper explanation search](https://www.youtube.com/results?search_query=Segment+Anything+paper+explained)  
  Use this after reading SAM. Prefer explanations that show the prompt encoder, mask decoder, and data engine.

## 3. Object Tracking

### Field Evolution

| Papers | Shift |
|---|---|
| **Simple Online and Realtime Tracking → Deep Association Metric** | <kbd>Kalman filtering</kbd>, <kbd>Hungarian matching</kbd> → <kbd>appearance embeddings</kbd> |
| **Deep Association Metric → Tracktor++ → CenterTrack** | <kbd>robust association</kbd> → <kbd>detector regression</kbd>, <kbd>joint detection and tracking</kbd> |
| **CenterTrack → TrackFormer → ByteTrack** | <kbd>object centers</kbd> → <kbd>transformer queries</kbd>, <kbd>low-confidence detection boxes</kbd> |
| **ByteTrack → SAM 2** | <kbd>data association</kbd> → <kbd>promptable video object tracking</kbd>, <kbd>segmentation memory</kbd> |

### Reading Order

1. [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), ICIP 2016  
   Skim for: <kbd>Kalman filtering</kbd>, <kbd>Hungarian matching</kbd>, <kbd>tracking-by-detection</kbd>

2. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402), ICIP 2017  
   Skim for: <kbd>appearance embeddings</kbd>, <kbd>robust association</kbd>

3. [Tracktor++: Tracking without Bells and Whistles](https://arxiv.org/abs/1903.05625), ICCV 2019  
   Skim for: <kbd>detector regression</kbd>, <kbd>tracking</kbd>

4. [CenterTrack: Tracking Objects as Points](https://arxiv.org/abs/2004.01177), ECCV 2020  
   Skim for: <kbd>joint detection and tracking</kbd>, <kbd>object centers</kbd>

5. [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702), CVPR 2022  
   Skim for: <kbd>transformer queries</kbd>, <kbd>detection-to-tracking</kbd>

6. [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), ECCV 2022  
   Skim for: <kbd>low-confidence detection boxes</kbd>, <kbd>data association</kbd>

7. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <kbd>promptable video object tracking</kbd>, <kbd>segmentation memory</kbd>

### YouTube Skim Resources

- [Multi-object tracking lecture search](https://www.youtube.com/results?search_query=multi+object+tracking+computer+vision+lecture+SORT+DeepSORT)  
  Start here for SORT, DeepSORT, Kalman filters, and data association.

- [ByteTrack paper explanation search](https://www.youtube.com/results?search_query=ByteTrack+paper+explained+multi+object+tracking)  
  Use this after SORT and DeepSORT.

- [TrackFormer paper explanation search](https://www.youtube.com/results?search_query=TrackFormer+multi+object+tracking+transformer+paper+explained)  
  Use this to connect DETR-style queries with tracking.

## 4. Depth, Geometry, and Optical Flow

### Field Evolution

| Papers | Shift |
|---|---|
| **RAFT → GMFlow** | <kbd>all-pairs correlation volume</kbd>, <kbd>iterative flow refinement</kbd> → <kbd>transformer-style global matching</kbd> |
| **MiDaS → Depth Anything → Depth Anything V2** | <kbd>relative depth</kbd>, <kbd>multi-dataset training</kbd> → <kbd>large-scale unlabeled data</kbd>, <kbd>synthetic data</kbd> |
| **DUSt3R → VGGT** | <kbd>pointmap prediction</kbd> → <kbd>feed-forward prediction</kbd> of <kbd>cameras</kbd>, <kbd>depth maps</kbd>, <kbd>point maps</kbd>, <kbd>tracks</kbd> |
| **VGGT → DEFOM-Stereo** | <kbd>depth maps</kbd>, <kbd>point maps</kbd> → <kbd>monocular depth foundation priors</kbd> for <kbd>stereo matching</kbd> |

### Reading Order

1. [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039), ECCV 2020  
   Skim for: <kbd>all-pairs correlation volume</kbd>, <kbd>iterative flow refinement</kbd>

2. [GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680), CVPR 2022  
   Skim for: <kbd>transformer-style global matching</kbd>, <kbd>optical flow</kbd>

3. [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer](https://arxiv.org/abs/1907.01341), TPAMI 2022  
   Skim for: <kbd>MiDaS</kbd>, <kbd>relative depth</kbd>, <kbd>multi-dataset training</kbd>

4. [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891), CVPR 2024  
   Skim for: <kbd>large-scale unlabeled data</kbd>, <kbd>teacher-student depth training</kbd>

5. [Depth Anything V2](https://arxiv.org/abs/2406.09414), NeurIPS 2024  
   Skim for: <kbd>synthetic data</kbd>, <kbd>sharper predictions</kbd>, <kbd>practical model variants</kbd>

6. [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: <kbd>pointmap prediction</kbd>, <kbd>unconstrained image pairs</kbd>

7. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: <kbd>feed-forward prediction</kbd>, <kbd>cameras</kbd>, <kbd>depth maps</kbd>, <kbd>point maps</kbd>, <kbd>tracks</kbd>

8. [DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_DEFOM-Stereo_Depth_Foundation_Model_Based_Stereo_Matching_CVPR_2025_paper.pdf), CVPR 2025  
   Skim for: <kbd>monocular depth foundation priors</kbd>, <kbd>stereo matching</kbd>

### YouTube Skim Resources

- [Optical flow lecture search](https://www.youtube.com/results?search_query=optical+flow+computer+vision+lecture+RAFT)  
  Watch a classical optical-flow lecture first, then RAFT-specific explanations.

- [RAFT paper explanation search](https://www.youtube.com/results?search_query=RAFT+optical+flow+paper+explained)  
  Use this to understand correlation volumes and recurrent update blocks.

- [Monocular depth estimation lecture search](https://www.youtube.com/results?search_query=monocular+depth+estimation+computer+vision+lecture)  
  Use this before MiDaS and Depth Anything.

- [DUSt3R and VGGT explanation search](https://www.youtube.com/results?search_query=DUSt3R+VGGT+visual+geometry+grounded+transformer+explained)  
  These papers are newer, so search results are more useful than relying on one canonical lecture.

## 5. SLAM, Odometry, and Occupancy

### Field Evolution

| Papers | Shift |
|---|---|
| **ORB-SLAM3 → DROID-SLAM** | <kbd>feature-based SLAM</kbd>, <kbd>loop closure</kbd> → <kbd>learned recurrent updates</kbd>, <kbd>dense bundle adjustment</kbd> |
| **DROID-SLAM → NICE-SLAM → Co-SLAM** | <kbd>dense bundle adjustment</kbd> → <kbd>neural implicit dense mapping</kbd>, <kbd>real-time neural SLAM</kbd> |
| **Scene as Occupancy → OccTransformer** | <kbd>camera-only 3D occupancy prediction</kbd> → <kbd>BEV-style perception</kbd>, <kbd>occupancy</kbd> |
| **OccTransformer → LiDAR-based 4D Occupancy Completion and Forecasting** | <kbd>occupancy</kbd> → <kbd>occupancy completion</kbd>, <kbd>future occupancy forecasting</kbd> |

### Reading Order

1. [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898), T-RO 2021  
   Skim for: <kbd>feature-based SLAM</kbd>, <kbd>visual-inertial integration</kbd>, <kbd>loop closure</kbd>, <kbd>multi-map reuse</kbd>

2. [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869), NeurIPS 2021  
   Skim for: <kbd>learned recurrent updates</kbd>, <kbd>dense bundle adjustment</kbd>

3. [NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://arxiv.org/abs/2112.12130), CVPR 2022  
   Skim for: <kbd>neural implicit dense mapping</kbd>, <kbd>tracking</kbd>

4. [Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM](https://arxiv.org/abs/2304.14377), CVPR 2023  
   Skim for: <kbd>real-time neural SLAM</kbd>, <kbd>hybrid encodings</kbd>

5. [Scene as Occupancy](https://arxiv.org/abs/2306.02851), ICCV 2023  
   Skim for: <kbd>camera-only 3D occupancy prediction</kbd>

6. [OccTransformer: Improving BEVFormer for 3D Camera-Only Occupancy Prediction](https://arxiv.org/abs/2402.18140), 2024  
   Skim for: <kbd>BEV-style perception</kbd>, <kbd>occupancy</kbd>

7. [LiDAR-based 4D Occupancy Completion and Forecasting](https://arxiv.org/abs/2310.11239), 2023  
   Skim for: <kbd>occupancy completion</kbd>, <kbd>future occupancy forecasting</kbd>

### YouTube Skim Resources

- [Cyrill Stachniss SLAM Course - Introduction to Robot Mapping](https://www.youtube.com/watch?v=wVsfCnyt5jA)  
  Best starting point for SLAM vocabulary, probabilistic mapping, and pose graph intuition.

- [Cyrill Stachniss SLAM course playlist search](https://www.youtube.com/results?search_query=Cyrill+Stachniss+SLAM+course+playlist)  
  Use this for graph-based SLAM, least squares, loop closure, and mapping.

- [ORB-SLAM and visual SLAM lecture search](https://www.youtube.com/results?search_query=ORB-SLAM3+visual+SLAM+lecture)  
  Use this after ORB-SLAM3 if you want implementation-level intuition.

- [NVIDIA DRIVE Labs occupancy prediction video search](https://www.youtube.com/results?search_query=NVIDIA+DRIVE+Labs+occupancy+prediction+autonomous+vehicle)  
  Useful for the autonomous-driving occupancy framing, even though it is less paper-specific.

## 6. 3D Reconstruction

### Field Evolution

| Papers | Shift |
|---|---|
| **Structure-from-Motion Revisited → NeRF** | <kbd>classical SfM pipeline</kbd>, <kbd>COLMAP</kbd> → <kbd>volumetric rendering</kbd> |
| **NeRF → Instant-NGP** | <kbd>positional encoding</kbd>, <kbd>view synthesis</kbd> → <kbd>hash grid encoding</kbd>, <kbd>fast neural scene fitting</kbd> |
| **Instant-NGP → 3D Gaussian Splatting** | <kbd>fast neural scene fitting</kbd> → <kbd>explicit anisotropic Gaussians</kbd>, <kbd>differentiable rasterization</kbd> |
| **3D Gaussian Splatting → pixelSplat → DUSt3R → VGGT** | <kbd>feed-forward Gaussian splat prediction</kbd> → <kbd>pointmap prediction</kbd>, <kbd>generalist 3D geometry</kbd> |

### Reading Order

1. [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf), CVPR 2016  
   Skim for: <kbd>classical SfM pipeline</kbd>, <kbd>COLMAP</kbd>

2. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934), ECCV 2020  
   Skim for: <kbd>volumetric rendering</kbd>, <kbd>positional encoding</kbd>, <kbd>view synthesis</kbd>

3. [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989), SIGGRAPH 2022  
   Skim for: <kbd>hash grid encoding</kbd>, <kbd>fast neural scene fitting</kbd>

4. [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), SIGGRAPH 2023  
   Skim for: <kbd>explicit anisotropic Gaussians</kbd>, <kbd>differentiable rasterization</kbd>

5. [pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction](https://arxiv.org/abs/2312.12337), CVPR 2024  
   Skim for: <kbd>feed-forward Gaussian splat prediction</kbd>

6. [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: <kbd>reconstruction without calibrated cameras</kbd>

7. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: <kbd>generalist 3D geometry</kbd>, <kbd>foundation model</kbd>

### YouTube Skim Resources

- [NeRF original project video](https://www.youtube.com/results?search_query=NeRF+representing+scenes+as+neural+radiance+fields+explained)  
  Watch a short NeRF explainer before reading Instant-NGP or 3D Gaussian Splatting.

- [3D Gaussian Splatting paper explanation search](https://www.youtube.com/results?search_query=3D+Gaussian+Splatting+paper+explained)  
  Focus on the difference between NeRF ray marching and Gaussian rasterization.

- [COLMAP and SfM tutorial search](https://www.youtube.com/results?search_query=COLMAP+structure+from+motion+tutorial+bundle+adjustment)  
  Useful for the classical reconstruction pipeline.

- [VGGT 3D reconstruction explanation search](https://www.youtube.com/results?search_query=VGGT+Visual+Geometry+Grounded+Transformer+3D+reconstruction+explained)  
  Use this after DUSt3R and 3D Gaussian Splatting.

## 7. Foundation Models

### Field Evolution

| Papers | Shift |
|---|---|
| **ViT → CLIP** | <kbd>ViT patch tokens</kbd>, <kbd>scale</kbd> → <kbd>image-text contrastive learning</kbd> |
| **CLIP → MAE → DINOv2** | <kbd>zero-shot transfer</kbd> → <kbd>masked image modeling</kbd>, <kbd>self-supervised features</kbd> |
| **DINOv2 → SigLIP → SAM** | <kbd>dense transfer</kbd>, <kbd>scalable image-text loss design</kbd> → <kbd>promptable segmentation</kbd> |
| **SAM → SAM 2 → VideoPrism → InternVL** | <kbd>image-video segmentation</kbd> → <kbd>video foundation representation learning</kbd>, <kbd>VLM alignment</kbd> |

### Reading Order

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021  
   Skim for: <kbd>ViT patch tokens</kbd>, <kbd>scale</kbd>, <kbd>inductive-bias tradeoffs</kbd>

2. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021  
   Skim for: <kbd>CLIP</kbd>, <kbd>image-text contrastive learning</kbd>, <kbd>zero-shot transfer</kbd>

3. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022  
   Skim for: <kbd>masked image modeling</kbd>, <kbd>asymmetric encoder-decoder design</kbd>

4. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), TMLR 2024  
   Skim for: <kbd>self-supervised features</kbd>, <kbd>curated data</kbd>, <kbd>dense transfer</kbd>

5. [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), ICCV 2023  
   Skim for: <kbd>SigLIP</kbd>, <kbd>scalable image-text loss design</kbd>

6. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: <kbd>promptable segmentation</kbd>, <kbd>foundation task</kbd>

7. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <kbd>image-video segmentation</kbd>, <kbd>memory</kbd>

8. [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217), 2024  
   Skim for: <kbd>video foundation representation learning</kbd>

9. [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238), CVPR 2024  
   Skim for: <kbd>vision encoder scaling</kbd>, <kbd>VLM alignment</kbd>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 8: Attention and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ)  
  Best first video before ViT, MAE, DINOv2, and SAM.

- [Vision Transformer quick guide](https://www.youtube.com/results?search_query=Vision+Transformer+quick+guide+paper+explained)  
  Use this if patch embeddings and class tokens are unclear.

- [CLIP paper explanation search](https://www.youtube.com/results?search_query=CLIP+paper+explained+contrastive+language+image+pretraining)  
  Focus on contrastive loss, prompt engineering, and zero-shot evaluation.

- [DINOv2 paper explanation search](https://www.youtube.com/results?search_query=DINOv2+paper+explained+computer+vision+foundation+model)  
  Use this after CLIP and MAE to understand self-supervised dense features.

- [SAM and SAM 2 explanation search](https://www.youtube.com/results?search_query=SAM+2+Segment+Anything+paper+explained)  
  Use this after reading SAM to connect promptable segmentation with video memory.

## 8. End-to-End Autonomous Driving and Robotics Models

### Field Evolution

| Papers | Shift |
|---|---|
| **End-to-end Learning of Driving Models → UniAD** | <kbd>large-scale imitation learning</kbd> → <kbd>perception</kbd>, <kbd>prediction</kbd>, <kbd>mapping</kbd>, <kbd>planning</kbd> |
| **UniAD → Diffusion Policy** | <kbd>planning</kbd> → <kbd>action diffusion</kbd>, <kbd>receding-horizon control</kbd>, <kbd>multimodal action distributions</kbd> |
| **Diffusion Policy → RT-2 → OpenVLA** | <kbd>robot actions</kbd> as <kbd>language-model tokens</kbd> → <kbd>open VLA training</kbd> |
| **OpenVLA → OpenDriveVLA → ORION** | <kbd>VLA framing</kbd> → <kbd>autonomous driving</kbd>, <kbd>language-instructed driving action generation</kbd> |

### Reading Order

1. [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), CVPR 2017  
   Skim for: <kbd>large-scale imitation learning</kbd>, <kbd>driving</kbd>

2. [Planning-Oriented Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html), CVPR 2023  
   Skim for: <kbd>UniAD</kbd>, <kbd>perception</kbd>, <kbd>prediction</kbd>, <kbd>mapping</kbd>, <kbd>planning</kbd>

3. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023 / IJRR 2024  
   Skim for: <kbd>action diffusion</kbd>, <kbd>receding-horizon control</kbd>, <kbd>multimodal action distributions</kbd>

4. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818), CoRL 2023  
   Skim for: <kbd>robot actions</kbd>, <kbd>language-model tokens</kbd>

5. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246), CoRL 2024  
   Skim for: <kbd>open VLA training</kbd>, <kbd>Open X-Embodiment data</kbd>, <kbd>parameter-efficient adaptation</kbd>

6. [OpenDriveVLA: Towards End-to-End Autonomous Driving with Large Vision Language Action Model](https://arxiv.org/abs/2503.23463), 2025  
   Skim for: <kbd>VLA framing</kbd>, <kbd>autonomous driving</kbd>

7. [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/abs/2503.19755), 2025  
   Skim for: <kbd>language-instructed driving action generation</kbd>

### YouTube Skim Resources

- [UniAD paper explanation search](https://www.youtube.com/results?search_query=UniAD+Planning-Oriented+Autonomous+Driving+paper+explained)  
  Use this to understand why planning is used as the organizing objective.

- [Diffusion Policy paper explanation search](https://www.youtube.com/results?search_query=Diffusion+Policy+visuomotor+policy+learning+paper+explained)  
  Focus on action diffusion and receding horizon control.

- [RT-2 video and paper explanation search](https://www.youtube.com/results?search_query=RT-2+vision+language+action+model+paper+explained)  
  Use this before OpenVLA.

- [OpenVLA explanation search](https://www.youtube.com/results?search_query=OpenVLA+vision+language+action+model+paper+explained)  
  Use this for the open-source VLA recipe and fine-tuning details.

## 9. Generative Vision and World Models

### Field Evolution

| Papers | Shift |
|---|---|
| **DDPM → Latent Diffusion** | <kbd>diffusion objective</kbd>, <kbd>denoising process</kbd> → <kbd>latent diffusion</kbd> |
| **Latent Diffusion → Video Diffusion → Stable Video Diffusion** | <kbd>cross-attention conditioning</kbd> → <kbd>video diffusion</kbd>, <kbd>image-to-video</kbd>, <kbd>text-to-video scaling</kbd> |
| **DreamerV3 → Genie** | <kbd>latent imagination</kbd>, <kbd>control</kbd> → <kbd>action-controllable environments</kbd> |
| **Genie → GAIA-1 → GAIA-2 → DrivingWorld** | <kbd>generative driving simulation</kbd> → <kbd>controllable multi-view generation</kbd>, <kbd>video-token world modeling</kbd> |

### Reading Order

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020  
   Skim for: <kbd>diffusion objective</kbd>, <kbd>denoising process</kbd>

2. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022  
   Skim for: <kbd>latent diffusion</kbd>, <kbd>cross-attention conditioning</kbd>

3. [Video Diffusion Models](https://arxiv.org/abs/2204.03458), NeurIPS 2022  
   Skim for: <kbd>video diffusion</kbd>

4. [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127), 2023  
   Skim for: <kbd>image-to-video</kbd>, <kbd>text-to-video scaling</kbd>

5. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), 2023  
   Skim for: <kbd>DreamerV3</kbd>, <kbd>latent imagination</kbd>, <kbd>control</kbd>

6. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391), ICML 2024  
   Skim for: <kbd>action-controllable environments</kbd>, <kbd>unlabeled videos</kbd>

7. [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080), 2023  
   Skim for: <kbd>generative driving simulation</kbd>, <kbd>video data</kbd>

8. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523), 2025  
   Skim for: <kbd>controllable multi-view generation</kbd>, <kbd>driving world generation</kbd>

9. [DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT](https://arxiv.org/abs/2412.19505), 2024  
   Skim for: <kbd>video-token world modeling</kbd>, <kbd>driving</kbd>

### YouTube Skim Resources

- [DDPM basics video](https://www.youtube.com/watch?v=h6T1UU9pzOY)  
  Good first skim for diffusion mechanics before DDPM and latent diffusion.

- [Stanford CS231n generative models and diffusion lecture search](https://www.youtube.com/results?search_query=Stanford+CS231n+diffusion+generative+models+lecture+2025)  
  Use this for a course-level introduction to modern generative modeling.

- [Stable Video Diffusion explanation search](https://www.youtube.com/results?search_query=Stable+Video+Diffusion+paper+explained)  
  Use this after latent diffusion.

- [Genie world model explanation search](https://www.youtube.com/results?search_query=Genie+Generative+Interactive+Environments+world+model+explained)  
  Use this for interactive world-model intuition.

- [GAIA driving world model explanation search](https://www.youtube.com/results?search_query=GAIA+generative+world+model+autonomous+driving+explained)  
  Use this for the driving-specific world-model framing.

## 10. Robustness, Uncertainty, and Domain Adaptation

### Field Evolution

| Papers | Shift |
|---|---|
| **Explaining and Harnessing Adversarial Examples → PGD adversarial training** | <kbd>adversarial-example setup</kbd> → <kbd>robust optimization</kbd> |
| **Domain-Adversarial Training → Deep Ensembles → Bayesian uncertainty** | <kbd>domain-invariant representations</kbd> → <kbd>ensemble uncertainty</kbd>, <kbd>calibration</kbd>, <kbd>epistemic uncertainty</kbd> |
| **Bayesian uncertainty → Tent** | <kbd>aleatoric uncertainty</kbd>, <kbd>dense prediction</kbd> → <kbd>entropy minimization</kbd>, <kbd>batch-norm adaptation</kbd> |
| **Tent → RobustBench → WILDS** | <kbd>test time</kbd> adaptation → <kbd>standardized robustness evaluation</kbd>, <kbd>realistic domain shift</kbd> |

### Reading Order

1. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), ICLR 2015  
   Skim for: <kbd>FGSM</kbd>, <kbd>adversarial-example setup</kbd>

2. [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), ICLR 2018  
   Skim for: <kbd>PGD adversarial training</kbd>, <kbd>robust optimization</kbd>

3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818), JMLR 2016  
   Skim for: <kbd>gradient reversal</kbd>, <kbd>domain-invariant representations</kbd>

4. [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles), NeurIPS 2017  
   Skim for: <kbd>ensemble uncertainty</kbd>, <kbd>calibration</kbd>, <kbd>OOD behavior</kbd>

5. [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977), NeurIPS 2017  
   Skim for: <kbd>epistemic uncertainty</kbd>, <kbd>aleatoric uncertainty</kbd>, <kbd>dense prediction</kbd>

6. [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726), ICLR 2021  
   Skim for: <kbd>entropy minimization</kbd>, <kbd>batch-norm adaptation</kbd>, <kbd>test time</kbd>

7. [RobustBench: a standardized adversarial robustness benchmark](https://arxiv.org/abs/2010.09670), NeurIPS 2021  
   Skim for: <kbd>standardized robustness evaluation</kbd>, <kbd>attack pitfalls</kbd>

8. [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), ICML 2021  
   Skim for: <kbd>realistic domain shift</kbd>, <kbd>benchmark design</kbd>

### YouTube Skim Resources

- [Adversarial robustness lecture search](https://www.youtube.com/results?search_query=adversarial+examples+robustness+PGD+lecture)  
  Use before FGSM, PGD adversarial training, and RobustBench.

- [Stanford CS330 Domain Adaptation lecture search](https://www.youtube.com/results?search_query=Stanford+CS330+domain+adaptation+lecture)  
  Good background before DANN and Tent.

- [Uncertainty in deep learning lecture search](https://www.youtube.com/results?search_query=uncertainty+in+deep+learning+deep+ensembles+lecture)  
  Use before Deep Ensembles and Bayesian uncertainty for vision.

- [Test-time adaptation paper explanation search](https://www.youtube.com/results?search_query=Tent+fully+test+time+adaptation+paper+explained)  
  Use after domain adaptation background.
