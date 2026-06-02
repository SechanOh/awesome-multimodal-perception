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

Object detection moved from <mark>hand-crafted features</mark> and <mark>sliding windows</mark> to <mark>CNN-based two-stage detectors</mark>, then to <mark>one-stage real-time detectors</mark>, <mark>transformer-based end-to-end detection</mark>, and finally <mark>language-grounded open-vocabulary detection</mark>.

### Reading Order

1. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS 2015  
   Skim for: <mark>region proposal networks</mark> <mark>two-stage detection</mark> <mark>anchors</mark> <mark>RoI heads</mark>

2. [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767), 2018  
   Skim for: <mark>one-stage real-time detection</mark> <mark>multi-scale prediction</mark> <mark>speed vs accuracy tradeoffs</mark>

3. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020  
   Skim for: <mark>DETR</mark> <mark>object queries</mark> <mark>Hungarian matching</mark> <mark>NMS-free detection</mark>

4. [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159), ICLR 2021  
   Skim for: <mark>sparse multi-scale attention</mark> <mark>faster DETR convergence</mark>

5. [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605), ICLR 2023  
   Skim for: <mark>denoising training</mark> <mark>anchor design</mark> <mark>strong DETR training recipes</mark>

6. [Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857), CVPR 2022  
   Skim for: <mark>phrase grounding</mark>

7. [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023  
   Skim for: <mark>text-conditioned open-set detection</mark>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the detection parts first. It covers detection task setup, R-CNN-style models, and modern detector families.

- [Stanford CS231n 2017 Lecture 11: Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo)  
  Older but useful for R-CNN and Faster R-CNN foundations.

- [DETR paper explanation search](https://www.youtube.com/results?search_query=DETR+paper+explained+object+detection+transformer)  
  Use this if Hungarian matching and object queries are unclear after the paper.

## 2. Segmentation

### Field Evolution

Segmentation began with <mark>class-level dense prediction</mark>, expanded into <mark>instance segmentation</mark> and <mark>panoptic segmentation</mark>, then shifted toward <mark>transformer mask classification</mark> and <mark>promptable foundation models</mark> for images and videos.

### Reading Order

1. [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017  
   Skim for: <mark>RoIAlign</mark> <mark>mask prediction</mark> <mark>instance detection</mark>

2. [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527), CVPR 2022  
   Skim for: <mark>Mask2Former</mark> <mark>universal segmentation</mark> <mark>semantic segmentation</mark> <mark>instance segmentation</mark> <mark>panoptic segmentation</mark>

3. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: <mark>promptable segmentation</mark> <mark>SA-1B</mark> <mark>ambiguity-aware mask prediction</mark>

4. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <mark>memory-based segmentation</mark> <mark>video segmentation</mark>

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the segmentation parts for task taxonomy, dense prediction, and mask-based outputs.

- [Stanford CS231n 2017 Lecture 11: Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo)  
  Useful for Mask R-CNN foundations.

- [Segment Anything paper explanation search](https://www.youtube.com/results?search_query=Segment+Anything+paper+explained)  
  Use this after reading SAM. Prefer explanations that show the prompt encoder, mask decoder, and data engine.

## 3. Object Tracking

### Field Evolution

Object tracking evolved from <mark>motion-model-based association</mark> and <mark>tracking-by-detection</mark> to <mark>appearance-aware association</mark>, <mark>joint detection-tracking models</mark>, <mark>transformer query tracking</mark>, and <mark>segmentation-memory-based video tracking</mark>.

### Reading Order

1. [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), ICIP 2016  
   Skim for: <mark>Kalman filtering</mark> <mark>Hungarian matching</mark> <mark>tracking-by-detection</mark>

2. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402), ICIP 2017  
   Skim for: <mark>appearance embeddings</mark> <mark>robust association</mark>

3. [Tracktor++: Tracking without Bells and Whistles](https://arxiv.org/abs/1903.05625), ICCV 2019  
   Skim for: <mark>detector regression</mark> <mark>tracking</mark>

4. [CenterTrack: Tracking Objects as Points](https://arxiv.org/abs/2004.01177), ECCV 2020  
   Skim for: <mark>joint detection and tracking</mark> <mark>object centers</mark>

5. [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702), CVPR 2022  
   Skim for: <mark>transformer queries</mark> <mark>detection-to-tracking</mark>

6. [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), ECCV 2022  
   Skim for: <mark>low-confidence detection boxes</mark> <mark>data association</mark>

7. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <mark>promptable video object tracking</mark> <mark>segmentation memory</mark>

### YouTube Skim Resources

- [Multi-object tracking lecture search](https://www.youtube.com/results?search_query=multi+object+tracking+computer+vision+lecture+SORT+DeepSORT)  
  Start here for SORT, DeepSORT, Kalman filters, and data association.

- [ByteTrack paper explanation search](https://www.youtube.com/results?search_query=ByteTrack+paper+explained+multi+object+tracking)  
  Use this after SORT and DeepSORT.

- [TrackFormer paper explanation search](https://www.youtube.com/results?search_query=TrackFormer+multi+object+tracking+transformer+paper+explained)  
  Use this to connect DETR-style queries with tracking.

## 4. Depth, Geometry, and Optical Flow

### Field Evolution

Geometry research started from <mark>projective geometry</mark>, <mark>stereo</mark>, and <mark>dense correspondence</mark>, then moved into <mark>learned optical flow</mark>, <mark>monocular depth</mark>, <mark>depth foundation models</mark>, and <mark>feed-forward geometry prediction</mark>.

### Reading Order

1. [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039), ECCV 2020  
   Skim for: <mark>all-pairs correlation volume</mark> <mark>iterative flow refinement</mark>

2. [GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680), CVPR 2022  
   Skim for: <mark>transformer-style global matching</mark> <mark>optical flow</mark>

3. [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer](https://arxiv.org/abs/1907.01341), TPAMI 2022  
   Skim for: <mark>MiDaS</mark> <mark>relative depth</mark> <mark>multi-dataset training</mark>

4. [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891), CVPR 2024  
   Skim for: <mark>large-scale unlabeled data</mark> <mark>teacher-student depth training</mark>

5. [Depth Anything V2](https://arxiv.org/abs/2406.09414), NeurIPS 2024  
   Skim for: <mark>synthetic data</mark> <mark>sharper predictions</mark> <mark>practical model variants</mark>

6. [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: <mark>pointmap prediction</mark> <mark>unconstrained image pairs</mark>

7. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: <mark>feed-forward prediction</mark> <mark>cameras</mark> <mark>depth maps</mark> <mark>point maps</mark> <mark>tracks</mark>

8. [DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_DEFOM-Stereo_Depth_Foundation_Model_Based_Stereo_Matching_CVPR_2025_paper.pdf), CVPR 2025  
   Skim for: <mark>monocular depth foundation priors</mark> <mark>stereo matching</mark>

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

SLAM and odometry progressed from <mark>feature tracking</mark>, <mark>bundle adjustment</mark>, and <mark>loop closure</mark> to <mark>learned dense optimization</mark>, <mark>neural implicit mapping</mark>, and <mark>occupancy-centric scene representations</mark> for autonomous driving.

### Reading Order

1. [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898), T-RO 2021  
   Skim for: <mark>feature-based SLAM</mark> <mark>visual-inertial integration</mark> <mark>loop closure</mark> <mark>multi-map reuse</mark>

2. [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869), NeurIPS 2021  
   Skim for: <mark>learned recurrent updates</mark> <mark>dense bundle adjustment</mark>

3. [NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://arxiv.org/abs/2112.12130), CVPR 2022  
   Skim for: <mark>neural implicit dense mapping</mark> <mark>tracking</mark>

4. [Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM](https://arxiv.org/abs/2304.14377), CVPR 2023  
   Skim for: <mark>real-time neural SLAM</mark> <mark>hybrid encodings</mark>

5. [Scene as Occupancy](https://arxiv.org/abs/2306.02851), ICCV 2023  
   Skim for: <mark>camera-only 3D occupancy prediction</mark>

6. [OccTransformer: Improving BEVFormer for 3D Camera-Only Occupancy Prediction](https://arxiv.org/abs/2402.18140), 2024  
   Skim for: <mark>BEV-style perception</mark> <mark>occupancy</mark>

7. [LiDAR-based 4D Occupancy Completion and Forecasting](https://arxiv.org/abs/2310.11239), 2023  
   Skim for: <mark>occupancy completion</mark> <mark>future occupancy forecasting</mark>

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

3D reconstruction grew out of <mark>Structure from Motion</mark> and <mark>multi-view stereo</mark>, then shifted to <mark>neural radiance fields</mark>, <mark>fast neural encodings</mark>, <mark>explicit Gaussian representations</mark>, and <mark>feed-forward reconstruction</mark>.

### Reading Order

1. [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf), CVPR 2016  
   Skim for: <mark>classical SfM pipeline</mark> <mark>COLMAP</mark>

2. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934), ECCV 2020  
   Skim for: <mark>volumetric rendering</mark> <mark>positional encoding</mark> <mark>view synthesis</mark>

3. [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989), SIGGRAPH 2022  
   Skim for: <mark>hash grid encoding</mark> <mark>fast neural scene fitting</mark>

4. [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), SIGGRAPH 2023  
   Skim for: <mark>explicit anisotropic Gaussians</mark> <mark>differentiable rasterization</mark>

5. [pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction](https://arxiv.org/abs/2312.12337), CVPR 2024  
   Skim for: <mark>feed-forward Gaussian splat prediction</mark>

6. [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: <mark>reconstruction without calibrated cameras</mark>

7. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: <mark>generalist 3D geometry</mark> <mark>foundation model</mark>

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

Vision foundation models moved from <mark>supervised CNN backbones</mark> to <mark>ViT scaling</mark>, <mark>contrastive image-text pretraining</mark>, <mark>masked image modeling</mark>, <mark>self-supervised dense features</mark>, <mark>promptable perception</mark>, and <mark>vision-language generalist encoders</mark>.

### Reading Order

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021  
   Skim for: <mark>ViT patch tokens</mark> <mark>scale</mark> <mark>inductive-bias tradeoffs</mark>

2. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021  
   Skim for: <mark>CLIP</mark> <mark>image-text contrastive learning</mark> <mark>zero-shot transfer</mark>

3. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022  
   Skim for: <mark>masked image modeling</mark> <mark>asymmetric encoder-decoder design</mark>

4. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), TMLR 2024  
   Skim for: <mark>self-supervised features</mark> <mark>curated data</mark> <mark>dense transfer</mark>

5. [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), ICCV 2023  
   Skim for: <mark>SigLIP</mark> <mark>scalable image-text loss design</mark>

6. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: <mark>promptable segmentation</mark> <mark>foundation task</mark>

7. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: <mark>image-video segmentation</mark> <mark>memory</mark>

8. [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217), 2024  
   Skim for: <mark>video foundation representation learning</mark>

9. [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238), CVPR 2024  
   Skim for: <mark>vision encoder scaling</mark> <mark>VLM alignment</mark>

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

End-to-end driving and robotics started with <mark>behavior cloning</mark> and <mark>modular perception-control stacks</mark>, then moved toward <mark>planning-oriented driving networks</mark>, <mark>diffusion-based visuomotor policies</mark>, and <mark>vision-language-action models</mark>.

### Reading Order

1. [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), CVPR 2017  
   Skim for: <mark>large-scale imitation learning</mark> <mark>driving</mark>

2. [Planning-Oriented Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html), CVPR 2023  
   Skim for: <mark>UniAD</mark> <mark>perception</mark> <mark>prediction</mark> <mark>mapping</mark> <mark>planning</mark>

3. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023 / IJRR 2024  
   Skim for: <mark>action diffusion</mark> <mark>receding-horizon control</mark> <mark>multimodal action distributions</mark>

4. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818), CoRL 2023  
   Skim for: <mark>robot actions</mark> <mark>language-model tokens</mark>

5. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246), CoRL 2024  
   Skim for: <mark>open VLA training</mark> <mark>Open X-Embodiment data</mark> <mark>parameter-efficient adaptation</mark>

6. [OpenDriveVLA: Towards End-to-End Autonomous Driving with Large Vision Language Action Model](https://arxiv.org/abs/2503.23463), 2025  
   Skim for: <mark>VLA framing</mark> <mark>autonomous driving</mark>

7. [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/abs/2503.19755), 2025  
   Skim for: <mark>language-instructed driving action generation</mark>

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

Generative vision advanced from <mark>image diffusion</mark> to <mark>latent diffusion</mark> and <mark>video generation</mark>, then expanded into <mark>action-conditioned interactive environments</mark>, <mark>latent world models</mark>, and <mark>controllable driving simulators</mark>.

### Reading Order

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020  
   Skim for: <mark>diffusion objective</mark> <mark>denoising process</mark>

2. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022  
   Skim for: <mark>latent diffusion</mark> <mark>cross-attention conditioning</mark>

3. [Video Diffusion Models](https://arxiv.org/abs/2204.03458), NeurIPS 2022  
   Skim for: <mark>video diffusion</mark>

4. [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127), 2023  
   Skim for: <mark>image-to-video</mark> <mark>text-to-video scaling</mark>

5. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), 2023  
   Skim for: <mark>DreamerV3</mark> <mark>latent imagination</mark> <mark>control</mark>

6. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391), ICML 2024  
   Skim for: <mark>action-controllable environments</mark> <mark>unlabeled videos</mark>

7. [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080), 2023  
   Skim for: <mark>generative driving simulation</mark> <mark>video data</mark>

8. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523), 2025  
   Skim for: <mark>controllable multi-view generation</mark> <mark>driving world generation</mark>

9. [DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT](https://arxiv.org/abs/2412.19505), 2024  
   Skim for: <mark>video-token world modeling</mark> <mark>driving</mark>

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

Robustness research began with <mark>adversarial examples</mark> and <mark>robust optimization</mark>, expanded into <mark>uncertainty estimation</mark> and <mark>calibration</mark>, then developed <mark>domain adaptation</mark>, <mark>test-time adaptation</mark>, and <mark>distribution-shift benchmarks</mark>.

### Reading Order

1. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), ICLR 2015  
   Skim for: <mark>FGSM</mark> <mark>adversarial-example setup</mark>

2. [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), ICLR 2018  
   Skim for: <mark>PGD adversarial training</mark> <mark>robust optimization</mark>

3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818), JMLR 2016  
   Skim for: <mark>gradient reversal</mark> <mark>domain-invariant representations</mark>

4. [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles), NeurIPS 2017  
   Skim for: <mark>ensemble uncertainty</mark> <mark>calibration</mark> <mark>OOD behavior</mark>

5. [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977), NeurIPS 2017  
   Skim for: <mark>epistemic uncertainty</mark> <mark>aleatoric uncertainty</mark> <mark>dense prediction</mark>

6. [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726), ICLR 2021  
   Skim for: <mark>entropy minimization</mark> <mark>batch-norm adaptation</mark> <mark>test time</mark>

7. [RobustBench: a standardized adversarial robustness benchmark](https://arxiv.org/abs/2010.09670), NeurIPS 2021  
   Skim for: <mark>standardized robustness evaluation</mark> <mark>attack pitfalls</mark>

8. [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), ICML 2021  
   Skim for: <mark>realistic domain shift</mark> <mark>benchmark design</mark>

### YouTube Skim Resources

- [Adversarial robustness lecture search](https://www.youtube.com/results?search_query=adversarial+examples+robustness+PGD+lecture)  
  Use before FGSM, PGD adversarial training, and RobustBench.

- [Stanford CS330 Domain Adaptation lecture search](https://www.youtube.com/results?search_query=Stanford+CS330+domain+adaptation+lecture)  
  Good background before DANN and Tent.

- [Uncertainty in deep learning lecture search](https://www.youtube.com/results?search_query=uncertainty+in+deep+learning+deep+ensembles+lecture)  
  Use before Deep Ensembles and Bayesian uncertainty for vision.

- [Test-time adaptation paper explanation search](https://www.youtube.com/results?search_query=Tent+fully+test+time+adaptation+paper+explained)  
  Use after domain adaptation background.
