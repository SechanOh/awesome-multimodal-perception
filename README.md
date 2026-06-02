# Awesome Multimodal Perception

A reading roadmap for skimming essential computer vision papers across modern perception, geometry, embodied AI, generative vision, and robustness.

Last updated: 2026-06-02

## Table of Contents

- [1. Object Detection](#1-object-detection)
- [2. Segmentation](#2-segmentation)
- [3. Tracking](#3-tracking)
- [4. Depth, Geometry, and Optical Flow](#4-depth-geometry-and-optical-flow)
- [5. SLAM, Odometry, and Occupancy](#5-slam-odometry-and-occupancy)
- [6. 3D Reconstruction](#6-3d-reconstruction)
- [7. Foundation Models](#7-foundation-models)
- [8. End-to-End Autonomous Driving and Robotics Models](#8-end-to-end-autonomous-driving-and-robotics-models)
- [9. Generative Vision and World Models](#9-generative-vision-and-world-models)
- [10. Robustness, Uncertainty, and Domain Adaptation](#10-robustness-uncertainty-and-domain-adaptation)
- [Four-Week Skimming Plan](#four-week-skimming-plan)
- [Minimal 20-Paper Route](#minimal-20-paper-route)

## 1. Object Detection

### Background Concepts

- CNN backbones, feature pyramids, anchors, non-maximum suppression
- Two-stage vs one-stage detection
- Transformer object queries and bipartite matching
- Open-vocabulary detection and language grounding

### Reading Order

1. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS 2015  
   Skim for: region proposal networks, two-stage detection, anchors, RoI heads.

2. [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767), 2018  
   Skim for: one-stage real-time detection, multi-scale prediction, speed vs accuracy tradeoffs.

3. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020  
   Skim for: DETR, object queries, Hungarian matching, NMS-free detection.

4. [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159), ICLR 2021  
   Skim for: sparse multi-scale attention and faster DETR convergence.

5. [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605), ICLR 2023  
   Skim for: denoising training, anchor design, and strong DETR training recipes.

6. [Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857), CVPR 2022  
   Skim for: reframing detection as phrase grounding.

7. [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023  
   Skim for: text-conditioned open-set detection.

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the detection parts first. It covers detection task setup, R-CNN-style models, and modern detector families.

- [Stanford CS231n 2017 Lecture 11: Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo)  
  Older but useful for R-CNN and Faster R-CNN foundations.

- [DETR paper explanation search](https://www.youtube.com/results?search_query=DETR+paper+explained+object+detection+transformer)  
  Use this if Hungarian matching and object queries are unclear after the paper.

## 2. Segmentation

### Background Concepts

- Semantic, instance, and panoptic segmentation
- Mask heads and RoIAlign
- Mask classification
- Promptable segmentation
- Video object segmentation

### Reading Order

1. [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017  
   Skim for: RoIAlign and adding mask prediction to instance detection.

2. [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527), CVPR 2022  
   Skim for: Mask2Former and unified semantic, instance, and panoptic segmentation.

3. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: promptable segmentation, SA-1B, and ambiguity-aware mask prediction.

4. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: memory-based promptable segmentation over video.

### YouTube Skim Resources

- [Stanford CS231n 2025 Lecture 9: Object Detection, Image Segmentation, Visualizing](https://www.youtube.com/watch?v=PTypu6GqEd4)  
  Use the segmentation parts for task taxonomy, dense prediction, and mask-based outputs.

- [Stanford CS231n 2017 Lecture 11: Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo)  
  Useful for Mask R-CNN foundations.

- [Segment Anything paper explanation search](https://www.youtube.com/results?search_query=Segment+Anything+paper+explained)  
  Use this after reading SAM. Prefer explanations that show the prompt encoder, mask decoder, and data engine.

## 3. Tracking

### Background Concepts

- Single-object tracking vs multi-object tracking
- Tracking-by-detection
- Data association
- Motion models and appearance embeddings
- Video object segmentation as mask tracking

### Reading Order

1. [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), ICIP 2016  
   Skim for: Kalman filtering, Hungarian matching, and tracking-by-detection.

2. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402), ICIP 2017  
   Skim for: appearance embeddings for robust association.

3. [Tracktor++: Tracking without Bells and Whistles](https://arxiv.org/abs/1903.05625), ICCV 2019  
   Skim for: reusing detector regression for tracking.

4. [CenterTrack: Tracking Objects as Points](https://arxiv.org/abs/2004.01177), ECCV 2020  
   Skim for: joint detection and tracking with object centers.

5. [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702), CVPR 2022  
   Skim for: transformer queries extended from detection to tracking.

6. [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), ECCV 2022  
   Skim for: association with low-confidence detection boxes.

7. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: promptable video object tracking through segmentation memory.

### YouTube Skim Resources

- [Multi-object tracking lecture search](https://www.youtube.com/results?search_query=multi+object+tracking+computer+vision+lecture+SORT+DeepSORT)  
  Start here for SORT, DeepSORT, Kalman filters, and data association.

- [ByteTrack paper explanation search](https://www.youtube.com/results?search_query=ByteTrack+paper+explained+multi+object+tracking)  
  Use this after SORT and DeepSORT.

- [TrackFormer paper explanation search](https://www.youtube.com/results?search_query=TrackFormer+multi+object+tracking+transformer+paper+explained)  
  Use this to connect DETR-style queries with tracking.

## 4. Depth, Geometry, and Optical Flow

### Background Concepts

- Camera projection, epipolar geometry, stereo matching
- Dense correspondence and cost volumes
- Optical flow vs scene flow
- Monocular relative depth vs metric depth
- Self-supervised depth from video
- Multi-view geometry and feed-forward geometry prediction

### Reading Order

1. [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039), ECCV 2020  
   Skim for: all-pairs correlation volume and iterative flow refinement.

2. [GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680), CVPR 2022  
   Skim for: transformer-style global matching for flow.

3. [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer](https://arxiv.org/abs/1907.01341), TPAMI 2022  
   Skim for: MiDaS, relative depth, and multi-dataset training.

4. [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891), CVPR 2024  
   Skim for: large-scale unlabeled data and teacher-student depth training.

5. [Depth Anything V2](https://arxiv.org/abs/2406.09414), NeurIPS 2024  
   Skim for: synthetic data, sharper predictions, and practical model variants.

6. [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: pointmap prediction from unconstrained image pairs.

7. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: feed-forward prediction of cameras, depth maps, point maps, and tracks.

8. [DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_DEFOM-Stereo_Depth_Foundation_Model_Based_Stereo_Matching_CVPR_2025_paper.pdf), CVPR 2025  
   Skim for: using monocular depth foundation priors in stereo matching.

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

### Background Concepts

- Visual odometry vs SLAM
- Feature tracking, loop closure, place recognition
- Bundle adjustment and pose graph optimization
- Visual-inertial odometry
- Dense mapping and neural implicit mapping
- BEV and 3D occupancy for autonomous driving

### Reading Order

1. [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898), T-RO 2021  
   Skim for: feature-based SLAM, visual-inertial integration, loop closure, and multi-map reuse.

2. [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869), NeurIPS 2021  
   Skim for: learned recurrent updates and dense bundle adjustment.

3. [NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://arxiv.org/abs/2112.12130), CVPR 2022  
   Skim for: neural implicit dense mapping and tracking.

4. [Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM](https://arxiv.org/abs/2304.14377), CVPR 2023  
   Skim for: real-time neural SLAM with hybrid encodings.

5. [Scene as Occupancy](https://arxiv.org/abs/2306.02851), ICCV 2023  
   Skim for: camera-only 3D occupancy prediction.

6. [OccTransformer: Improving BEVFormer for 3D Camera-Only Occupancy Prediction](https://arxiv.org/abs/2402.18140), 2024  
   Skim for: extending BEV-style perception to occupancy.

7. [LiDAR-based 4D Occupancy Completion and Forecasting](https://arxiv.org/abs/2310.11239), 2023  
   Skim for: occupancy completion and future occupancy forecasting.

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

### Background Concepts

- Structure from Motion
- Multi-view stereo
- Bundle adjustment
- Neural radiance fields
- Differentiable rendering
- Explicit vs implicit 3D representations
- Feed-forward reconstruction

### Reading Order

1. [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf), CVPR 2016  
   Skim for: the classical SfM pipeline and why COLMAP remains a standard dependency.

2. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934), ECCV 2020  
   Skim for: volumetric rendering, positional encoding, and view synthesis.

3. [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989), SIGGRAPH 2022  
   Skim for: hash grid encoding and fast neural scene fitting.

4. [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079), SIGGRAPH 2023  
   Skim for: explicit anisotropic Gaussians and differentiable rasterization.

5. [pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction](https://arxiv.org/abs/2312.12337), CVPR 2024  
   Skim for: feed-forward Gaussian splat prediction.

6. [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024  
   Skim for: reconstruction without calibrated cameras.

7. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651), CVPR 2025  
   Skim for: generalist 3D geometry as a foundation model.

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

### Background Concepts

- Transformer encoder architecture
- Contrastive image-text pretraining
- Masked image modeling
- Self-supervised visual representation learning
- Promptable perception
- Vision-language model alignment
- Video foundation encoders

### Reading Order

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021  
   Skim for: ViT patch tokens, scale, and inductive-bias tradeoffs.

2. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021  
   Skim for: CLIP, image-text contrastive learning, and zero-shot transfer.

3. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022  
   Skim for: masked image modeling and asymmetric encoder-decoder design.

4. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), TMLR 2024  
   Skim for: self-supervised features, curated data, and dense transfer.

5. [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), ICCV 2023  
   Skim for: SigLIP and scalable image-text loss design.

6. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023  
   Skim for: promptable segmentation as a foundation task.

7. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024  
   Skim for: image-video segmentation with memory.

8. [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217), 2024  
   Skim for: video foundation representation learning.

9. [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238), CVPR 2024  
   Skim for: vision encoder scaling and VLM alignment.

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

### Background Concepts

- Imitation learning and behavior cloning
- Planning vs end-to-end action prediction
- BEV perception and occupancy
- Visuomotor control
- Diffusion policies
- Vision-language-action models
- Robot embodiment and action tokenization

### Reading Order

1. [End-to-end Learning of Driving Models from Large-scale Video Datasets](https://arxiv.org/abs/1612.01079), CVPR 2017  
   Skim for: early large-scale imitation learning for driving.

2. [Planning-Oriented Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html), CVPR 2023  
   Skim for: UniAD and the integration of perception, prediction, mapping, and planning.

3. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023 / IJRR 2024  
   Skim for: action diffusion, receding-horizon control, and multimodal action distributions.

4. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818), CoRL 2023  
   Skim for: treating robot actions as language-model tokens.

5. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246), CoRL 2024  
   Skim for: open VLA training, Open X-Embodiment data, and parameter-efficient adaptation.

6. [OpenDriveVLA: Towards End-to-End Autonomous Driving with Large Vision Language Action Model](https://arxiv.org/abs/2503.23463), 2025  
   Skim for: adapting the VLA framing to autonomous driving.

7. [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/abs/2503.19755), 2025  
   Skim for: language-instructed driving action generation.

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

### Background Concepts

- Score matching and denoising diffusion
- Latent diffusion
- Video generation and temporal consistency
- Action-conditioned generation
- Latent world models
- Interactive environments
- Autonomous-driving simulation

### Reading Order

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020  
   Skim for: the diffusion objective and denoising process.

2. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022  
   Skim for: latent diffusion and cross-attention conditioning.

3. [Video Diffusion Models](https://arxiv.org/abs/2204.03458), NeurIPS 2022  
   Skim for: extending diffusion to video.

4. [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127), 2023  
   Skim for: practical image-to-video and text-to-video scaling.

5. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), 2023  
   Skim for: DreamerV3 and latent imagination for control.

6. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391), ICML 2024  
   Skim for: learning action-controllable environments from unlabeled videos.

7. [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080), 2023  
   Skim for: generative driving simulation from video data.

8. [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523), 2025  
   Skim for: controllable multi-view driving world generation.

9. [DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT](https://arxiv.org/abs/2412.19505), 2024  
   Skim for: video-token world modeling for driving.

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

### Background Concepts

- Adversarial examples and threat models
- PGD adversarial training
- Calibration and predictive uncertainty
- Epistemic vs aleatoric uncertainty
- Domain adaptation and domain generalization
- Test-time adaptation
- Distribution-shift benchmarks

### Reading Order

1. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), ICLR 2015  
   Skim for: FGSM and the basic adversarial-example setup.

2. [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), ICLR 2018  
   Skim for: PGD adversarial training and robust optimization.

3. [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818), JMLR 2016  
   Skim for: gradient reversal and domain-invariant representations.

4. [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles), NeurIPS 2017  
   Skim for: ensemble uncertainty, calibration, and OOD behavior.

5. [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977), NeurIPS 2017  
   Skim for: epistemic vs aleatoric uncertainty in dense prediction.

6. [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726), ICLR 2021  
   Skim for: entropy minimization and batch-norm adaptation at test time.

7. [RobustBench: a standardized adversarial robustness benchmark](https://arxiv.org/abs/2010.09670), NeurIPS 2021  
   Skim for: standardized robustness evaluation and attack pitfalls.

8. [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421), ICML 2021  
   Skim for: realistic domain shift and benchmark design.

### YouTube Skim Resources

- [Adversarial robustness lecture search](https://www.youtube.com/results?search_query=adversarial+examples+robustness+PGD+lecture)  
  Use before FGSM, PGD adversarial training, and RobustBench.

- [Stanford CS330 Domain Adaptation lecture search](https://www.youtube.com/results?search_query=Stanford+CS330+domain+adaptation+lecture)  
  Good background before DANN and Tent.

- [Uncertainty in deep learning lecture search](https://www.youtube.com/results?search_query=uncertainty+in+deep+learning+deep+ensembles+lecture)  
  Use before Deep Ensembles and Bayesian uncertainty for vision.

- [Test-time adaptation paper explanation search](https://www.youtube.com/results?search_query=Tent+fully+test+time+adaptation+paper+explained)  
  Use after domain adaptation background.

## Four-Week Skimming Plan

### Week 1: Foundation Models, Detection, Segmentation, and Tracking

- Day 1: ViT, CLIP, MAE
- Day 2: DINOv2, SAM, SAM 2
- Day 3: Faster R-CNN, YOLOv3, DETR
- Day 4: Deformable DETR, DINO, Grounding DINO
- Day 5: Mask R-CNN, Mask2Former, SAM, ByteTrack

### Week 2: Geometry and Reconstruction

- Day 1: RAFT, GMFlow
- Day 2: MiDaS, Depth Anything, Depth Anything V2
- Day 3: COLMAP, NeRF, Instant-NGP
- Day 4: 3D Gaussian Splatting, pixelSplat
- Day 5: DUSt3R, VGGT, DEFOM-Stereo

### Week 3: SLAM, Occupancy, Driving, and Robotics

- Day 1: ORB-SLAM3, DROID-SLAM
- Day 2: NICE-SLAM, Co-SLAM
- Day 3: Scene as Occupancy, OccTransformer, 4D Occupancy Completion
- Day 4: UniAD, Diffusion Policy, RT-2
- Day 5: OpenVLA, OpenDriveVLA, ORION

### Week 4: Generative Vision, World Models, and Robustness

- Day 1: DDPM, Latent Diffusion, Video Diffusion
- Day 2: Stable Video Diffusion, DreamerV3, Genie
- Day 3: GAIA-1, GAIA-2, DrivingWorld
- Day 4: FGSM, PGD adversarial training, DANN
- Day 5: Deep Ensembles, Bayesian uncertainty, Tent, RobustBench, WILDS

## Minimal 20-Paper Route

If time is limited, read only these first:

1. ViT
2. CLIP
3. MAE
4. DINOv2
5. SAM
6. SAM 2
7. DETR
8. Grounding DINO
9. Mask2Former
10. RAFT
11. Depth Anything V2
12. DUSt3R
13. VGGT
14. ORB-SLAM3
15. DROID-SLAM
16. 3D Gaussian Splatting
17. UniAD
18. RT-2
19. OpenVLA
20. Tent
