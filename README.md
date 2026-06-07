# Camera-Based End-to-End Autonomous Driving: Minimal Paper Skimming Roadmap

Last updated: 2026-06-07

This roadmap is intentionally selective. The goal is not to survey every subfield, but to skim the minimum set of papers needed to understand the main technical line from camera images to 3D/BEV scene representation, motion reasoning, planning, and end-to-end autonomous driving.

Notation:

- **[CV LANDMARK]** means the paper was broadly influential across computer vision, not merely important inside autonomous driving.
- A paper without the landmark label can still be essential for camera-based autonomous driving.
- The recommended reading mode is skimming: read the abstract, method figure, representation, loss, evaluation protocol, and failure cases. Do not spend equal effort on implementation details unless you are reproducing the method.

---

## 1. Topic Composition Feedback

Your original topic list is directionally correct, but it mixes low-level tasks, 3D geometry, scene representation, temporal reasoning, planning, and foundation models in one flat list. For camera-based end-to-end autonomous driving, it is better to organize the topics by the information flow of the driving stack.

### Recommended structure

| Layer | Topics | Priority for camera-only E2E driving |
|---|---|---|
| A. Image-level perception | 2D detection, 2D segmentation, 2D tracking, visual backbones | P1 |
| B. Geometry to 3D/BEV | depth estimation, multi-view geometry, 3D detection, camera calibration, BEV lifting | P0 |
| C. Scene representation | BEV perception, occupancy, online HD map, 3D segmentation, 3D tracking | P0 |
| D. Temporal reasoning | tracking, motion forecasting, temporal BEV, 4D occupancy, world state prediction | P0 |
| E. Decision making | planning, control, imitation learning, trajectory optimization, policy learning | P0 |
| F. Foundation and generative models | ViT, CLIP, SAM, DINO, VLM/VLA, NeRF, 3D Gaussian Splatting, world models | P1 |
| G. Deployment and safety | robustness, uncertainty, OOD detection, closed-loop evaluation, safety constraints | P0 |

### Topics to add explicitly

1. **Datasets, benchmarks, and evaluation**  
   Autonomous driving papers are heavily shaped by the benchmark and metric. You need at least a shallow understanding of KITTI, Cityscapes, nuScenes, Waymo Open Dataset, CARLA, and nuPlan.

2. **Online HD map construction**  
   Lane centerlines, dividers, crosswalks, road boundaries, and vectorized maps matter directly for planning. Put HDMapNet, VectorMapNet, and MapTR near BEV perception.

3. **Occupancy and 4D occupancy**  
   Camera-only autonomous driving is moving from box-centric perception toward dense scene representation. Occupancy is often a better interface for planning than boxes alone.

4. **Closed-loop evaluation**  
   Open-loop prediction/planning metrics can look good while closed-loop driving fails. Keep CARLA, nuPlan, and closed-loop safety metrics in view from the beginning.

### Topics to de-emphasize

- **2D detection, 2D segmentation, and 2D tracking**: skim the historical milestones and move on. They are background for modern AD, not the core bottleneck.
- **NeRF, 3D Gaussian Splatting, and reconstruction**: important for simulation, scene generation, and world models, but usually secondary for reading camera-only E2E planning papers.
- **General robot learning**: useful conceptually, especially imitation learning and diffusion policies, but do not let it distract from BEV/occupancy/planning if your main target is autonomous driving.

---

## 2. Recommended Reading Order

1. Multi-view geometry, depth, and camera-to-BEV lifting.
2. BEV perception, online mapping, and occupancy.
3. Tracking and motion forecasting.
4. Planning and control.
5. End-to-end autonomous driving frameworks.
6. Foundation models and generative/world models.
7. Robustness, uncertainty, and safety.

The central question to keep asking is:

> What is the intermediate representation between raw camera images and the final ego trajectory?

Most modern camera-based AD papers differ mainly in how they answer that question: boxes, BEV grids, sparse queries, vectorized maps, occupancy, latent tokens, or generated futures.

---

## 3. Minimal Paper List by Topic

### 3.1 Datasets, Benchmarks, and Evaluation

| Paper / Resource | Why skim it |
|---|---|
| [KITTI Vision Benchmark Suite](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | Classic benchmark for stereo, optical flow, visual odometry, and 3D detection. |
| [Cityscapes](https://arxiv.org/abs/1604.01685) | Standard urban semantic segmentation dataset. |
| [nuScenes](https://arxiv.org/abs/1903.11027) | Core modern benchmark for 3D detection, tracking, prediction, and planning-style research. |
| [Waymo Open Dataset](https://arxiv.org/abs/1912.04838) | Large-scale autonomous driving dataset with strong 3D perception and motion prediction influence. |
| [CARLA](https://arxiv.org/abs/1711.03938) | Simulation benchmark commonly used for closed-loop driving evaluation. |
| [nuPlan](https://arxiv.org/abs/2106.11810) | Planning benchmark focused on closed-loop ML-based driving evaluation. |

Skim objective: understand what each dataset measures, what sensors it provides, and whether the paper you are reading is evaluated open-loop, closed-loop, or both.

---

### 3.2 Visual Backbones and Foundation Models

| Paper | Label | Why skim it |
|---|---|---|
| [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | **[CV LANDMARK]** | The deep learning turning point for large-scale visual recognition. |
| [ResNet](https://arxiv.org/abs/1512.03385) | **[CV LANDMARK]** | Residual learning became the default grammar of deep visual backbones. |
| [Vision Transformer](https://arxiv.org/abs/2010.11929) | **[CV LANDMARK]** | Introduced pure transformer image modeling at scale. Query-based BEV and AD papers inherit much of this language. |
| [CLIP](https://arxiv.org/abs/2103.00020) | **[CV LANDMARK]** | Image-text contrastive pretraining and zero-shot transfer. Important for VLM-based driving. |
| [DINOv2](https://arxiv.org/abs/2304.07193) |  | Strong self-supervised visual representation; useful for frozen visual encoder discussions. |
| [Segment Anything](https://arxiv.org/abs/2304.02643) | **[CV LANDMARK]** | Promptable segmentation foundation model. Important as a general-purpose perception prior. |

Skim objective: know the representational assumptions behind CNNs, transformers, contrastive vision-language models, and promptable perception.

---

### 3.3 2D Object Detection

| Paper | Label | Why skim it |
|---|---|---|
| [R-CNN](https://arxiv.org/abs/1311.2524) | **[CV LANDMARK]** | Established the CNN-based detection pipeline. |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | **[CV LANDMARK]** | Region Proposal Network and two-stage detection baseline. |
| [YOLO](https://arxiv.org/abs/1506.02640) | **[CV LANDMARK]** | One-stage real-time detection paradigm. |
| [DETR](https://arxiv.org/abs/2005.12872) | **[CV LANDMARK]** | Set prediction, object queries, bipartite matching. Directly relevant to BEVFormer, PETR, UniAD, and query-based AD. |

Skim objective: understand the shift from proposal-based detection to dense one-stage detection to query-based set prediction.

---

### 3.4 2D Segmentation

| Paper | Label | Why skim it |
|---|---|---|
| [FCN](https://arxiv.org/abs/1411.4038) | **[CV LANDMARK]** | Converted classification CNNs into dense prediction networks. |
| [U-Net](https://arxiv.org/abs/1505.04597) | **[CV LANDMARK]** | Encoder-decoder with skip connections; still a default dense prediction pattern. |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | **[CV LANDMARK]** | Strong instance segmentation baseline built on detection. |
| [Segment Anything](https://arxiv.org/abs/2304.02643) | **[CV LANDMARK]** | General promptable segmentation model; useful for foundation-model era perception. |

Skim objective: understand semantic vs instance segmentation, dense prediction heads, and why segmentation priors matter for scene understanding.

---

### 3.5 3D Segmentation and Occupancy

| Paper | Label | Why skim it |
|---|---|---|
| [PointNet](https://arxiv.org/abs/1612.00593) | **[CV LANDMARK]** | Direct neural processing of point clouds. Core 3D representation paper. |
| [MinkowskiNet / Sparse Conv](https://arxiv.org/abs/1904.08755) |  | Sparse 3D/4D convolution. Important for voxel and occupancy reasoning. |
| [SurroundOcc](https://arxiv.org/abs/2303.09551) |  | Multi-camera 3D occupancy prediction around the ego vehicle. |
| [Occ3D](https://arxiv.org/abs/2304.14365) |  | Occupancy benchmark direction for nuScenes and Waymo-style AD. |

Skim objective: understand why occupancy can be a more general scene representation than 3D boxes.

---

### 3.6 2D and 3D Object Tracking

| Paper | Label | Why skim it |
|---|---|---|
| [SORT](https://arxiv.org/abs/1602.00763) |  | Minimal tracking-by-detection: Kalman filter plus Hungarian matching. |
| [DeepSORT](https://arxiv.org/abs/1703.07402) |  | Adds appearance embeddings to the SORT association pipeline. |
| [ByteTrack](https://arxiv.org/abs/2110.06864) |  | Uses low-score detections for association; simple and strong MOT baseline. |
| [CenterTrack](https://arxiv.org/abs/2004.01177) |  | Links detection and tracking through point-based representation. |
| [MOTR](https://arxiv.org/abs/2105.03247) |  | Extends DETR-style queries to multi-object tracking. Useful for temporal query thinking. |

Skim objective: understand tracking-by-detection, association, temporal queries, and how identity is maintained over time.

---

### 3.7 Depth Estimation

| Paper | Label | Why skim it |
|---|---|---|
| [Monodepth2](https://arxiv.org/abs/1806.01260) |  | Core self-supervised monocular depth baseline: reprojection, masking, occlusion handling. |
| [DPT](https://arxiv.org/abs/2103.13413) |  | Transformer-based dense prediction and robust relative depth. |
| [Depth Anything V2](https://arxiv.org/abs/2406.09414) |  | Large-scale monocular depth model; useful for current generalization and scaling trends. |

Skim objective: understand why monocular depth is ill-posed, how self-supervision works, and why depth supervision helps BEV lifting.

---

### 3.8 Multiple View Geometry and Matching

| Paper / Book | Label | Why skim it |
|---|---|---|
| [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) | **[CV LANDMARK]** | Classical local feature matching foundation for SfM, SLAM, and multi-view geometry. |
| [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) |  | Best compact reference for epipolar geometry, homography, triangulation, and camera pose. |
| [SuperPoint](https://arxiv.org/abs/1712.07629) |  | Learned keypoints and descriptors. |
| [SuperGlue](https://arxiv.org/abs/1911.11763) |  | Graph neural network matching between local features. |
| [LoFTR](https://arxiv.org/abs/2104.00680) |  | Detector-free transformer matching; important for low-texture and wide-baseline matching. |

Skim objective: know epipolar geometry, projection, triangulation, feature matching, and how learned matching replaces parts of the classical pipeline.

---

### 3.9 3D Object Detection from Cameras

| Paper | Label | Why skim it |
|---|---|---|
| [FCOS3D](https://arxiv.org/abs/2104.10956) |  | Monocular 3D detection by extending 2D dense detectors into 3D. |
| [DETR3D](https://arxiv.org/abs/2110.06922) |  | 3D object queries sample multi-view 2D image features. Core camera-only 3D detection pattern. |
| [PETR](https://arxiv.org/abs/2203.05625) |  | Injects 3D position embeddings into image features. Another core query-based camera 3D detection line. |
| [BEVDepth](https://arxiv.org/abs/2206.10092) |  | Shows the importance of explicit depth supervision in camera-BEV 3D detection. |

Skim objective: understand how camera-only methods bridge the gap from 2D images to metric 3D boxes.

---

### 3.10 BEV Perception and Online Mapping

| Paper | Label | Why skim it |
|---|---|---|
| [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711) |  | Canonical camera-to-BEV lifting: lift image features into frustums, then splat into BEV. |
| [BEVDet](https://arxiv.org/abs/2112.11790) |  | Practical LSS-style BEV detection pipeline. |
| [BEVDepth](https://arxiv.org/abs/2206.10092) |  | Depth-supervised BEV detection; often the most important LSS-family idea to remember. |
| [BEVFormer](https://arxiv.org/abs/2203.17270) |  | Spatiotemporal transformer BEV representation using BEV queries, spatial cross-attention, and temporal self-attention. |
| [HDMapNet](https://arxiv.org/abs/2107.06307) |  | Online map construction from sensor observations. |
| [MapTR](https://arxiv.org/abs/2208.14437) |  | Vectorized map element prediction from BEV features. Highly relevant to planning. |

Skim objective: compare dense BEV grids, BEV queries, depth-based lifting, temporal fusion, and vectorized map outputs.

---

### 3.11 SLAM and Odometry

| Paper | Label | Why skim it |
|---|---|---|
| [ORB-SLAM2](https://arxiv.org/abs/1610.06475) |  | Canonical feature-based visual SLAM system: tracking, mapping, loop closure, relocalization. |
| [DeepVO](https://arxiv.org/abs/1709.08429) |  | Early end-to-end visual odometry from raw image sequences. |
| [RAFT](https://arxiv.org/abs/2003.12039) |  | Strong optical flow baseline; useful for dense video motion reasoning. |
| [DROID-SLAM](https://arxiv.org/abs/2108.10869) |  | Learned features plus dense bundle adjustment; bridges classical SLAM and learned optimization. |

Skim objective: understand pose estimation, ego-motion, optical flow, bundle adjustment, and why temporal alignment matters in BEV.

---

### 3.12 Reconstruction

| Paper | Label | Why skim it |
|---|---|---|
| [Structure-from-Motion Revisited / COLMAP](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html) |  | Practical standard for incremental SfM. |
| [MVSNet](https://arxiv.org/abs/1804.02505) |  | Differentiable homography warping and cost-volume-based multi-view stereo. |
| [DUSt3R](https://arxiv.org/abs/2312.14132) |  | Dense 3D reconstruction without requiring a traditional calibration-first pipeline. |
| [MASt3R](https://arxiv.org/abs/2406.09756) |  | 3D-grounded image matching built on the DUSt3R direction. |

Skim objective: know the difference between sparse SfM, dense MVS, and newer transformer-based 3D reconstruction/matching.

---

### 3.13 NeRF and Gaussian Splatting

| Paper | Label | Why skim it |
|---|---|---|
| [NeRF](https://arxiv.org/abs/2003.08934) | **[CV LANDMARK]** | Differentiable volume rendering and neural scene representation. |
| [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | **[CV LANDMARK]** | Real-time high-quality neural rendering with explicit 3D Gaussians. |
| [DriveDreamer](https://arxiv.org/abs/2309.09777) |  | Connects generative driving videos and controllable autonomous-driving scene generation. |
| [GAIA-1](https://arxiv.org/abs/2309.17080) |  | Driving world model using video, text, and action conditioning. |

Skim objective: do not over-focus on rendering. For AD, the important question is how scene representations support simulation, data generation, and future prediction.

---

### 3.14 Motion Forecasting

| Paper | Label | Why skim it |
|---|---|---|
| [VectorNet](https://arxiv.org/abs/2005.04259) |  | Vectorized map and trajectory representation instead of rasterized scene images. |
| [LaneGCN](https://arxiv.org/abs/2007.13732) |  | Lane graph and actor-map interaction. |
| [Wayformer](https://arxiv.org/abs/2207.05844) |  | Transformer architecture for multi-agent motion forecasting. |
| [MTR](https://arxiv.org/abs/2209.13508) |  | Motion Transformer with global intention localization and local movement refinement. |

Skim objective: understand multimodality, map conditioning, agent interaction, and why forecasting is not just trajectory regression.

---

### 3.15 Planning and Control

| Paper | Label | Why skim it |
|---|---|---|
| [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410) |  | Early command-conditioned vision-based driving policy. |
| [ChauffeurNet](https://arxiv.org/abs/1812.03079) |  | Shows why pure behavior cloning needs perturbation, bad-case synthesis, and auxiliary losses. |
| [TCP](https://arxiv.org/abs/2206.08129) |  | Combines trajectory prediction and direct control prediction. |
| [PlanT](https://arxiv.org/abs/2210.14222) |  | Object-level transformer planning; useful contrast against dense BEV planning. |

Skim objective: track the difference between direct control, trajectory planning, behavior cloning, and planning under distribution shift.

---

### 3.16 Robot Learning

| Paper | Label | Why skim it |
|---|---|---|
| [DAgger](https://arxiv.org/abs/1011.0686) |  | Core imitation learning paper for covariate shift. Highly relevant to E2E driving failure modes. |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) |  | Models multimodal action distributions with diffusion; relevant to policy generation. |
| [RT-1](https://arxiv.org/abs/2212.06817) |  | Scales robot policy learning with large datasets and transformers. |
| [RT-2](https://arxiv.org/abs/2307.15818) |  | Vision-language-action direction; useful for language-conditioned robotics and driving discussions. |

Skim objective: understand covariate shift, action multimodality, and large-scale policy learning.

---

### 3.17 End-to-End Autonomous Driving

| Paper | Label | Why skim it |
|---|---|---|
| [ST-P3](https://arxiv.org/abs/2207.07601) |  | Interpretable vision-based E2E AD with spatial-temporal feature learning. |
| [UniAD](https://arxiv.org/abs/2212.10156) |  | Planning-oriented framework unifying detection, tracking, mapping, motion forecasting, occupancy, and planning. |
| [VAD](https://arxiv.org/abs/2303.12077) |  | Vectorized scene representation for efficient E2E planning. |
| [SparseDrive](https://arxiv.org/abs/2405.19620) |  | Sparse scene representation for unified detection, tracking, mapping, prediction, and planning. |
| [GenAD](https://arxiv.org/abs/2402.11502) |  | Casts E2E autonomous driving as a generative modeling problem over future evolution. |

Skim objective: identify the representation, task decomposition, training losses, planning metric, and open-loop vs closed-loop evaluation.

---

### 3.18 Generative Vision and World Models

| Paper | Label | Why skim it |
|---|---|---|
| [DDPM](https://arxiv.org/abs/2006.11239) | **[CV LANDMARK]** | Core modern diffusion model formulation. |
| [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | **[CV LANDMARK]** | Makes high-quality text/image-conditioned diffusion practical in latent space. |
| [GAIA-1](https://arxiv.org/abs/2309.17080) |  | Controllable generative world model for autonomous driving. |
| [DriveDreamer](https://arxiv.org/abs/2309.09777) |  | Driving video generation conditioned by structured controls. |
| [GenAD](https://arxiv.org/abs/2402.11502) |  | Connects generative modeling directly to E2E driving planning. |

Skim objective: separate three use cases: data generation, future world prediction, and direct planning through generated futures.

---

### 3.19 Robustness, Uncertainty, and Safety

| Paper | Label | Why skim it |
|---|---|---|
| [MC Dropout](https://arxiv.org/abs/1506.02142) |  | Classic approximate Bayesian uncertainty estimate for neural networks. |
| [Deep Ensembles](https://arxiv.org/abs/1612.01474) |  | Simple and strong practical uncertainty baseline. |
| [ImageNet-C / ImageNet-P](https://arxiv.org/abs/1903.12261) |  | Corruption and perturbation robustness benchmark for vision. |
| [Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?](https://arxiv.org/abs/2006.14911) |  | Directly frames distribution shift and adaptation for autonomous vehicles. |
| [InterFuser](https://arxiv.org/abs/2207.14024) |  | Adds interpretable intermediate features and safety-aware reasoning to E2E driving. |

Skim objective: understand epistemic vs aleatoric uncertainty, OOD detection, corruption robustness, and the gap between benchmark scores and deployment safety.

---

## 4. Ultra-Conservative CV Landmark Shortlist

These are the papers from the roadmap that are most defensible as broad computer-vision landmarks. This list is intentionally strict.

| Area | Landmark papers |
|---|---|
| Deep visual recognition | [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), [ResNet](https://arxiv.org/abs/1512.03385) |
| Object detection | [R-CNN](https://arxiv.org/abs/1311.2524), [Faster R-CNN](https://arxiv.org/abs/1506.01497), [YOLO](https://arxiv.org/abs/1506.02640), [DETR](https://arxiv.org/abs/2005.12872) |
| Segmentation | [FCN](https://arxiv.org/abs/1411.4038), [U-Net](https://arxiv.org/abs/1505.04597), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Segment Anything](https://arxiv.org/abs/2304.02643) |
| Geometry and 3D vision | [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), [PointNet](https://arxiv.org/abs/1612.00593), [NeRF](https://arxiv.org/abs/2003.08934), [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) |
| Foundation and generative vision | [Vision Transformer](https://arxiv.org/abs/2010.11929), [CLIP](https://arxiv.org/abs/2103.00020), [DDPM](https://arxiv.org/abs/2006.11239), [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) |

Important distinction: BEVFormer, Lift-Splat-Shoot, BEVDepth, UniAD, VAD, SparseDrive, and GenAD are highly important for camera-based autonomous driving, but they are not marked as broad CV landmarks here because their influence is more domain-specific.

---

## 5. One-Pass Minimal Route

If you want the shortest coherent path, skim in this order:

1. [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) plus selected chapters from [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/).
2. [Monodepth2](https://arxiv.org/abs/1806.01260) and [Depth Anything V2](https://arxiv.org/abs/2406.09414).
3. [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711), [BEVDepth](https://arxiv.org/abs/2206.10092), and [BEVFormer](https://arxiv.org/abs/2203.17270).
4. [MapTR](https://arxiv.org/abs/2208.14437), [SurroundOcc](https://arxiv.org/abs/2303.09551), and [Occ3D](https://arxiv.org/abs/2304.14365).
5. [VectorNet](https://arxiv.org/abs/2005.04259), [LaneGCN](https://arxiv.org/abs/2007.13732), and [MTR](https://arxiv.org/abs/2209.13508).
6. [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410), [ChauffeurNet](https://arxiv.org/abs/1812.03079), and [PlanT](https://arxiv.org/abs/2210.14222).
7. [ST-P3](https://arxiv.org/abs/2207.07601), [UniAD](https://arxiv.org/abs/2212.10156), [VAD](https://arxiv.org/abs/2303.12077), [SparseDrive](https://arxiv.org/abs/2405.19620), and [GenAD](https://arxiv.org/abs/2402.11502).
8. [DDPM](https://arxiv.org/abs/2006.11239), [Latent Diffusion Models](https://arxiv.org/abs/2112.10752), [GAIA-1](https://arxiv.org/abs/2309.17080), and [DriveDreamer](https://arxiv.org/abs/2309.09777).
9. [Deep Ensembles](https://arxiv.org/abs/1612.01474), [ImageNet-C / ImageNet-P](https://arxiv.org/abs/1903.12261), and [InterFuser](https://arxiv.org/abs/2207.14024).

This route prioritizes the actual camera-based AD stack over general CV history.

---

## 6. Practical Skimming Checklist

For each paper, extract only the following:

1. **Input**: single camera, multi-camera, video, LiDAR, map, route command, ego state, or text/action conditioning.
2. **Representation**: image feature, depth distribution, BEV grid, sparse query, vector map, occupancy, latent token, trajectory, or policy.
3. **Temporal modeling**: none, recurrent BEV, transformer attention, tracking memory, world model, or rollout.
4. **Learning signal**: detection loss, depth loss, segmentation loss, forecasting loss, imitation loss, planning loss, RL signal, or generative likelihood.
5. **Evaluation**: open-loop metric, closed-loop metric, collision rate, route completion, comfort, mAP/NDS, minADE/minFDE, IoU, or calibration/OOD metric.
6. **Failure mode**: depth ambiguity, occlusion, long-tail objects, multi-agent interaction, compounding error, covariate shift, distribution shift, or unsafe open-loop optimization.


## References
- hoya012/deep_learning_object_detection https://github.com/hoya012/deep_learning_object_detection
- worldbench/awesome-3d-in-the-wild https://github.com/worldbench/awesome-3d-in-the-wild
- ZHOUYI1023/awesome-radar-perception https://github.com/ZHOUYI1023/awesome-radar-perception
- Radar-Camera-Fusion/Awesome-Radar-Perception https://github.com/Radar-Camera-Fusion/Awesome-Radar-Perception
- changh95/visual-slam-roadmap https://github.com/changh95/visual-slam-roadmap
- OpenDriveLab/End-to-end-Autonomous-Driving https://github.com/opendrivelab/end-to-end-autonomous-driving