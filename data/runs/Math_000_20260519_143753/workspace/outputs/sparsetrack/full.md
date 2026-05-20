# Sparse Track: Multi-Object Tracking by Performing Scene Decomposition based on Pseudo-Depth

Zelin Liu, Xinggang Wang, Cheng Wang, Wenyu Liu, Xiang Bai

A—. u MOT17 and MOT20 benchmarks. Code and models are publicly available at https://github.com/hustvl/SparsTrack.

Index Terms—2D multiobject tracking, occluded object tracking, scene decomposition, and data association.

## 1 INTRODUCTION

M in fields such as autonomous driving, surveillance, and intelligent transportation. It aims to consistently identify the same object in different video frames as the same identity in the form of bounding boxes. Although previous trackers have achieved high performance on multiple tracking datasets [2], [3], [4], dense crowds and frequent occlusions still make multi-object tracking tasks challenging.

Current mainstream tracking methods follow the paradigm of tracking-by-detection (TBD) [5] and perform frame-by-frame data association. In order to solve the obstacle of occlusion association in dense scenes, some simple methods, such as ByteTrack [6], have achieved effective tracking of occluded targets in dense scenes by separately associating low-score detections. Although ByteTrack demonstrates proficiency in processing low-score detections separately, its accuracy in location association is prone to deterioration in scenes characterized by a high volume of low-score occlusions or frequent overcrowding, as shown in Fig. 1. Other methods [7], [8], [9], [10], [11], [12] ensure tracking performance of occluded instances by using powerful temporal modeling and trajectory query mechanisms. However, these methods are typically associated with high computational costs, particularly in scenes populated by a multitude of objects and frequent occlusions.

In this work, we prove that the target set decomposition based on depth information is an effective approach

![](images/f093e9f9f20b2bbd1aed9aa7d35beeac547eeff1f2c165c79872242a0f69c2d4.jpg)

Fig. 1. An illustration of associating low-score detections in crowded scenes. In ByteTrack [6], a set of low-score detections are matched with the track set at the same time using the loU metric, which is easy to make mistakes due to the high location similarity between the low-score detections. This paper attempts to solve this problem from the low-score detection decomposition perspective rather than matching them at the same time.

for dealing with dense occlusions in data association. As illustrated in Fig. 2, we show an occlusion order for the objects in a local region and the occlusion order is consistent with the depth order, from near to far. Thus, a set of dense occlusions can be divided into several non-overlapping subsets by utilizing a segmentation strategy based on depth information. Between adjacent target subsets, their x-y locations could be similar, but their x-y-depth locations can be much easier to distinguish. The tracker performs data association separately for sparse subsets at different depth levels. Compared to directly associating the entire occluded object set at the same time, the sparse decomposition would be more effective for alleviating the collision probability of trajectories with similar positions but different depths during data association. Specifically, we prioritize the association of target subsets with smaller depths since the occlusion order is highly correlated with the depth order. As a result, the targets in each subset can be handled in a fine-grained manner and be less affected by the targets in other depth levels.

![](images/c8d558b16abab5f9287608963d3350ca0487288b2a207f83e5562130e4665aa9.jpg)

To realize the above designs, we propose a method for obtaining the relative depth of targets from 2D images: the pseudo-depth method, which is based on two scene priors: 1) the camera that captures objects is higher than the ground, 2) all objects in the scene are on flat ground. In other words, there are no obvious undulations on the ground. In this case, we can project the relative depth of targets from 3D space onto the 2D image plane and obtain target pseudodepth values, which is the distance from the bottom of the target bounding box to the bottom edge of the image. Notably, the pseudo-depth value is used as a reference to measure the relative depth relationship between targets using the ground as a reference system, rather than the ground truth depth of the target in 3D space. Furthermore, we design a depth cascade matching (DCM) algorithm to execute hierarchical association based on the target pseudodepth information. Specifically, we divide trajectories and detections into multiple target subsets according to the distribution of pseudo-depth value. The DCM algorithm performs IoU [13] association on these sparse target subsets in the order of pseudo-depth value from nearest to farthest. By integrating the pseudo-depth method and DCM into the data association process, we propose a novel tracker called SparseTrack. The essence of SparseTrack lies in associating occlusions hierarchically based on depth via the DCM, as shown in Fig. 3.

SparseTrack achieves impressive performance on multiple tracking datasets, which proves the effectiveness of target set decomposition based on pseudo-depth. We take ByteTrack as a baseline. On the MOT17 [3] test set, Sparse-Track achieved 65.1 HOTA [14], 81.0 MOTA, and 80.1 IDF1, which are gains of +2.0 HOTA, +0.7 MOTA, and +2.8 IDF1 compared to the baseline. On the MOT20 [4] dataset, SparseTrack achieved 63.4 HOTA, 78.2 MOTA, and 77.3 IDF1, which are gains of +2.1 HOTA, +0.4 MOTA, and +2.1 IDF1 compared to the baseline. Furthermore, we evaluate SparseTrack on DanceTrack [15] benchmark and obtain a gain of +7.8 HOTA, +1.7 MOTA, +4.4 IDF1 compared to the baseline. It is worth noting that our proposed DCM algorithm is plug-and-play and can be integrated into different trackers, resulting in consistent performance improvements.

Fig. 3. An simple example of the hierarchical association from DCM, where the white square regions indicate that no data association is performed between the corresponding detection and trajectory subsets along both horizontal and vertical directions. The darkness of the square regions reflects the similarity between trajectories and detections, with darker colors indicating higher similarity. DCM decomposes the target set into multiple subsets according to depth order from near to far and performs data association on the detection and trajectory subsets at the same depth level. For the unmatched trajectories and detections from each depth level, the tracker will process them at the next depth level.

The specific details are discussed in the experimental section.

Our contributions are summarized as follows:

•We propose a method for obtaining the relative depth of targets from 2D images: the pseudo-depth method, which is based on two prior in the scene, and can effectively obtain the pseudo-depth value of the target to compare the relative depth relationships between different objects.

• Based on the depth information provided by the pseudo-depth method, we design an effective depth cascade matching approach for associating occlusions in dense scenes. It can decompose dense target sets into multiple sparse target subsets to achieve scene decomposition.

Based on the aforementioned design, we propose a new IoU-only tracker named as SparseTrack, which significantly outperforms the previous IoU-only trackers and achieves comparable results with recent state-of-the-art MOT methods on a wide range of MOT benchmarks.

## 2 RELATED WORK

## 2.1 2D Multi-Object Tracking

The current mainstream work on 2D multi-object tracking focuses on ensuring robust temporal associations, which can lead to the development of various tracking methods. Based on different temporal modeling methods, current mainstream tracking approaches can be roughly divided into four categories: temporal associations via positional information, temporal associations via appearance features, temporal associations via graph optimization, and temporal associations via attention mechanisms. Early trackers [5], [16], based on deep learning perform temporal associations between consecutive frames via positional information and employ the Kalman filter [17] based on constant velocity motion priors to model the inter-frame motion of the targets, which can ensure the stability of the trajectories. To achieve the long-term association, DeepSORT [18] and MOTDT [19] enhance SORT [5] by adding appearance cues to compensate for the defect of positional information in associating long-term trajectories. Although appearance cues generated by re-identification (ReID) models [20], [21], [22], [23] are effective for associating long-term objects, challenges are posed by target occlusions and motion blurs to the reliability of these cues. Some methods [24], [25], [26], [27], [28] use the ReID models enhanced by self-supervised pretraining [29], [30], [31], [32], [33] to alleviate the poor appearance of occluded targets, but it comes with a higher computational cost. Other trackers, such as [34], [35], [36], improve computational efficiency by jointly optimizing the detection task and the re-identification task within a single framework, which somewhat limits the upper bound of detection performance and appearance quality. Partial works [7], [8], [11], [12], [37] are inspired by Transformer [38], [39], [40], [41] and attempt to implement trajectory temporal propagation from the perspective of attention mechanism and achieve impressive performance in complex scenes [15], [42], While the computational overhead generated by selfattention and feed-forward networks is still non-negligible. Although the methods [43], [44], [45], [46], [47] offer computationally efficient trajectory temporal propagation based on convolutional neural network (CNN) [48], they struggle with the association problem for crowded targets. Recent works [9], [49], [50], [51], [52], [53], formulate multiple object tracking as a graph optimization problem, using graph neural networks or graph matching to achieve robust data association. However, the computational overhead based on graphs tends to increase square fold when the number of targets in the scene increases.

## 2.2 Methods for Handling Object Occlusion

The performance of a tracker is to some extent dependent on the ability to handle occlusions. Recent works have attempted to tackle occlusion from different perspectives. For example, MotionTrack [54] learns the motion pattern of trajectories and combines it with historical information to effectively model tracks of occlusions. MOTR [11], an endto-end MOT framework, processes new appearing targets and tracked targets separately using detection queries and trajectory queries, respectively, with a multi-frame training manner. Due to the relative independence among queries and sufficient temporal training, MOTR can perform well in temporal modeling of occlusions across time steps. DP-MOT [55] proposes a subject-ordered depth estimation (SODE) method to automatically sort the depth positions of detected objects in 2D scenes in an unsupervised manner. By constructing a pseudo-3D Kalman filter, DP-MOT achieves robust association for occluded targets. BoT-SORT [56] combines camera motion compensation (CMC) and IoU-ReID fusion, which can integrate motion and appearance clues of the object for achieving accurate tracking of occlusions. OUTrack [28] uses an unsupervised re-identification module and occlusion-aware module to predict the locations where target occlusions occur, in order to compensate for missed detections. ApLift [57] introduces an advanced approximate solver for tackling the disjoint path problem, specifically engineered to manage extended and congested trajectory sequences, with performance comparable to mainstream tracking methods. [58] segment the trajectories of pairwise occluded targets and recalculates trajectory similarities to effectively associate occluded targets. SparseTrack provides a completely different solution for associating occlusions by performing the target set decomposition to align the location interval of occlusions.

## 2.3 Targets Set Decomposition

Actually, many trackers that follow the tracking-bydetection paradigm perform some degree of the target set decomposition. For example, DeepSORT [18] achieves stepby-step association of the detection set through multi-stage cascaded matching. FairMOT [35] processes the detection set separately based on appearance and position clues, which can also be seen as a division of the detection set via tracking clues. LMGP [59] effectively eliminates trajectory errors in single-camera trackers by introducing a pre-clustering method driven by 3D geometric projections, which groups detected objects for association. ByteTrack [6] divides the detection set into high-score and low-score detections based on confidence scores, and uses the correlation between scores and occlusions to separately process lowscore occluded targets. However, in dense crowds, congestion leads to occlusions, which results in low-score detections. Although ByteTrack decouples low-score occlusions from the scene, low-score targets are still crowded. In this case, IoU-based data association is prone to matching errors that limit the tracking ability to handle occlusions. To this end, SparseTrack divides the detection set and the trajectory set based on the pseudo-depth level to ensure that targets in each association step are no longer crowded.

## 2.4 The Applications of Depth Information in MOT

Depth information is commonly used in applications related to 3D scenes, such as monocular or stereo depth estimation, 3D detection and tracking, and recently popular neural radiance fields [60]. The standard depth provides abundant information on the appearance and position of objects, leading to different 3D tracking methods. For instance, TrackR-CNN [61] uses appearance and motion features for tracking and enhances accuracy and stability by alternately tracking objects in 2D images and 3D point clouds. FANTrack [62] employs a feature-based association network (FAN) for object tracking, where features come from a 3D convolutional neural network (CNN) [48] for point cloud data and a 2D CNN for camera images. CenterPoint [63] uses a keypoint detector to detect the center of an object and regress to other attributes, including 3D size, 3D direction, and speed. Then, it performs 3D object tracking via simple greedy nearest point matching. In fact, a 2D image can be considered as the projection of a 3D scene under perspective transformation. According to the camera model, we can easily infer the variable relationship between a 2D image and a 3D scene. SparseTrack leverages this variable relationship to obtain the pseudo-depth values of targets in 2D image, which partially replace the role of standard depth in describing the position relationships among objects. To the best of our knowledge, SparseTrack is the first method that utilizes depth information to decompose targets set.

![](images/60bc02357258a81c0913e13c897af87476f43285608584698920249d4f9f74b6.jpg)  
depth method.

![](images/5f318a8e5f0a260097e00a55a6ab46716f8c30ce9d9847570eee43c7568d2742.jpg)  
Fig. 5. The illustration of pseudo-depth method. The pseudo depth value is equal to the Euclidean distance between $P _ { 1 }$ and D. In fact, pseudodepth can be regarded as a projection that can be obtained by projecting the distance between $P _ { 0 }$ and $D _ { 0 }$ in 3D space onto the image plane. In the projection transformation, since the Euclidean distance between $P _ { 0 }$ and $D _ { 0 }$ is positively correlated with the pseudo-depth value, the pseudodepth can reflect the depth of the target in the scene.

## 3 METHOD

## 3.1 General Framework

SparseTrack follows the paradigm of tracking-by-detection and performs frame-by-frame data association. The overall framework is shown in Fig. 4. Given a frame $f _ { i }$ in a video sequence, it is first processed by the YOLOX [64] detector, which outputs detection results in the form of bounding boxes $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ and confidence scores s. Then based on each detected bounding box and input image size, we use the pseudo-depth method to obtain pseudo-depth values of targets in the 2D image. The specific details are described in Sec. 3.2. Pseudo-depth values can reflect the relative depth relationship between targets in the image. In the data association process, we utilize the pseudo-depth values of the detection and trajectory instances and use the DCM algorithm described in Sec. 3.3 to accomplish the decomposition and association for the tracks and the detections, separately. Finally, SparseTrack outputs tracking results as $( x _ { 1 } , y _ { 1 } , w , h , s )$ , where the coordinates (w, h) represent the width and height of instance bounding boxes.

## 3.2 Pseudo-Depth Method

The enabled conditions of pseudo-depth method is based on two prior assumptions in general tracking scenes: the camera that captures objects is above the ground and all objects in the scene are on the same plane. These assumptions are easily satisfied in general scenarios. Specifically, pseudodepth method is a simple geometric method for obtaining pseudo-depth values. Firstly, we acquire the position point $P _ { 0 }$ where the object in 3D space contacts the ground, project point $P _ { 0 }$ onto the image plane $F$ of the camera, and obtain the corresponding projection point $P _ { 1 }$ .Secondly, we draw a perpendicular from point $\hat { P } _ { 1 }$ to the bottom edge $L _ { b }$ of the image and obtain the intersection point D between the perpendicular and the bottom edge $L _ { b }$ We define the Euclidean distance between the projection point $P _ { 1 }$ and the intersection point $D$ as the pseudo-depth value of objects, as shown in Fig. 5. Set the height of the image to $H ,$ The pixel coordinates of $P _ { 1 }$ is denoted as $( x _ { p } , y _ { p } )$ . The pseudo-depth is computed as:

$$
\begin{array} { r } { L _ { p } = H - y _ { p } , } \end{array}\tag{1}
$$

where $L _ { p }$ is the pseudo-depth value. Although pseudodepth values can not represent the ground truth depth of the target in 3D space, it is able to sufficiently measure the relative depth relationship between different objects on the same ground, which also provides a basis for dense crowd division at the depth level.

## 3.3 Depth Cascade Matching

With the depth information obtained by the pseudo-depth method, we can divide targets in the scene based on their pseudo-depth values, resulting in a sparser distribution of previously dense crowds. Firstly, we compute the minimum pseudo-depth value of the detection set $D _ { d - m i n }$ and the maximum pseudo-depth value of the detection set $D _ { d - m a x }$ . Then we uniformly divide the interval distance between $D _ { d - m i n }$ and $D _ { d - m a x }$ into k depth intervals $\{ I _ { 0 } , I _ { 1 } , . . . , I _ { k - 1 } \} _ { d e t } ,$ which serve as k different depth levels. With the same method, we can also obtain the minimum pseudo-depth value of the trajectory set $D _ { t - m i n } ,$ maximum pseudo-depth value of the trajectory set $D _ { t - m a x } ,$ and k depth intervals $\{ \hat { I } _ { 0 } , I _ { 1 } , . . . , I _ { k - 1 } \} _ { t r a c k }$ . Based on the detection depth intervals $\{ I _ { 0 } , I _ { 1 } , . . . , I _ { k - 1 } \} _ { d e t }$ and the track depth intervals $\{ I _ { 0 } , I _ { 1 } , . . . , I _ { k - 1 } \} _ { t r a c k } ,$ we separately obtain the trajectory subsets $T _ { s u b }$ and detection subsets $D _ { s u b }$ that are located at different depth levels according to pseudo-depth values. Subsequently, the tracker performs IoU association $\mathcal { T }$ between the trajectory and detection subsets at the same depth level. Unmatched trajectories $\mathcal { T } _ { 0 }$ and detections $\mathcal { D } _ { 0 }$ will participate in the association process of the next depth level after the data association at each depth level. The pseudo-code of the depth cascade matching algorithm is shown in Algorithm 1.

## 3.4 The Association Process of SparseTrack

We propose a simple yet powerful data association method. Unlike other methods [18], [25], [35], [36], [44], [65], [66] that focus on improving the appearance features and motion cues to ensure the reliability of association, SparseTrack divides all targets in the scene into multiple sparse target subsets by the pseudo-depth method and DCM to effectively associate occlusions in crowds. We divide the detection set into high-score and low-score detection subsets based on confidence scores via [6]. Then, we use the aforementioned depth cascaded matching algorithm to perform data association on the high-score detection subset. For trajectories that appear in the previous frame but are not matched, we associate them with the low-score detection subset via the depth cascade matching algorithm. For unmatched highscore detections, we associate it with newly appeared trajectories in the previous frame. Finally, unmatched highscore detections are initialized as new trajectories, while lost trajectories that exceed the max number of losing frames are removed. The pseudo-code of data association of Sparse-Track is shown in Algorithm 2.

Algorithm 1: Depth Cascade Matching.   
Input: tracks set $\mathrm { T } ;$ detections set $\mathrm { D } ;$ the number of depth levels   
k; Hungarian matching H; the function of IoU distance   
metric $\breve { \tau } ;$ the function of getting max pseudo-depth   
value $\mathcal { M } _ { m a x } ;$ the function of getting min pseudo-depth   
value $\mathcal { M } _ { m i n } ;$ initialize matching threshold T   
Output: Matched tracks $\tau { } _ { ; }$ Unmatched tracks $\tau _ { 0 } ;$ Umatched   
detections Do   
1 Initialization: $\mathcal { T }  \emptyset , \mathcal { T } _ { 0 }  \emptyset , \mathcal { D } _ { 0 }  \emptyset$   
/\* generate sparse subsets of detections $\star /$   
2 $D _ { d - m a x }  \mathcal { M } _ { m a x } ( \mathrm { D } )$   
3 $D _ { d - m i n }  \mathcal { M } _ { m i n } ( \mathrm { D } )$   
4 $\{ I _ { 0 } , . . . , I _ { k - 1 } \} _ { d e t } \gets s p l i t ( [ D _ { d - m i n } , D _ { d - m a x } ] , k )$   
5 $D _ { s u b } \gets \emptyset / \star \mathsf { \ a l l }$ of subsets of detections $\star /$   
6 for $\{ I _ { i } \} _ { d e t } i n \left\{ I _ { 0 } , . . . , I _ { k - 1 } \right\} _ { d e t }$ do   
7 $D _ { i } \gets \emptyset$   
8 for d in D do   
9 if d.depth  {Ii}det then   
10 $\begin{array} { r l } { \Big | } & { { } D _ { i }  D _ { i } \cup d } \end{array}$   
11 end   
12 end   
13 $D _ { s u b } \gets D _ { s u b } \cup D _ { i }$   
14 end   
/\* generate sparse subsets of tracks $\star /$   
15 $D _ { t - m a x }  \mathcal { M } _ { m a x } ( \mathrm { T } )$   
16 $D _ { t - m i n } \gets \mathcal { M } _ { m i n } ( \mathbb { T } )$   
17 $\{ I _ { 0 } , . . . , I _ { k - 1 } \} _ { t r a c k } \gets s p l i t ( [ D _ { t - m i n } , D _ { t - m a x } ] , k )$   
18 $T _ { s u b } \gets \emptyset / \star \mathsf { \ a l l }$ of subsets of tracks $\star /$   
19 for $\{ I _ { i } \} _ { t r a c k }$ in $\{ I _ { 0 } , . . . , I _ { k - 1 } \} _ { t r a c k }$ do   
20 $T _ { i } \gets \emptyset$   
21 for t in T do   
22 if t.depth $\in \{ I _ { i } \} _ { t r a c k }$ then   
23 $T _ { i } \gets T _ { i } \cup t$   
24 end   
25 end   
26 $T _ { s u b } \gets T _ { s u b } \cup T _ { i }$   
27 end   
/\* depth cascade matching \*/   
28 for track subset $T _ { i } ,$ detection subset $D _ { i }$ in $( T _ { s u b } , D _ { s u b } )$ do   
$/ \star$ unmatched objects from previous stage   
participate the association of current   
stage. \*/   
29 $T _ { i } \gets T _ { i } \cup \mathcal { T } _ { 0 }$   
30 $D _ { i }  D _ { i } \cup \mathcal { D } _ { 0 }$   
$/ \star$ get cost matrix based on IoU distance $\star /$   
31 $C _ { i } \gets \mathbb { Z } ( T _ { i } , D _ { i } )$   
/\* association \*/   
32 $T _ { m a t c h e d } , T _ { 0 } , \mathcal { D } _ { 0 } \gets \mathcal { H } ( C _ { i } , \tau )$   
/\* add matched tracks $\star /$   
33 $\tau  \tau \cup T _ { m }$ atched   
34 end   
35 Return: $\tau , \tau _ { 0 } , \mathcal { D } _ { 0 }$

As the confidence score is related to the occlusion of the target, we inherit the decomposition method based on the confidence score from [6]. Since high-score detections often mean a lower degree of occlusion, we set the number of pseudo-depth levels to 1 or 2 during associating high-score objects. Conversely, for associating low-score detections, we set the number of pseudo-depth levels to 4 or 8, which helps in the fine-grained association of dense occlusions. It is worth mentioning that the DCM is plug-and-play and can be integrated into various trackers. The specific settings are detailed in Sec. 4.

Algorithm 2: The Data Association of SparseTrack.   
Input: A video sequence V; object detector Det; high-score   
detection threshold τ; the function of depth cascade   
matching DCM; the Kalman motion model KF   
Output: tracking results of the video T   
1 Initialization: $\tau  \emptyset$   
2 for frame $f _ { i }$ in V do   
/\* get high-score and low-score detections   
from current frame \*/   
3 $D _ { i }  \mathsf { D e t } ( f _ { i } ) / \star$ detections per frame $\star /$   
4 $D _ { h i g h }  \emptyset$   
5 $D _ { l o w }  \emptyset$   
6 for d in $D _ { i }$ do   
7 if $d _ { s c o r e } > \tau$ then   
8 $D _ { h i g h }  D _ { h i g h } \cup \{ d \}$   
9 end   
10 else   
11 $| \quad \_ D _ { l o w }  D _ { l o w } \cup \{ d \}$   
12 end   
13 end   
$/ \star$ predict the location of tracks from   
previous frames $\star /$   
14 $\mathcal { T }  \mathrm { K F } ( \mathcal { T } )$   
/\* associate high-score detections by DCM   
\*/   
15 Tmatched $D _ { u n m a t c h e d }  \mathsf { D C M } ( \mathsf { T } , D _ { h i g h } )$   
16 Tunmatched $\gets \emptyset$   
17 for t in $( T - T _ { m a t c h e d } )$ do   
18   
19 Tunmatched $ T _ { i }$ unmatched U {t}   
20 end   
21 end   
/\* associate low-score detections by DcM \*/   
22 $T _ { r e - m a t c h e d } \gets \mathsf { D C M } ( T _ { u n m a t c h e d } , D _ { l o w } )$   
$/ \star$ update tracking results \*/   
23 $\mathcal { T } \gets \{ T _ { m a t c h e d } , T _ { r e - m a t c h e d } \}$   
$/ \star$ add new tracks $\star /$   
24 for d in $\mathcal { D } _ { \tau }$ nmatched do   
25 ${ \mathcal { T } } \gets { \mathcal { T } } \cup \{ d \}$   
26 end   
27 end   
28 Return: $\tau$  
Track confirmation is not shown in the pseudo-code for simplicity. Function DCM can refer to Algorithm 1. The key steps of SparseTrack are in green.

## 4 EXPERIMENTS

## 4.1 Setting

## 4.1.1 Datasets

We evaluate SparseTrack on the MOT17 [3], MOT20 [4], and DanceTrack [15] benchmark datasets, and submit the evaluation results to compare the performance with other methods. For ablation experiments, we split half of the training sets of MOT17 and MOT20 (dense scene) as validation sets and conduct experiments on the MOT17 validation set, MOT20 validation set and DanceTrack validation set to analyse the effects with different settings.

## 4.1.2 Metrics

We use the common CLEAR metrics [67] (MOTA, FP, FN, IDs, etc.), IDF1 [68], HOTA [14], AssA and DetA, to evaluate tracking performance. MOTA is computed based on FP, FN, and IDs. As the number of FP and FN is larger than that of IDs, MOTA prefers to reflect detection performance. Additionally, IDF1 and AssA focus more on association performance. DetA is used to measure detection accuracy. HOTA is a comprehensive measure used to evaluate the overall effectiveness of detection and association.

## 4.1.3 Implementation details

During the data association stage, we set different numbers of pseudo-depth levels for different datasets according to the scene crowding level. For MOT17, we set the default number of pseudo-depth levels to 3 and keep the maximum length of lost tracks at 30 frames. For MOT20, we set the default number of depth levels to 8 and keep the maximum length of lost tracks at 60 frames. For DanceTrack, we set the number of depth levels to 12 and keep the lost tracks until 60 frames. The high-score and low-score detection threshold is set to 0.6 and 0.1 by default, separately, which aligns with the baseline approach. To align the positions of detections and tracks on depth levels as closely as possible, we implement a fast online version of global motion compensation (GMC). It is worth noting that all our experimental results rely on IoU distance association and do not use any appearance feature component.

During the inference phase, we configure the input resolution as 800 × 1440 for MOT17 and DanceTrack, while we utilize an input resolution of 896 × 1600 for MOT20. To make a fair comparison with the baseline method [6], we adopt the pre-trained YOLOX [64] detector from ByteTrack and use the same weights and non-maximal suppression (NMS) thresh as the baseline method. For the ablation experiments on the MOT20 validation set, we train SparseTrack on the CrowdHuman and the half of MOT20 train set. For the ablation experiments on the DanceTrack validation set, we adopt publicly available YOLOX detector from [15]. It is worth mentioning that DanceTrack only provides pretrained detectors of the YOLOX-x model. All implementations are performed on NVIDIA GeForce RTX 3090 GPUs.

## 4.2 Evaluation of Different Benchmark

We evaluate SparseTrack on the test set of MOT17, MOT20 and DanceTrack to compare with other methods. All evaluation results are based on the private detection protocol and are shown in Tab. 1, Tab. 2 and Tab. 3, respectively 1

MOT17. By performing the target set decomposition based on pseudo-depth, SparseTrack achieves impressive performance on the MOT17 test set. Compared to the baseline (ByteTrack) with the same pre-trained detector, Sparse-Track achieves gains of +0.7 MOTA, +2.8 IDF1, +2.0 HOTA and IDs decrease to almost twice the original level. Instead of other methods that use appearance features and graph convolutional networks, SparseTrack achieves comparable performance with SOTA using only simple IoU association, which demonstrates that SparseTrack is simple and strong.

TABLE 1  
The comparison of SparseTrack with other methods on the MOT17 test set.
<table><tr><td>Tracker</td><td>|HOTA↑ MOTA↑ IDF1↑</td><td></td><td></td><td>FP↓</td><td>FN↓</td><td></td><td>IDs↓ FPS↑</td></tr><tr><td>enhance motion : Tube_TK [69] CenterTrack [44] TraDes [66] MAT [70]</td><td>48.0 52.2 52.7 53.8</td><td>63.0 67.8 69.1</td><td>58.6 64.7 63.9</td><td>20892 150060 3555</td><td>18498 160332 3039</td><td>27060 177483 4137</td><td>3.0 17.5 17.5</td></tr><tr><td>PermaTrackPr [71] OC-SORT [72] MotionTrack [54]</td><td>55.5 63.2 65.1</td><td>69.5 73.8 78.0 81.1</td><td>63.1 68.9 77.5 80.1</td><td>30660 138741</td><td>28998 115104 15129 107055</td><td>2844 3699 1950 23802 81660 1140</td><td>9.0 11.9 29.0 15.7</td></tr><tr><td>embedding :</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DAN [73]</td><td>39.3</td><td>52.4</td><td>49.5</td><td>25423 234592 8431</td><td></td><td></td><td>&lt;3.9</td></tr><tr><td>QuasiDense [25]</td><td>53.9</td><td>68.7</td><td>66.3</td><td></td><td></td><td>26589 146643 3378</td><td>20.3</td></tr><tr><td>SOTMOT [74]</td><td>-</td><td>71.0</td><td>71.9</td><td>39537 118983</td><td></td><td>5184</td><td>16.0</td></tr><tr><td>Semi-TCL [27]</td><td>59.8</td><td>73.3</td><td>73.2</td><td>22944 124980</td><td></td><td>2790</td><td>-</td></tr><tr><td>FairMOT [35]</td><td>59.3</td><td>73.7</td><td>72.3</td><td>27507 117477</td><td></td><td>3303</td><td>25.9</td></tr><tr><td>CSTrack [36]</td><td>59.3</td><td>74.9</td><td>72.6</td><td>23847 114303 3567</td><td></td><td></td><td>15.8</td></tr><tr><td>SiamMOT [45]</td><td></td><td>76.3</td><td>72.3</td><td></td><td></td><td></td><td>12.8</td></tr><tr><td>ReMOT [75]</td><td>59.7</td><td>77.0</td><td>72.0</td><td>33204</td><td>93612</td><td>2853</td><td>1.8</td></tr><tr><td>StrongSORT [65]</td><td>64.4</td><td>79.6</td><td>79.5</td><td>27876</td><td>86205</td><td>1194</td><td>7.1</td></tr><tr><td>BoT-SORT-ReID [56]</td><td>65.0</td><td>80.5</td><td>80.2</td><td>22521</td><td>86037</td><td>1212</td><td>4.5</td></tr><tr><td>attention :</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CTracker [47]</td><td>49.0</td><td>66.6</td><td>57.4</td><td>22284 160491</td><td></td><td></td><td>6.8</td></tr><tr><td>TransCenter [7]</td><td>54.5</td><td>73.2</td><td>62.2</td><td>23112</td><td>123738</td><td>5529</td><td>1.0</td></tr><tr><td>RelationTrack [76]</td><td>61.0</td><td>73.8</td><td>74.7</td><td>27999</td><td></td><td>4614</td><td>8.5</td></tr><tr><td>TransTrack [8]</td><td>54.1</td><td>75.2</td><td>63.5</td><td>50157</td><td>118623</td><td>1374</td><td>10.0</td></tr><tr><td>MOTRv2 [12]</td><td></td><td></td><td>75.0</td><td>23409</td><td>86442</td><td>3603</td><td>7.5</td></tr><tr><td>P3AFormer [10]</td><td>62.0 -</td><td>78.6 81.2</td><td>78.1</td><td>17281</td><td>94797 86861</td><td>2619 1893</td><td>-</td></tr><tr><td>graphical &amp;</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>correlation :</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GSDT [53]</td><td>55.2</td><td>73.2</td><td>66.5</td><td></td><td></td><td>26397 120666 3891</td><td>4.9</td></tr><tr><td>FUFET [77]</td><td>57.9</td><td>76.2</td><td>68.0</td><td>32796</td><td>98475</td><td>3237</td><td>6.8</td></tr><tr><td>CorrTracker [78]</td><td>60.7</td><td>76.5</td><td>73.6</td><td>29808</td><td>99510</td><td>3369</td><td>15.6</td></tr><tr><td>TransMOT [9]</td><td>61.7</td><td>76.7</td><td>75.1</td><td>36231</td><td>93150</td><td>2346</td><td>9.6</td></tr><tr><td>IoU only :</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ByteTrack [6]</td><td>63.1</td><td>80.3</td><td>77.3</td><td>25491</td><td>83721</td><td>2196</td><td>29.6</td></tr><tr><td>BOT-SORT [56]</td><td>64.6</td><td>80.6</td><td>79.5</td><td>22524</td><td>85398</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>1257</td><td>6.6</td></tr><tr><td>SparseTrack (ours)</td><td>65.1</td><td>81.0</td><td>80.1</td><td>23904</td><td>81927</td><td>1170</td><td>19.9</td></tr></table>

MOT20. Compared to MOT17, MOT20 has denser scenes, more occlusions, and longer videos, which increases the difficulty of the tracker in handling occlusions. Sparse-Track uses the same pre-trained detector as the baseline and achieves a gain of +0.4 MOTA, +2.1 IDF1, and +2.1 HOTA. Compared to other methods that utilize appearance cues, attention mechanism [38], [39] and graph [50], [51], [52], SparseTrack provides a more convenient solution: the target set decomposition based on hierarchical depth. It is worth mentioning that SparseTrack achieves very low IDs and FP, as well as high HOTA, which indicates that the target set decomposition is helpful in associating dense occlusions.

DanceTrack. DanceTrack is a more challenging benchmark for multi-object tracking with frequent occlusions, non-rigid motions and similar appearances. With the same pre-trained detector, SparseTrack achieves significant improvements compared to the baseline with a gain of +7.8 HOTA, +1.7 MOTA, +4.4 IDF1, +7.0 AssA and +7.9 DetA, which demonstrates the enormous potential of depth-based target set decomposition in handling occlusions. Even with simple IoU distance association, SparseTrack still achieves

TABLE 2  
The comparison of SparseTrack with other methods on the MOT20 test set.
<table><tr><td>Tracker</td><td colspan="4">HOTA↑ MOTA↑ IDF1↑ FP↓</td><td colspan="3">FN↓ IDs↓ FPS↑</td></tr><tr><td>enhance motion : MotionTrack [54]</td><td>62.8</td><td>78.0</td><td>76.5</td><td>28629</td><td>84152 1165 15.0</td><td></td><td></td></tr><tr><td>embedding : FairMOT [35]</td><td></td><td></td><td>67.3</td><td></td><td>103440 88901 5243</td><td></td><td> 13.2</td></tr><tr><td>Semi-TCL [27]</td><td>54.6 55.3</td><td>61.8 65.2</td><td>70.1</td><td></td><td>61209 114709 4139</td><td></td><td>-</td></tr><tr><td>CSTrack [36]</td><td>54.0</td><td>66.6</td><td>68.6</td><td></td><td>25404 144358 3196</td><td></td><td>4.5</td></tr><tr><td>SiamMOT [45]</td><td>-</td><td>67.1</td><td>69.1</td><td>-</td><td></td><td>-</td><td>4.3</td></tr><tr><td>SOTMOT [74]</td><td>-</td><td>68.6</td><td>71.4</td><td></td><td>57064 101154 4209</td><td></td><td>8.5</td></tr><tr><td>BoT-SORT-ReID [56]</td><td>63.3</td><td>77.8</td><td>77.5</td><td>24638</td><td>88863 1257</td><td></td><td>2.4</td></tr><tr><td>UTM [26]</td><td>62.5</td><td>78.2</td><td>76.9</td><td>29964</td><td>81516 1228</td><td></td><td></td></tr><tr><td>attention : TransCenter [7]</td><td>-</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>61.9</td><td>50.4</td><td></td><td>45895 146347 4653</td><td></td><td>1.0</td></tr><tr><td>TransTrack [8]</td><td>48.5</td><td>65.0</td><td>59.4</td><td></td><td>27197 150197 3608</td><td></td><td>7.2</td></tr><tr><td>RelationTrack [76]</td><td>56.5</td><td>67.2</td><td>70.5</td><td></td><td>61134 104597 4243</td><td></td><td>2.7</td></tr><tr><td>MOTR [11]</td><td>57.8</td><td>73.4</td><td>68.6</td><td>-</td><td>-</td><td>2439</td><td>-</td></tr><tr><td>P3AFormer [10]</td><td>-</td><td>78.1</td><td>76.4</td><td>25413</td><td>86510 1332</td><td></td><td></td></tr><tr><td>graphical &amp; correlation :</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLT [79]</td><td>43.2</td><td>48.9</td><td>54.6</td><td></td><td></td><td></td><td></td></tr><tr><td>CorrTracker [78]</td><td>-</td><td></td><td></td><td></td><td></td><td>45660 216803 2187</td><td>3.7</td></tr><tr><td>GSDT [53]</td><td>53.6</td><td>65.2</td><td>69.1</td><td></td><td></td><td>79429 95855 5183</td><td>8.5</td></tr><tr><td></td><td></td><td>67.1</td><td>67.5</td><td></td><td>31913 135409 3131</td><td></td><td>0.9</td></tr><tr><td>IoU only :</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ByteTrack [6]</td><td>61.3</td><td>77.8</td><td>75.2</td><td>26249</td><td></td><td>87594 1223</td><td>17.5</td></tr><tr><td>BOT-SORT [56]</td><td>62.6</td><td>77.7</td><td>76.3</td><td>22521</td><td></td><td>86037 1212</td><td>6.6</td></tr><tr><td>SparseTrack (ours)</td><td>63.4</td><td>78.2</td><td></td><td>25108</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>77.3</td><td></td><td></td><td>86720 1116</td><td>12.5</td></tr></table>

TABLE 3

The comparison of SparseTrack with other methods on the Dance Track test set.
<table><tr><td>Tracker HOTA↑ MOTA↑ IDF1↑ AssA↑ DetA↑</td></tr><tr><td>enhance motion :</td></tr><tr><td>CenterTrack [44] 41.8 86.8 35.7 41.2</td></tr><tr><td>TraDes [66] 43.3 86.2 OC-SORT [72] 55.1 92.0</td></tr><tr><td>54.6</td></tr><tr><td>embedding : FairMOT [35] 39.7 82.2 40.8</td></tr><tr><td>FCG* [80] 48.7 89.9 46.5</td></tr><tr><td>QuasiDense [25] 54.2 87.7 50.4</td></tr><tr><td>36.8 80.1 attention :</td></tr><tr><td>TransTrack [8] 45.5 88.4 45.2 27.5 75.9 72.5</td></tr><tr><td>GTR [81] 48.0 84.7 50.3 31.9</td></tr><tr><td>MOTR [11] 54.2 79.7 51.5 40.2</td></tr><tr><td>73.5</td></tr><tr><td>IoU only :</td></tr><tr><td>ByteTrack [6] 47.7 89.6 53.9 32.1 71.0</td></tr><tr><td>BOT-SORT [56] 54.7 91.3 56.0 37.8 79.6 78.9</td></tr><tr><td>SparseTrack (ours) 55.5 91.3 58.3 39.1</td></tr></table>

comparable or even better performance than other methods.

## 4.3 Ablations

## 4.3.1 The impact of the number of pseudo-depth levels

As DCM focuses on addressing dense occlusions, we conduct experiments on the MOT20 validation set and Dance-Track validation set (More crowded scenario and more frequent occlusions.) to better observe the impact of the number of pseudo-depth levels on tracking performance. Specifically, we use the BYTE [6] association method as the baseline and integrate DCM into the low-score association process within BYTE. We set the number of pseudo-depth levels for the process of associating low-score detections to 1, 2, 5, 7, 9, separately. It is worth noting that we lower the detection confidence score from the initial 0.1 to 0.01 to involve more low-scoring detections in the association process. The experimental results are shown in Tab. 4.

TABLE 4  
Comparison of the association performance with different numbers of the pseudo-depth level on MOT20 and Dance Track validation sets.
<table><tr><td rowspan="2">Levels</td><td colspan="3">MOT20 DanceTrack</td></tr><tr><td colspan="3">|HOTA↑AssA↑IDF1↑|HOtA↑AssA↑IDF1↑</td></tr><tr><td>1</td><td>68.6 65.9</td><td>83.0</td><td>52.6 36.5 54.2</td></tr><tr><td>2</td><td>68.7 66.1</td><td>83.1 53.4</td><td>37.7 55.4</td></tr><tr><td>5</td><td>68.8 66.1</td><td>83.2 53.9</td><td>38.5 55.8</td></tr><tr><td>7</td><td>68.9 66.3</td><td>83.4 53.8</td><td>38.4 55.7</td></tr><tr><td>9</td><td>68.9 66.3</td><td>83.3 53.8</td><td>38.3 55.7</td></tr></table>

![](images/67c763d0136bc43ae09ac20a849cced0d45b1966cd75b9147b49214d366689ce.jpg)  
Fig. 6. The graph depicts the changes in the association performance of Sparse Track and ByteTrack [6] as the resolution varies. While there is marginal improvement in detector performance with increasing resolution, the association performance of the trackers consistently strengthens. It is worth noting that SparseTrack demonstrates more significant enhancements compared to ByteTrack.

After integrating DCM, the association performance of BYTE gradually improves with the increase in the number of pseudo-depth levels. It indicates that a higher number of pseudo-depth levels is more beneficial to associate lowscore occlusions. However, it does not yield additional gains when the number of pseudo-depth levels is exorbitant (e.g. the number of pseudo-depth levels exceeds 7). We speculate that excessively frequent layering schemes result in increased sparsity among both low-scoring detections and participating tracks, which can not yield additional benefits compared to an already sparse scheme.

TABLE 5  
Comparison of the association performance with or without global motion compensation on MOT17 and MOT20 validation sets.
<table><tr><td rowspan="2">w/ GMC</td><td colspan="2">MOT17</td><td colspan="2">MOT20</td></tr><tr><td>|HOtA↑AssA↑ IDF1↑HOtA↑ AssA↑ IDF1↑</td><td></td><td></td><td></td></tr><tr><td rowspan="2">✓</td><td>67.9 69.9</td><td>79.1</td><td>69.9</td><td>67.8 84.5</td></tr><tr><td>69.2</td><td>72.3 81.4</td><td>69.9</td><td>67.7 84.4</td></tr></table>

## 4.3.2 The impact of the input with different resolutions

We investigate the impact of input resolution on the tracking performance of SparseTrack. Specifically, we maintain consistent detection settings and test the association performance of SparseTrack and ByteTrack on the MOT20 validation dataset with varying input resolutions. We set the pseudo-depth levels to 8 in SparseTrack for associating low-scoring detections. The results are shown in Fig. 6.

As the increase in resolution, the performance of object detection shows a gradual improvement. However, beyond a certain threshold in resolution, the detector exhibits diminishing returns, with little to no substantial gains. Interestingly, both SparseTrack and ByteTrack show significant enhancements in their tracking performance as resolution continues to climb. This phenomenon can be attributed to the increased detection capability to recognize and involve more severely occluded targets within the tracking process. Notably, our observations suggest that SparseTrack outperforms ByteTrack in terms of association enhancements during the process.

## 4.3.3 The impact of the global motion compensation

We perform ablative analyses of the global motion compensation (GMC) module using SparseTrack on both the MOT17 and MOT20 validation datasets. During the lowscore association stage, we fix the number of pseudo-depth levels at 8. We maintain the default input resolution settings for MOT17 and MOT20. The results are presented in Tab. 5.

By incorporating GMC, substantial enhancements in the performance of SparseTrack are achieved on the MOT17 validation dataset. This improvement primarily stem from the more pronounced camera motion prevalent in the MOT17 dataset. Conversely, GMC yield little gains in tracking performance on the MOT20 dataset.

## 4.4 Comparison with Other Simple Methods

We compare SparseTrack with other location-dependent association methods on the MOT17, MOT20, and DanceTrack validation datasets respectively. The results as shown in Tab. 6. To ensure a fair comparison, we remove all specific hyper-parameters for each video to prevent any bias towards any method and all experiments maintain the same YOLOX-x detector model weights and NMS [13] hyperparameters. For the SparseTrack, the number of pseudodepth layers during the low-score association phase is consistently set to 8 and the low-score detection threshold remains at the default value of 0.1. It is worth mentioning that SparseTrack without GMC module reports the results on the MOT20 and DanceTrack validation set.

TABLE 6  
Comparison of the association methods on MOT17, MOT20 and Dance Track validation sets.
<table><tr><td rowspan="2">Methods</td><td colspan="4">MOT17</td><td colspan="4">MOT20</td><td colspan="4">DanceTrack</td></tr><tr><td>Hota↑ AssA↑ IDF1↑ Mota↑|</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>|Hota↑ AssA↑ IDF1↑ Mota↑ Hota↑ AssA↑ IDF1↑ Mota↑</td></tr><tr><td>SORT [5]</td><td>62.9</td><td>61.9</td><td>70.5</td><td>73.5</td><td>65.2</td><td>59.1</td><td>74.8</td><td>85.1</td><td>43.3</td><td>26.0</td><td>40.8</td><td>86.0</td></tr><tr><td>ByteTrack [6]</td><td>68.0</td><td>69.9</td><td>79.1</td><td>76.3</td><td>69.4</td><td>66.9</td><td>83.3</td><td>86.1</td><td>52.2</td><td>36.2</td><td>54.3</td><td>87.5</td></tr><tr><td>BoT-SORT [56]</td><td>68.9</td><td>71.2</td><td>80.7</td><td>76.9</td><td>69.5</td><td>67.1</td><td>83.7</td><td>86.1</td><td>53.4</td><td>37.6</td><td>56.1</td><td>87.7</td></tr><tr><td>OC-SORT ]</td><td>66.3</td><td>68.8</td><td>77.5</td><td>74.2</td><td>68.0</td><td>64.9</td><td>82.0</td><td>85.1</td><td>51.2</td><td>34.8</td><td>51.1</td><td>85.1</td></tr><tr><td>SparseTrack</td><td>69.2</td><td>72.3</td><td>81.4</td><td>76.8</td><td>69.9</td><td>67.8</td><td>84.5</td><td>86.1</td><td>53.8</td><td>38.5</td><td>56.0</td><td>87.1</td></tr></table>

Compared to previous methods that rely on position information and Kalman filter [17], the association approach of SparseTrack achieves the best tracking performance on the aforementioned three validation sets.

## 4.5 Visualization

To exhibiting the target set decomposition process and the performance of associating crowded targets intuitively, we conduct several different visualization experiments, as shown in Fig. 7, Fig. 8 and Fig. 9, respectively.

Visualization of the target set decomposition. Although the target set decomposition is a sub-process of DCM, it is the key to decoupling dense occlusions. In fact, not all targets in the scene are occluded state, but occluded targets may gather together and form dense occlusions easier, as shown in Fig. 7. By decomposing dense occlusion set based on pseudo-depth information, overlapping targets can be assigned to different depth levels, effectively. We visualize the decomposition of a set of dense occlusions, as shown in Fig. 8.

Visualization of associating occlusive targets via SparseTrack and ByteTrack. We conduct visualizations using both SparseTrack and ByteTrack on four different videos from the MOT challenge, while maintaining consistent detection hyper-parameter settings for the publicly available detector. The comparative results are shown in Fig. 9. Comparing to ByteTrack, SparseTrack demonstrates greater stability in handling challenging occlusion scenarios. It is attributed to its ability to differentiate between targets at various depths crowded together, leveraging discriminative pseudo-depth information originating from the targets.

## 4.6 Discussion

While SparseTrack demonstrates remarkable performance improvements on both the MOT datasets and the Dance-Track dataset, it has poor performance on the public detection benchmark of the MOT dataset. The association performance of SparseTrack is substantially contingent on the detector's capacity to accurately identify occluded targets. To be precise, SparseTrack places a higher reliance on the detection quality of partially obscured and low-confidence targets compared to conventional detection-based tracking methods. In addition, SparseTrack faces challenges in other demanding scenarios such as [82], [83] where targets exhibit rapid motion and deformation. In such cases, it becomes difficult for the pseudo-depth method to capture the relative depth relationships accurately, which affects the tracking performance of SparseTrack. For videos with low frame rates [42], simple Kalman filter motion model struggles to obtain accurate motion cues, which would result in even less accurate pseudo-depth information. In the future work, we will explore the different decomposition methods that can be adapted to various tracking scenarios and aim to make the implementation more elegant.

## 5 CONCLUSION

We propose a simple tracker for multi-object tracking named as SparseTrack. It leverages the pseudo-depth method to estimate the relative depth relationship between different targets and divides the target set into multiple sparse subsets in order of increasing depth. In order to associate occluded targets distributed across these sparse subsets, we introduce the Depth Cascade Matching (DCM) algorithm, which performs the association between detection subsets and trajectory subsets at the same depth level. Compared to previous tracking methods, SparseTrack offers a different perspective on addressing occlusion: the target set decomposition, which partitions the dense target set into sparse subsets by pseudo-depth information. SparseTrack achieves competitive performance on the MOT17 and MOT20 datasets comparing to state-of-the-art methods via solely utilizing simple IoU distance association, without relying on robust appearance embeddings or enhanced motion prediction. This demonstrates the effectiveness of decomposition based on depth information. It is worth noting that DCM is plugand-play and can be integrated into any existing tracker, yielding consistent performance improvements. We hope that SparseTrack can provide alternative solutions for multiobject tracking tasks and inspire the development of more powerful and elegant approaches based on the concept of sparsity in the future.

## ACKNOWLEDGMENTS

We thank Yifu Zhang, Bencheng Liao, Jiemin Fang, Yunchi Zhang, and Jinfeng Yao for their insightful discussions and suggestions. This work is in part supported by the National Key Research and Development Program of China under Grant 2022YFB4500602.

## REFERENCES

[1] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool, "Multi-task learning for dense prediction tasks: A survey," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. 1

![](images/07fa00207781d695f7ba0c50ceba3e2a920816f86e33c4be1c5f5480ae8cb941.jpg)

often clustered, resulting in dense occlusions.  
![](images/971348aa74b0dfbe9444ca2ff4b46f9810335553125bfb7b42bb386bac3ff649.jpg)  
dividing pseudo-depth levels.

[2] L. Leal-Taixé, A. Milan, I. Reid, S. Roth, and K. Schindler, "MOTChallenge 2015: Towards a benchmark for multi-target tracking," arXiv:1504.01942 [cs], Apr. 2015, arXiv: 1504.01942. [Online]. Available: http: / / arxiv.org/abs/1504.01942 1

[3] A. Milan, L. Leal-Taixé, I. Reid, S. Roth, and K. Schindler, "Mot16: A benchmark for multi-object tracking," arXiv preprint arXiv:1603.00831, 2016. 1, 2, 6

[4] P. Dendorfer, H. Rezatofighi, A. Milan, J. Shi, D. Cremers, I. Reid, S. Roth, K. Schindler, and L. Leal-Taixé, "Mot20: A benchmark for multi object tracking in crowded scenes," arXiv preprint arXiv:2003.09003, 2020. 1, 2, 6

[5] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple online and realtime tracking," in 2016 IEEE International Conference on Image Processing (ICIP), 2016, pp. 34643468. 1, 2, 3, 9

[6] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, "Bytetrack: Multi-object tracking by associating every detection box," 2022. 1, 3, 5, 6, 7, 8, 9

[7] Y. Xu, Y. Ban, G. Delorme, C. Gan, D. Rus, and X. Alameda-Pineda, "Transcenter: Transformers with dense queries for multiple-object tracking," arXiv preprint arXiv:2103.15145, 2021. 1, 3, 7

[8] P. Sun, Y. Jiang, R. Zhang, E. Xie, J. Cao, X. Hu, T. Kong, Z. Yuan, C. Wang, and P. Luo, "Transtrack: Multiple-object tracking with transformer," arXiv preprint arXiv:2012.15460, 2020. 1, 3, 7

[9] P. Chu, J. Wang, Q. You, H. Ling, and Z. Liu, "Transmot: Spatialtemporal graph transformer for multiple object tracking," arXiv preprint arXiv:2104.00194, 2021. 1, 3, 7

[10] Z. Zhao, Z. Wu, Y. Zhuang, B. Li, and J. Jia, "Tracking objects as pixel-wise distributions," 2022. 1, 7

[11] F. Zeng, B. Dong, Y. Zhang, T. Wang, X. Zhang, and Y. Wei, "Motr: End-to-end multiple-object tracking with transformer," in European Conference on Computer Vision (ECCV), 2022. 1, 3, 7

[12] Y. Zhang, T. Wang, and X. Zhang, "Motrv2: Bootstrapping endto-end multi-object tracking by pretrained object detectors," Computer Vision and Pattern Recognition CVPR, 2023. 1, 3, 7

[13] R. Girshick, J. Donahue, T. Darrell, and J. Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," in Computer Vision and Pattern Recognition (CVPR), 2014. 2, 8

[14] J. Luiten, A. Osep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, and B. Leibe, "Hota: A higher order metric for evaluating multiobject tracking," International journal of computer vision, vol. 129, no. 2, pp. 548578, 2021. 2, 6

[15] P. Sun, J. Cao, Y. Jiang, Z. Yuan, S. Bai, K. Kitani, and P. Luo, "Dancetrack: Multi-object tracking in uniform appearance and diverse motion," arXiv preprint arXiv:2111.14690, 2021. 2, 3, 6

[16] E. Bochinski, V. Eiselein, and T. Sikora, "High-speed trackingby-detection without using image information," in International Workshop on Traffic and Street Surveillance for Safety and Security at IEEE AVSS 2017, Lecce, Italy, Aug. 2017. [Online]. Available: http:/ /elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf 2

[17] K. RE, "A new approach to linear filtering and prediction problems." J Fluids Eng, vol. 82, no. 1, pp. 3545, 1960. 3, 9

[18] N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," in 2017 IEEE International Conference on Image Processing (ICIP). IEEE, 2017, pp. 36453649. 3, 5

[19] C. Long, A. Haizhou, Z. Zijie, and S. Chong, "Real-time multiple people tracking with deeply learned candidate selection and person re-identification," in IME, 2018. 3

[20] L. He, X. Liao, W. Liu, X. Liu, P. Cheng, and T. Mei, "Fastreid: A pytorch toolbox for general instance re-identification," arXiv preprint arXiv:2006.02631, 2020. 3

[21] K. Zhou and T. Xiang, "Torchreid: A library for deep learning person re-identification in pytorch," arXiv preprint arXiv:1910.10093, 2019. 3

[22] K. Zhou, Y. Yang, A. Cavallaro, and T. Xiang, "Omni-scale feature learning for person re-identification," in ICV, 2019. 3

[23] —, "Learning generalisable omni-scale representations for person re-identification," 2021. 3

[24] Z. Wang, H. Zhao, Y.-L. Li, S. Wang, P. Torr, and L. Bertinetto, "Do different tracking tasks require different appearance models?" Thirty-Fifth Conference on Neural Infromation" Processing Systems, 2021.3

[25] J. Pang, L. Qiu, X. Li, H. Chen, Q. Li, T. Darrell, and F. Yu, "Quasi-dense similarity learning for multiple object tracking," in

IEEE/CVF Conference on Computer Vision and Pattern Recognition, June 2021. 3, 5, 7

[26] S. You, H. Yao, B.-K. Bao, and C. Xu, "Utm: A unified multiple object tracking model with identity-aware feature enhancement," Computer Vision and Pattern Recognition (CVPR), 2023. 3, 7

[27] W. Li, Y. Xiong, S. Yang, M. Xu, Y. Wang, and W. Xia, "Semi-tcl: Semi-supervised track contrastive representation learning," arXiv preprint arXiv:2107.02396, 2021. 3, 7

[28] Q. Liu, D. Chen, Q. Chu, L. Yuan, B. Liu, L. Zhang, and N. Yu, "Online multi-object tracking with unsupervised reidentification learning and occlusion estimation," Neurocomput., vol. 483, no. C, p. 333347, apr 2022. [Online]. Available: https:/ /doi.org/10.1016/j.neucom.2022.01.008 3

[29] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum contrast for unsupervised visual representation learning," arXiv preprint arXiv:1911.05722, 2019. 3

[30] X. Chen, H. Fan, R. Girshick, and K. He, "Improved baselines with momentum contrastive learning," arXiv preprint arXiv:2003.04297, 2020.3

[31] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," arXiv preprint arXiv:2002.05709, 2020. 3

[32] T. Chen, S. Kornblith, K. Swersky, M. Norouzi, and G. Hinton, "Big self-supervised models are strong semi-supervised learners," arXiv preprint arXiv:2006.10029, 2020. 3

[33] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, "Masked autoencoders are scalable vision learners," arXiv:2111.06377, 2021. 3

[34] Z. Wang, L. Zheng, Y. Liu, and S. Wang, "Towards real-time multi-object tracking," The European Conference on Computer Vision (ECCV), 2020. 3

[35] Y. Zhang, C. Wang, X. Wang, W. Zeng, and W. Liu, "Fairmot: On the fairness of detection and re-identification in multiple object tracking," International Journal of Computer Vision, vol. , pp. 30693087, 2021. 3, 5, 7

[36] C. Liang, Z. Zhang, X. Zhou, B. Li, S. Zhu, and W. Hu, "Rethinking the competition between detection and reid in multiobject tracking," IEÉE Trans Image Process, pp. 31823196, 2022. 3, 5, 7

[37] T. Meinhardt, A. Kirillov, L. Leal-Taixe, and C. Feichtenhofer, Trackformer: Multi-object tracking with transformers," in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2022. 3

[38] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in neural information processing systems, 2017, pp. 5998 6008. 3, 7

[39] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in International Conference on Learning Representations, 2021. [Ónline]. Available: https:/ /openreview.net/forum?id=YicbFdNTTy 3, 7

[40] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in European conference on computer vision. Springer, 2020, pp. 213229. 3

[41] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, "Deformable {detr}: Deformable transformers for end-toend object detection," in International Conference on Learning Representations, 2021. [Online]. Available: https:/ /openreview. net/forum?id=gZ9hCDWe6ke 3

[42] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell, "Bdd100k: A diverse driving dataset for heterogeneous multitask learning," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 26362645. 3, 9

[43] C. Feichtenhofer, A. Pinz, and A. Zisserman," "Detect to track and track to detect," in International Conference on Computer Vision (ICCV), 2017. 3

[44] X. Zhou, V. Koltun, and P. Krähenbühl, "Tracking objects as points," ECCV, 2020. 3, 5, 7

[45] B. Shuai, A. Berneshawi, X. Li, D. Modolo, and J. Tighe, "Siammot: Siamese multi-object tracking," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12 37212 382. 3, 7

[46] P. Bergmann, T. Meinhardt, and L. Leal-Taixé, "Tracking without bells and whistles," in The IEEE International Conference on Computer Vision (ICCV), October 2019. 3

[47] J. Peng, C. Wang, F. Wan, Y. Wu, Y. Wang, Y. Tai, C. Wang, J. Li, F. Huang, and Y. Fu, "Chained-tracker: Chaining paired attentive regression results for end-to-end joint multiple-object detection and tracking," in Proceedings of the European Conference on Computer Vision, 2020. 3, 7

[48] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "Imagenet classification with deep convolutional neural networks," NIPS, 2012. 3

[49] O. Cetintas, G. Bras'o, and L. Leal-Taix'e, "Unifying short and long-term tracking with graph hierarchies," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 22 87722 887. 3

[50] G. Braso and L. Leal-Taixe, "Learning a neural solver for multiple object tracking," CVPR, 2020. 3, 7

[51] J. He, Z. Huang, N. Wang, and Z. Zhang, "Learnable graph matching: Incorporating graph partitioning with deep feature learning for multiple object tracking," 2021, p. 52995309. 3, 7

[52] J. Li, X. Gao, and T. Jiang, "Graph networks for multiple object tracking," In Proceedings of the "IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), March 2020. 3, 7

[53] Y. Wang, K. Kitani, and X. Weng, "Joint object detection and multi-object tracking with graph neural networks," arXiv preprint arXiv:2006.13164, 2020. 3, 7

[54] Z. Qin, S. Zhou, L. Wang, J. Duan, G. Hua, and W. Tang, "Motiontrack: Learning robust short-term and long-term motions for multi-object tracking," Computer Vision and Pattern Recognition (CVPR), 2023. 3, 7

[55] K. G. Quach, H. Le, P. Nguyen, C. N. Duong, T. D. Bui, and K. Luu, "Depth perspective-aware multiple object tracking," 2023. 3

[56] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "Bot-sort: Robust associations multi-pedestrian tracking," arXiv preprint arXiv:2206.14651, 2022. 3, 7, 9

[57] A. Hornakova, T. Kaiser, P. Swoboda, M. Rolinek, B. Rosenhahn, and R. Henschel, "Making higher order mot scalable: An efficient approximate solver for lifted disjoint paths," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICV), October 2021, pp. 63306340. 3

[58] Y. Liu, X. Zhang, B. Zhang, X. Zhang, S. Wang, and J. Xu, "Multicamera vehicle tracking based on occlusion-aware and intervehicle information," in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022. 3

[59] D. H. Nguyen, R. Henschel, B. Rosenhahn, D. Sonntag, and P. Swoboda, "Lmgp: Lifted multicut meets geometry projections for multi-camera multi-object tracking," in 2022 IEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3

[60] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, "Nerf: Representing scenes as neural radiance fields for view synthesis," in ECCV, 2020. 3

[61] P. Voigtlaender, M. Krause, A. Osep, J. Luiten, B. B. G. Sekar, A. Geiger, and B. Leibe, "MOTS: Multi-object tracking and segmentation," in CVPR, 2019. 3

[62] E. Baser, V. Balasubramanian, P. Bhattacharyya, and K. Czarnecki, "Fantrack: 3d multi-object tracking with feature association network," in IEEE Intelligent Vehicles Symposium (IV 19), 2019. 3

[63] T. Yin, X. Zhou, and P. Krähenbühl, "Center-based 3d object detection and tracking," CVPR, 2021. 3

[64] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, "Yolox: Exceeding yolo series in 2021," arXiv preprint arXiv:2107.08430, 2021. 4, 6

[65] Y. Du, Y. Song, B. Yang, and Y. Zhao, "Strongsort: Make deepsort great again," arXiv preprint arXiv:2202.13514, 2022. 5, 7

[66] J. Wu, J. Cao, L. Song, Y. Wang, M. Yang, and J. Yuan, "Track to detect and segment: An online multi-object tracker," in Procedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12 35212 361. 5, 7

[67] K. Bernardin and R. Stiefelhagen, "Evaluating multiple object tracking performance: the clear mot metrics," EURASIP Journal on Image and Video Processing, vol. 2008, pp. 110, 2008. 6

[68] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, "Performance measures and a data set for multi-target, multi-camera tracking," in ECCV. Springer, 2016, pp. 1735. 6

[69] B. Pang, Y. Li, Y. Zhang, M. Li, and C. Lu, "Tubetk: Adopting tubes to track multi-object in a one-step training model," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 63086318. 7

[70] S. Han, P. Huang, H. Wang, E. Yu, D. Liu, X. Pan, and J. Zhao, "Mat: Motion-aware multi-object tracking," arXiv preprint arXiv:2009.04794, 2020. 7

[71] P. Tokmakov, J. Li, W. Burgard, and A. Gaidon, "Learning to track with object permanence," arXiv preprint arXiv:2103.14258, 2021. 7

[72] J. Cao, X. Weng, R. Khirodkar, J. Pang, and K. Kitani, "Observation-centric sort: Rethinking sort for robust multi-object tracking," arXiv preprint arXiv:2203.14360, 2022. 7, 9

[73] S. Sun, N. Akhtar, H. Song, A. S. Mian, and M. Shah, "Deep affinity network for multiple object tracking," IEEE transactions on pattern analysis and machine intelligence, 2019. 7

[74] L. Zheng, M. Tang, Y. Chen, G. Zhu, J. Wang, and H. Lu, "Improving multiple object tracking with single object tracking," in Proceedings of the IÉEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 24532462. 7

[75] F. Yang, X. Chang, S. Sakti, Y. Wu, and S. Nakamura, "Remot: A model-agnostic refinement for multiple object tracking," Image and Vision Computing, vol. 106, p. 104091, 2021. 7

[76] E. Yu, Z. Li, S. Han, and H. Wang, "Relationtrack: Relation-aware multiple object tracking with decoupled representation," arXiv preprint arXiv:2105.04322, 2021. 7

[77] C. Shan, C. Wei, B. Deng, J. Huang, X.-S. Hua, X. Cheng, and K. Liang, "Tracklets predicting based adaptive graph tracking," arXiv preprint arXiv:2010.09015, 2020. 7

[78] Q. Wang, Y. Zheng, P. Pan, and Y. Xu, "Multiple object tracking with correlation learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3876 3886. 7

[79] Y. Zhang, H. Sheng, Y. Wu, S. Wang, W. Ke, and Z. Xiong, "Multiplex labeling graph for near-online tracking in crowded scenes," IEEE Internet of Things Journal, vol. 7, no. 9, pp. 78927902, 2020. 7

[80] A. Girbau, F. Marqués, and S. Satoh, "Multiple object tracking from appearance by hierarchically clustering tracklets," BMVC, 2022. 7

[81] X. Zhou, T. in, V. Koltun, and P. Krähenbühl, "Global tracig transformers," in CVPR, 2022. 7

[82] P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, and H. Ling, "Detection and tracking meet drones challenge," IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 11, 2021. 9

[83] D. Du, Y. Qi, H. Yu, Y. Yang, K, Duan, G. Li, W. Zhang, Q. Huang, and Q. Tian, "The unmanned aerial vehicle benchmark: Object detection and tracking," ECCV, pp. 370386, 2018. 9

![](images/9ed862ecaf93bfa0ee93574369bb19632dcd20223ce01d68adb8b32b5aa15112.jpg)  
Fi TOT x yellow numbers signify that the image cropping regions originate from the same frame.