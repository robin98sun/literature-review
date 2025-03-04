

# Literature review of Spring'25

From Feb 20th, 2025

0. ### Content:

	- [ASPLOS'24](#ASPLOS'24)
	- [HPCA'24](#HPCA'24)
	- [HPDC'24](#HPDC'24)
	- [EuroSys'25](#EuroSys'25)
	- [NSDI'24](#NSDI'24)
	- [SOSP'24](#SOSP'24)

	- going to collect:

		```
		https://mlsys.org/Conferences/2024
		https://acmsocc.org/2024/schedule.html
		https://sc24.supercomputing.org/proceedings/tech_paper/
		https://www.usenix.org/conference/atc24/technical-sessions
		https://sigops.org/s/conferences/sosp/2024/accepted.html
		https://www.usenix.org/conference/osdi24/technical-sessions
		```

<a name="ASPLOS'24"></a>
1. [ASPLOS'24](https://www.asplos-conference.org/asplos2024/main-program/index.html)

<table>

  <tr>
  	<td>
	    <b><a href="https://openreview.net/pdf/58de1dd82ec19b52473be7e4af3f6ed777c4a525.pdf">Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning</a></b>
	    <p>
			Chang Chen, Xiuhong Li, and Qianchao Zhu (Peking University) <br>
			Jiangfei Duan (Chinese University of Hong Kong) <br> 
			Peng Sun and Xingcheng Zhang (Shanghai AI Lab) <br> 
			Chao Yang (Peking University) <br>
	    </p>
	    <p>
	    	<b>Labels:</b> ML communication
	    </p>
	    <p><b><i>Awarded Best Paper!</i></b></p>
		<p> 
			<b>Abstract:</b>
			Efficiently training large language models (LLMs) necessitates the adoption of hybrid parallel methods, integrating multiple communications collectives within distributed partitioned graphs. Overcoming communication bottlenecks is crucial and is often achieved through communication and computation overlaps. However, existing overlap methodologies tend to lean towards either fine-grained kernel fusion or limited operation scheduling, constraining performance optimization in heterogeneous training environments.
			<br>
			This paper introduces Centauri, an innovative framework that encompasses comprehensive communication partitioning and hierarchical scheduling schemes for optimized overlap. We propose a partition space comprising three inherent abstraction dimensions: primitive substitution, topology-aware group partitioning, and workload partitioning. These dimensions collectively create a comprehensive optimization space for efficient overlap. To determine the efficient overlap of communication and computation operators, we decompose the scheduling tasks in hybrid parallel training into three hierarchical tiers: operation, layer, and model. Through these techniques, Centauri effectively overlaps communication latency and enhances hardware utilization. Evaluation results demonstrate that Centauri achieves up to 1.49× speedup over prevalent methods across various parallel training configurations.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3620665.3640410">T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives</a></b>
	    <p>
			Suchita Pati (University of Wisconsin-Madison and AMD)<br>
			Shaizeen Aga, Mahzabeen Islam, and Nuwan Jayasena (AMD)<br>
			Matthew D. Sinclair (University of Wisconsin-Madison and AMD)
	    </p>
	    <p>
	    	<b>Labels:</b> ML communication
	    </p>
		<p> 
			<b>Abstract:</b>
			Efficiently training large language models (LLMs) necessitates the adoption of hybrid parallel methods, integrating multiple communications collectives within distributed partitioned graphs. Overcoming communication bottlenecks is crucial and is often achieved through communication and computation overlaps. However, existing overlap methodologies tend to lean towards either fine-grained kernel fusion or limited operation scheduling, constraining performance optimization in heterogeneous training environments.
			<br>
			This paper introduces Centauri, an innovative framework that encompasses comprehensive communication partitioning and hierarchical scheduling schemes for optimized overlap. We propose a partition space comprising three inherent abstraction dimensions: primitive substitution, topology-aware group partitioning, and workload partitioning. These dimensions collectively create a comprehensive optimization space for efficient overlap. To determine the efficient overlap of communication and computation operators, we decompose the scheduling tasks in hybrid parallel training into three hierarchical tiers: operation, layer, and model. Through these techniques, Centauri effectively overlaps communication latency and enhances hardware utilization. Evaluation results demonstrate that Centauri achieves up to 1.49× speedup over prevalent methods across various parallel training configurations.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3620665.3640427">Two-Face: Combining Collective and One-Sided Communication for Efficient Distributed SpMM</a></b>
	    <p>
			Charles Block, Gerasimos Gerogiannis, and Charith Mendis (University of Illinois at Urbana-Champaign)<br>
			Ariful Azad (Indiana University) <br>
			Josep Torrellas (University of Illinois at Urbana-Champaign)
	    </p>
	    <p>
	    	<b>Labels:</b> ML communication
	    </p>
		<p> 
			<b>Abstract:</b>
			Sparse matrix dense matrix multiplication (SpMM) is commonly used in applications ranging from scientific computing to graph neural networks. Typically, when SpMM is executed in a distributed platform, communication costs dominate. Such costs depend on how communication is scheduled. If it is scheduled in a sparsity-unaware manner, such as with collectives, execution is often inefficient due to unnecessary data transfers. On the other hand, if communication is scheduled in a fine-grained sparsity-aware manner, communicating only the necessary data, execution can also be inefficient due to high software overhead.
			<br>
			We observe that individual sparse matrices often contain regions that are denser and regions that are sparser. Based on this observation, we develop a model that partitions communication into sparsity-unaware and sparsity-aware components. Leveraging the partition, we develop a new algorithm that performs collective communication for the denser regions, and fine-grained, one-sided communication for the sparser regions. We call the algorithm Two-Face. We show that Two-Face attains an average speedup of 2.11x over prior work when evaluated on a 4096-core supercomputer. Additionally, Two-Face scales well with the machine size.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640367">A Journey of a 1,000 Kernels Begins with a Single Step: A Retrospective of Deep Learning on GPUs</a></b>
	    <p>
			Michael Davies, Ian McDougall, Selvaraj Anandaraj, Deep Machchhar, Rithik Jain, and Karthikeyan Sankaralingam (University of Wisconsin-Madison)
	    </p>
	    <p>
	    	<b>Labels:</b> ML benchmark
	    </p>
		<p> 
			<b>Abstract:</b>
			We are in age of AI, with rapidly changing algorithms and a somewhat synergistic change in hardware. MLPerf is a recent benchmark suite that serves as a way to compare and evaluate hardware. However it has several drawbacks – it is dominated by CNNs and does a poor job of capturing the diversity of AI use cases, and only represents a sliver of production AI use cases. This paper performs a longitudinal study of state-of-art AI applications spanning vision, physical simulation, vision synthesis, language and speech processing, and tabular data processing, across three generations of hardware to understand how the AI revolution has panned out. We call this collection of applications and execution scaffolding the CaSiO suite. The paper reports on data gathered at the framework level, device API level, and hardware and microarchitecture level. The paper provides insights on the hardware-software revolution with pointers to future trends..
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3617232.3624853">Expanding Datacenter Capacity with DVFS Boosting: A safe and scalable deployment experience</a></b>
	    <p>
			Leonardo Piga, Iyswarya Narayanan, Aditya Sundarrajan, Matt Skach, and Qingyuan Deng (Meta)<br>
			Biswadip Maity (University of California Irvine)<br>
			Manoj Chakkaravarthy, Alison Huang, Abhishek Dhanotia, and Parth Malani (Meta)
	    </p>
	    <p>
	    	<b>Labels:</b> energy scaling
	    </p>
		<p> 
			<b>Abstract:</b>
			COVID-19 pandemic created unexpected demand for our physical infrastructure. We increased our computing supply by growing our infrastructure footprint as well as expanded existing capacity by using various techniques among those DVFS boosting. This paper describes our experience in deploying DVFS boosting to expand capacity. There are several challenges in deploying DVFS boosting at scale. First, frequency scaling incurs additional power demand, which can exacerbate power over-subscription and incur unexpected capacity loss for the services due to power capping. Second, heterogeneity is commonplace in any large scale infrastructure. We need to deal with the service and hardware heterogeneity to determine the optimal setting for each service and hardware type. Third, there exists a long tail of services with scarce resources and support for performance evaluation. Finally and most importantly, we need to ensure that large scale changes to CPU frequency do not risk the reliability of the services and the infrastructure. We present our solution that has overcome the above challenges and has been running in production for over 3 years. It created 12 MW of supply which is equivalent to building and populating half a datacenter in our fleet. In addition to the real world performance of our solution, we also share our key takeaways to improve fleetwide efficiency via DVFS boosting in a safe manner.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620666.3651335">SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification</a></b>
	    <p>
			Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, and Zeyu Wang (Carnegie Mellon University)<br>
			Zhengxin Zhang (Tsinghua University)<br>
			Rae Ying Yee Wong (Stanford University)<br>
			Alan Zhu and Lijie Yang (Carnegie Mellon University)<br>
			Xiaoxiang Shi (Shanghai Jiao Tong University)<br>
			Chunan Shi (Peking University)<br>
			Zhuoming Chen and Daiyaan Arfeen (Carnegie Mellon University)<br>
			Reyna Abhyankar (University of California San Diego)<br>
			Zhihao Jia (Carnegie Mellon University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML inference systems
	    </p>
		<p> 
			<b>Abstract:</b>
			This paper introduces SpecInfer, a system that accelerates generative large language model (LLM) serving with tree-based speculative inference and verification. The key idea behind SpecInfer is leveraging small speculative models to predict the LLM’s outputs; the predictions are organized as a token tree, whose nodes each represent a candidate token sequence. The correctness of all candidate token sequences represented by a token tree is verified against the LLM in parallel using a novel tree-based parallel decoding mechanism. SpecInfer uses an LLM as a token tree verifier instead of an incremental decoder, which significantly reduces the end-to-end latency and computational requirement for serving generative LLMs while provably preserving model quality. Our evaluation shows that SpecInfer outperforms existing LLM serving systems by 1.5-2.8× for distributed LLM inference and by 2.6-3.5× for offloading-based LLM inference, while preserving the same generative performance. SpecInfer is publicly available at https://github.com/flexflow/FlexFlow/
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640383">ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference</a></b>
	    <p>
			Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, and Junyeol Lee (Hanyang University)<br>
			Du-seong Chang (KT Corporation)<br>
			Jiwon Seo (Hanyang University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML inference systems
	    </p>
		<p> 
			<b>Abstract:</b>
			This paper presents ExeGPT, a distributed system designed for constraint-aware LLM inference. ExeGPT finds and runs with an optimal execution schedule to maximize inference throughput while satisfying a given latency constraint. By leveraging the distribution of input and output sequences, it effectively allocates resources and determines optimal execution configurations, including batch sizes and partial tensor parallelism. We also introduce two scheduling strategies based on Round-Robin Allocation and Workload-Aware Allocation policies, suitable for different NLP workloads. We evaluate ExeGPT on six LLM instances of T5, OPT, and GPT-3 and five NLP tasks, each with four distinct latency constraints. Compared to FasterTransformer, ExeGPT achieves up to 15.2× improvements in throughput and 6× improvements in latency. Overall, ExeGPT achieves an average throughput gain of 2.9× across twenty evaluation scenarios. Moreover, when adapting to changing sequence distributions, the cost of adjusting the schedule in ExeGPT is reasonably modest. ExeGPT proves to be an effective solution for optimizing and executing LLM inference for diverse NLP workload and serving conditions.
		</p>
		<p></p>
	</td>
  </tr>



  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3617232.3624849">Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling</a></b>
	    <p>
			Sohaib Ahmad and Hui Guan (University of Massachusetts Amherst)<br>
			Brian D. Friedman and Thomas Williams (Nokia Bell Labs)<br>
			Ramesh K. Sitaraman (University of Massachusetts Amherst)<br>
			Thomas Woo (Nokia Bell Labs)
	    </p>
	    <p>
	    	<b>Labels:</b> ML inference systems
	    </p>
		<p> 
			<b>Abstract:</b>
			Existing machine learning inference-serving systems largely rely on hardware scaling by adding more devices or using more powerful accelerators to handle increasing query demands. However, hardware scaling might not be feasible for fixed-size edge clusters or private clouds due to their limited hardware resources. A viable alternate solution is accuracy scaling, which adapts the accuracy of ML models instead of hardware resources to handle varying query demands. This work studies the design of a high-throughput inference-serving system with accuracy scaling that can meet throughput requirements while maximizing accuracy. To achieve the goal, this work proposes to identify the right amount of accuracy scaling by jointly optimizing three sub-problems: how to select model variants, how to place them on heterogeneous devices, and how to assign query workloads to each device. It also proposes a new adaptive batching algorithm to handle variations in query arrival times and minimize SLO violations. Based on the proposed techniques, we build an inference-serving system called Proteus and empirically evaluate it on real-world and synthetic traces. We show that Proteus reduces accuracy drop by up to 3× and latency timeouts by 2-10× with respect to baseline schemes, while meeting throughput requirements.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    *** <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640411">SpotServe: Serving Generative Large Language Models on Preemptible Instances</a></b>
	    <p>
			Xupeng Miao (Carnegie Mellon University)<br>
			Chunan Shi (Peking University)<br>
			Jiangfei Duan (The Chinese University of Hong Kong)<br>
			Xiaoli Xi (Carnegie Mellon University)<br>
			Dahua Lin (Chinese University of Hong Kong and Sensetime Research)<br>
			Bin Cui (Peking University)<br>
			Zhihao Jia (Carnegie Mellon University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML inference systems
	    </p>
		<p> 
			<b>Abstract:</b>
			The high computational and memory requirements of generative large language models (LLMs) make it challenging to serve them cheaply. This paper aims to reduce the monetary cost for serving LLMs by leveraging preemptible GPU instances on modern clouds, which offer accesses to spare GPU resources at a much cheaper price than regular instances but may be preempted by the cloud provider at any time. Serving LLMs on preemptible instances requires addressing challenges induced by frequent instance preemptions and the necessity of migrating instances to handle the preemptions. This paper presents SpotServe, the first distributed LLM serving system on preemptible instances. Several key techniques of SpotServe realize fast and reliable serving of generative LLMs on cheap preemptible instances. First, SpotServe dynamically adapts the LLM parallelization configuration for dynamic instance availability and fluctuating workload, while balancing the trade-off among the overall throughput, inference latency and monetary costs. Second, to minimize the cost of migrating instances for dynamic reparallelization, the task of migrating instances is formulated as a bipartite graph matching problem in SpotServe, which uses the Kuhn-Munkres algorithm to identify an optimal migration plan that minimizes communication cost. Finally, to take advantage of the grace period offered by modern cloud platforms, we introduce stateful inference recovery, a new inference mechanism that commits inference progress at a much finer granularity and allows SpotServe to cheaply resume inference upon preemption. We evaluate SpotServe on real spot instance preemption traces and various popular LLMs and show that SpotServe can reduce the P99 tail latency by 2.4 – 9.1× compared with the best existing LLM serving systems. We also show that SpotServe can leverage the price advantage of preemptive instances, saving 54% monetary cost compared with only using on-demand instances. The code is publicly available at: https://github.com/Hsword/SpotServe.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3623278.3624753">DREAM: A Dynamic Scheduler for Dynamic Real-time Multi-model ML Workloads</a></b>
	    <p>
			Seah Kim (University of California Berkeley)<br>
			Hyoukjun Kwon (University of California Irvine)<br>
			Jinook Song, Jihyuck Jo, Yu-Hsin Chen, Liangzhen Lai, and Vikas Chandra (Meta)
	    </p>
	    <p>
	    	<b>Labels:</b> ML inference systems
	    </p>
		<p> 
			<b>Abstract:</b>
			Emerging real-time multi-model ML (RTMM) workloads such as AR/VR and drone control involve dynamic behaviors in various granularity; task, model, and layers within a model. Such dynamic behaviors introduce new challenges to the system software in an ML system since the overall system load is not completely predictable, unlike traditional ML workloads. In addition, RTMM workloads require real-time processing, involve highly heterogeneous models, and target resource-constrained devices. Under such circumstances, developing an effective scheduler gains more importance to better utilize underlying hardware considering the unique characteristics of RTMM workloads. Therefore, we propose a new scheduler, DREAM, which effectively handles various dynamicity in RTMM workloads targeting multi-accelerator systems. DREAM quantifies the unique requirements for RTMM workloads and utilizes the quantified scores to drive scheduling decisions, considering the current system load and other inference jobs on different models and input frames. DREAM utilizes tunable parameters that provide fast and effective adaptivity to dynamic workload changes. In our evaluation of five scenarios of RTMM workload, DREAM reduces the overall UXCost, which is an equivalent metric of the energy-delay product (EDP) for RTMM defined in the paper, by 32.2% and 50.0% in the geometric mean (up to 80.8% and 97.6%) compared to state-of-the-art baselines, which shows the efficacy of our scheduling methodology.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3617232.3624847">SoCFlow: Efficient and Scalable DNN Training on SoC-Clustered Edge Servers</a></b>
	    <p>
			Daliang Xu (Peking University)<br>
			Mengwei Xu (State Key Laboratory of Networking and Switching Technology)<br>
			Chiheng Lou (Peking University);Li Zhang (State Key Laboratory of Networking and Switching Technology)<br>
			Gang Huang, Xin Jin, and Xuanzhe Liu (Peking University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML Cluster Scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			SoC-Cluster, a novel server architecture composed of massive mobile system-on-chips (SoCs), is gaining popularity in industrial edge computing due to its energy efficiency and compatibility with existing mobile applications. However, we observe that the deployed SoC-Cluster servers are not fully utilized, because the hosted workloads are mostly user-triggered and have significant tidal phenomena. To harvest the free cycles, we propose to co-locate deep learning tasks on them. We present SoCFlow, the first framework that can efficiently train deep learning models on SoC-Cluster. To deal with the intrinsic inadequacy of commercial SoC-Cluster servers, SoCFlow incorporates two novel techniques: (1) the group-wise parallelism with delayed aggregation that can train deep learning models fast and scalably without being influenced by the network bottleneck; (2) the data-parallel mixed-precision training algorithm that can fully unleash the heterogeneous processors’ capability of mobile SoCs. We have fully implemented SoCFlow and demonstrated its effectiveness through extensive experiments. The experiments show that SoCFlow significantly and consistently outperforms all baselines regarding the training speed while preserving the convergence accuracy, e.g., 1.6×–740× convergence speedup with 32 SoCs. Compared to commodity GPU (NVIDIA V100) under the same power budget, SoCFlow achieves comparable training speed but reduces energy consumption by 2.31×–10.23× with the same convergence accuracy.
		</p>
		<p></p>
	</td>
  </tr>



  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3617232.3624863">Training Job Placement in Clusters with Statistical In-Network Aggregation</a></b>
	    <p>
			Bohan Zhao and Wei Xu (Tsinghua University)<br>
			Shuo Liu, Yang Tian, and Qiaoling Wang (Huawei)<br>
			Wenfei Wu (Peking University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML Cluster Scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			In-Network Aggregation (INA) offloads the gradient aggregation in distributed training (DT) onto programmable switches, where the switch memory could be allocated to jobs in either synchronous or statistical multiplexing mode. Statistical INA has advantages in switch memory utilization, control-plane simplicity, and management safety, but it faces the problem of cross-layer resource efficiency in job placement. This paper presents a job placement system NetPack for clusters with statistical INA, which aims to maximize the utilization of both computation and network resources. NetPack periodically batches and places jobs into the cluster. When placing a job, NetPack runs a steady state estimation algorithm to acquire the available resources in the cluster, heuristically values each server according to its available resources (GPU and bandwidth), and runs a dynamic programming algorithm to efficiently search for servers with the highest value for the job. Our prototype of NetPack and the experiments demonstrate that NetPack outperforms prior job placement methods by 45% in terms of average job completion time on production traces.
		</p>
		<p></p>
	</td>
  </tr>



  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640406">RAP: Resource-aware Automated GPU Sharing for Multi-GPU Recommendation Model Training and Input Preprocessing</a></b>
	    <p>
			Zheng Wang (University of California San Diego)<br>
			Yuke Wang and Jiaqi Deng (University of California Santa Barbara)<br>
			Da Zheng (Amazon)<br>
			Ang Li (Pacific Northwest National Laboratory)<br>
			Yufei Ding (University of California San Diego)
	    </p>
	    <p>
	    	<b>Labels:</b> ML Cluster Scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Ensuring high-quality recommendations for newly onboarded users requires the continuous retraining of Deep Learning Recommendation Models (DLRMs) with freshly generated data. To serve the online DLRM retraining, existing solutions use hundreds of CPU computing nodes designated for input preprocessing, causing significant power consumption that surpasses even the power usage of GPU trainers. To this end, we propose RAP, an end-to-end DLRM training framework that supports Resource-aware Automated GPU sharing for DLRM input Preprocessing and Training. The core idea of RAP is to accurately capture the remaining GPU computing resources during DLRM training for input preprocessing, achieving superior training efficiency without requiring additional resources. Specifically, RAP utilizes a co-running cost model to efficiently assess the costs of various input preprocessing operations, and it implements a resource-aware horizontal fusion technique that adaptively merges smaller kernels according to GPU availability, circumventing any interference with DLRM training. In addition, RAP leverages a heuristic searching algorithm that jointly optimizes both the input preprocessing graph mapping and the co-running schedule to maximize the end-to-end DLRM training throughput. The comprehensive evaluation shows that RAP achieves 1.99× speedup on average over the sequential GPU-based DLRM input preprocessing baseline. In addition, the end-to-end training throughput of RAP is only 3.24% lower than the ideal case, which has no input preprocessing overhead.
		</p>
		<p></p>
	</td>
  </tr>



  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640375">Heet: Accelerating Elastic Training in Heterogeneous Deep Learning Clusters</a></b>
	    <p>
			Zizhao Mo, Huanle Xu, and Chengzhong Xu (University of Macau)
	    </p>
	    <p>
	    	<b>Labels:</b> ML Cluster Scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Modern GPU clusters inherently exhibit heterogeneity, encompassing various aspects such as computation and communication. This heterogeneity poses a significant challenge for the elastic scheduling of deep learning workloads. Unfortunately, existing elastic schedulers often overlook the impact of heterogeneity on scaling efficiency, resulting in considerably prolonged job completion times. 
			<br>
			In this paper, we present Heet, a new Heterogeneity-aware system explicitly developed for elastic training in DL clusters. Heet addresses two critical issues. First, it utilizes a 3-D collaborative filtering method to accurately measure the scaling efficiency of all elastic configurations on heterogeneous hosts, substantially reducing profiling overhead. Second, Heet introduces a unique price function to effectively balance scaling efficiency and scheduling efficiency. Building upon this function, Heet incorporates a scalable mechanism that employs minimum-weight full bipartite matching and opportunistic resource trading to generate dynamic scheduling decisions. Evaluations conducted on cloud clusters and large-scale simulations demonstrate that Heet can reduce job completion time by up to 2.46× compared to existing solutions.
		</p>
		<p></p>
	</td>
  </tr>



  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640381">Efficient Microsecond-scale Blind Scheduling with Tiny Quanta</a></b>
	    <p>
			Zhihong Luo, Sam Son, and Dev Bali (University of California Berkeley)<br>
			Emmanuel Amaro (VMware Research)<br>
			Amy Ousterhout (University of California San Diego)<br>
			Sylvia Ratnasamy (University of California Berkeley)<br>
			Scott Shenker (ICSI and University of California Berkeley)
	    </p>
	    <p>
	    	<b>Labels:</b> scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			A longstanding performance challenge in datacenter-based applications is how to efficiently handle incoming client requests that spawn many very short (?s scale) jobs that must be handled with high throughput and low tail latency. When no assumptions are made about the duration of individual jobs, or even about the distribution of their durations, this requires blind scheduling with frequent and efficient preemption, which is not scalably supported for ?s-level tasks. We present Tiny Quanta (TQ), a system that enables efficient blind scheduling of ?s-level workloads. TQ performs fine-grained preemptive scheduling and does so with high performance via a novel combination of two mechanisms: forced multitasking and two-level scheduling. Evaluations with a wide variety of ?s-level workloads show that TQ achieves low tail latency while sustaining 1.2x to 6.8x the throughput of prior blind scheduling systems.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3620666.3651376">AUDIBLE: A Convolution-Based Resource Allocator for Oversubscribing Burstable Virtual Machines</a></b>
	    <p>
			Seyedali Jokar Jandaghi and Kaveh Mahdaviani (University of Toronto)<br>
			Amirhossein Mirhosseini (University of Michigan)<br>
			Sameh Elnikety (Microsoft Research)<br>
			Cristiana Amza and Bianca Schroeder (University of Toronto)
	    </p>
	    <p>
	    	<b>Labels:</b> scheduling, over-committing
	    </p>
		<p> 
			<b>Abstract:</b>
			In an effort to increase the utilization of data center resources cloud providers have introduced a new type of virtual machine (VM) offering, called a burstable VM (BVM). Our work is the first to study the characteristics of burstable VMs (based on traces from production systems at a major cloud provider) and resource allocation approaches for BVM workloads. We propose new approaches for BVM resource allocation and use extensive simulations driven by field data to compare them with two baseline approaches used in practice. We find that traditional approaches based on using a fixed oversubscription ratio or based on the Central Limit Theorem do not work well for BVMs: They lead to either low utilization or high server capacity violation rates. Based on the lessons learned from our workload study, we develop a new approach to BVM scheduling, called Audible, using a non-parametric statistical model, which makes the approach light-weight and workload independent, and obviates the need for training machine learning models and for tuning their parameters. We show that Audible achieves high system utilization while being able to enforce stringent requirements on server capacity violations.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3623278.3624762">CPS: A Cooperative Para-virtualized Scheduling Framework for Manycore Machines</a></b>
	    <p>
			Yuxuan Liu, Tianqiang Xu, Zeyu Mi, Zhichao Hua, Binyu Zang, and Haibo Chen (Shanghai Jiao Tong University)
	    </p>
	    <p>
	    	<b>Labels:</b> scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Today’s cloud platforms oﬀer large virtual machine (VM) instances with multiple virtual CPUs (vCPU) on manycore machines. These machines typically have a deep memory hierarchy to enhance communication between cores. Although previous researches have primarily focused on addressing the performance scalability issues caused by the double scheduling problem in virtualized environments, they mainly concentrated on solving the preemption problem of synchronization primitives and the traditional NUMA architecture. This paper speciﬁcally targets a new aspect of scalability issues caused by the absence of runtime hypervisor-internal states (RHS). We demonstrate two typical RHS problems, namely the invisible pCPU (physical CPU) load and dynamic cache group mapping. These RHS problems result in a collapse in VM performance and low CPU utilization because the guest VM lacks visibility into the latest runtime internal states maintained by the hypervisor, such as pCPU load and vCPU-pCPU mappings. Consequently, the guest VM makes ineﬃcient scheduling decisions. To address the RHS issue, we argue that the solution lies in exposing the latest scheduling decisions made by both the guest and host schedulers to each other. Hence, we present a cooperative para-virtualized scheduling framework called CPS, which facilitates the proactive exchange of timely scheduling information between the hypervisor and guest VMs. To ensure eﬀective scheduling decisions for VMs, a series of techniques are proposed based on the exchanged information. We have implemented CPS in Linux KVM and have designed corresponding solutions to tackle the two RHS problems. Evaluation results demonstrate that CPS signiﬁcantly improves the performance of PARSEC by 81.1% and FxMark by 1.01x on average for the two identiﬁed problems.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620666.3651357">PrimePar: Efficient Spatial-temporal Tensor Partitioning for Large Transformer Model Training</a></b>
	    <p>
			Haoran Wang, Lei Wang, Haobo Xu, Ying Wang, Yuming Li, and Yinhe Han (Chinese Academy of Sciences)
	    </p>
	    <p>
	    	<b>Labels:</b> ML training optimization
	    </p>
		<p> 
			<b>Abstract:</b>
			With the rapid up-scaling of transformer-based large language models (LLM), training these models is becoming increasingly demanding on novel parallel training techniques. Tensor partitioning is an extensively researched parallel technique, encompassing data and model parallelism, and has a significant influence on LLM training performance. However, existing state-of-the-art parallel training systems are based on incomplete tensor partitioning space, where the distribution of partitioned sub-operators is limited to the spatial dimension. We discover that introducing the temporal dimension into tensor partitioning of LLM training instance provides extra opportunities to avoid collective communication across devices, saving memory space and also overlapping device-to-device communication with computation. In this paper, we propose a new tensor partition primitive that distributes sub-operators along both the spatial and temporal dimensions to further explore communication and memory overhead reduction over current solutions. This new primitive creates a broader parallelization space and leads to parallel solutions that achieve better training throughput with lower peak memory occupancy compared to state-of-the-art techniques. To efficiently deploy optimized parallel transformer model training to multiple devices, we further present an optimization algorithm that can find optimal parallel solutions from our spatial-temporal tensor partition space with acceptable search time. Our evaluation shows that our optimized tensor partitioning achieves up to 1.68 × training throughput with 69% peak memory occupancy compared to state-of-the-art distributed training systems when training LLMs. Upon scaling to 32 GPUs, the geo-mean speedup across benchmarks is 1.30 ×. When applied in 3D parallelism, up to 1.46 × training throughput can be achieved.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620666.3651369">EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree</a></b>
	    <p>
			Zhaodong Chen (University of California Santa Barbara)<br>
			Andrew Kerr, Richard Cai, Jack Kosaian, and Haicheng Wu (NVIDIA)<br>
			Yufei Ding (University of California San Diego)<br>
			Yuan Xie (The Hong Kong University of Science and Technology)
	    </p>
	    <p>
	    	<b>Labels:</b> ML training optimization
	    </p>
		<p> 
			<b>Abstract:</b>
			As deep learning models become increasingly complex, the deep learning compilers are critical for enhancing the system efficiency and unlocking hidden optimization opportunities. Although excellent speedups have been achieved in inference workloads, existing compilers face significant limitations in training. Firstly, the training computation graph involves intricate operations challenging to fuse, such as normalization, loss functions, and reductions, which limit optimization opportunities like kernel fusion. Secondly, the training graph’s additional edges connecting forward and backward operators pose challenges in finding optimal and feasible partitions for kernel fusion. More importantly, existing compilers cannot either generate kernels with state-of-the-art performance on modern GPUs or accommodate diverse fusion patterns. In this paper, we introduce Epilogue Visitor Tree (EVT), a novel compiler that overcomes these limitations. EVT employs novel graph-level compilation passes to unlock hidden fusion and optimization opportunities. It also incorporates a Work partially done during the internship at NVIDIA The source code to reproduce the results is publically available at https: //github.com/apuaaChen/EVT_AE. The CUDA templates of EVT are actively maintained under the official CUTLASS repository https://github. com/NVIDIA/cutlass. novel integer linear programming-based partitioner that efficiently solves the optimal and feasible partitions in complex joint forward-backward graphs. Moreover, we present the Epilogue Visitor Abstraction and introduce the EVT operator compiler that automatically generates flexible epilogues that can be integrated with high-performance main loop implementations from CUTLASS and other SOTA libraries. EVT is evaluated on diverse training workloads across domains and achieves 1.26∼3.1× speedup.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3620665.3640399">Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training</a></b>
	    <p>
			Hongzheng Chen (Cornell University)<br>
			Cody Hao Yu and Shuai Zheng (Boson AI)<br>
			Zhen Zhang (Amazon Web Services)<br>
			Zhiru Zhang (Cornell University)<br>
			Yida Wang (Amazon Web Services)
	    </p>
	    <p>
	    	<b>Labels:</b> ML training optimization
	    </p>
		<p> 
			<b>Abstract:</b>
			Recent years have seen an increase in the development of large deep learning (DL) models, which makes training efficiency crucial. Common practice is struggling with the trade-off between usability and performance. On one hand, DL frameworks such as PyTorch use dynamic graphs to facilitate model developers at a price of sub-optimal model training performance. On the other hand, practitioners propose various approaches to improving the training efficiency by sacrificing some of the flexibility, ranging from making the graph static for more thorough optimization (e.g., XLA) to customizing optimization towards large-scale distributed training (e.g., DeepSpeed and Megatron-LM). In this paper, we aim to address the tension between usability and training efficiency through separation of concerns. Inspired by DL compilers that decouple the platform-specific optimizations of a tensor-level operator from its arithmetic definition, this paper proposes a schedule language, Slapo, to decouple model execution from definition. Specifically, Slapo works on a PyTorch model and uses a set of schedule primitives to convert the model for common model training optimizations such as high-performance kernels, effective 3D parallelism, and efficient activation checkpointing. Compared to existing optimization solutions, Slapo progressively optimizes the model “as-needed” through high-level primitives, and thus preserving programmability and debuggability for users to a large extent. Our evaluation results show that by scheduling the existing hand-crafted optimizations in a systematic way using Slapo, we are able to improve training throughput by up to 2.92× on a single machine with 8 NVIDIA V100 GPUs, and by up to 1.41× on multiple machines with up to 64 GPUs, when compared to the out-of-the-box performance of DeepSpeed and Megatron-LM.
		</p>
		<p></p>
	</td>
  </tr>

</table>


<a name="HPCA'24"></a>
2. [HPCA'24](https://www.hpca-conf.org/2024/program/main.php)

<table>
  <tr>
  	<td>
	    <b><a href="https://ieeexplore.ieee.org/abstract/document/10476398">Enabling Large Dynamic Neural Network Training with Learning-based Memory Management</a></b>
	    <p>
			Jie Ren (William & Mary); 
			<br>
			Dong Xu and Shuangyan Yang (University of California, Merced); 
			<br>
			Jiacheng Zhao and Zhicheng Li (University of Chinese Academy of Sciences); 
			<br>
			Christian Navasca (University of California, Los Angeles); 
			<br>
			Chenxi Wang (University of Chinese Academy of Sciences); 
			<br>
			Harry Xu (University of California, Los Angeles); 
			<br>
			Dong Li (University of California, Merced)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system; memory management
	    </p>
		<p> 
			<b>Abstract:</b>
			Dynamic neural network (DyNN) enables high computational efficiency and strong representation capability. However, training DyNN can face a memory capacity problem because of increasing model size or limited GPU memory capacity. Managing tensors to save GPU memory is challenging, because of the dynamic structure of DyNN. We present DyNN-Offload, a memory management system to train DyNN. DyNN-Offload uses a learned approach (using a neural network called the pilot model) to increase predictability of tensor accesses to facilitate memory management. The key of DyNN-Offload is to enable fast inference of the pilot model in order to reduce its performance overhead, while providing high inference (or prediction) accuracy. DyNNOffload reduces input feature space and model complexity of the pilot model based on a new representation of DyNN; DyNNOffload converts the hard problem of making prediction for individual operators into a simpler problem of making prediction for a group of operators in DyNN. DyNN-Offload enables 8 × larger DyNN training on a single GPU compared with using PyTorch alone (unprecedented with any existing solution). Evaluating with AlphaFold (a production-level, large-scale DyNN), we show that DyNN-Offload outperforms unified virtual memory (UVM) and dynamic tensor rematerialization (DTR), the most advanced solutions to save GPU memory for DyNN, by 3 × and 2.1 × respectively in terms of maximum batch size.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/html/2311.15269">Tetris: Boosting Distributed DNN Execution with Flexible Schedule Search</a></b>
	    <p>
			Zhiqi Lin (University of Science and Technology of China);
			<br>
			Youshan Miao (Microsoft Research);
			<br>
			Guanbin Xu and Cheng Li (University of Science and Technology of China);
			<br>
			Olli Saarikivi, Saeed Maleki, Fan Yang (Microsoft Research)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system; scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Increasingly complex and diverse deep neural network (DNN) models necessitate distributing the execution across multiple devices for training and inference tasks, and also require carefully planned schedules for performance. However, existing practices often rely on predefined schedules that may not fully exploit the benefits of emerging diverse model-aware operator placement strategies. Handcrafting high-efficiency schedules can be challenging due to the large and varying schedule space. This paper presents Tessel, an automated system that searches for efficient schedules for distributed DNN training and inference for diverse operator placement strategies. To reduce search costs, Tessel leverages the insight that the most efficient schedules often exhibit repetitive pattern (repetend) across different data inputs. This leads to a two-phase approach: repetend construction and schedule completion. By exploring schedules for various operator placement strategies, Tessel significantly improves both training and inference performance. Experiments with representative DNN models demonstrate that Tessel achieves up to 5.5× training performance speedup and up to 38% inference latency reduction.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://ieeexplore.ieee.org/document/10476417">SpecFL: An Efficient Speculative Federated Learning System for Tree-based Model Training</a></b>
	    <p>
			Yuhui Zhang, Lutan Zhao, Cheng Che (Key Laboratory of Cyberspace Security Defense, Institute of Information Engineering, CAS and School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China); 
			<br>
			XiaoFeng Wang (Indiana University Bloomington, Bloomington, America); 
			<br>
			Dan Meng, Rui Hou (Key Laboratory of Cyberspace Security Defense, Institute of Information Engineering, CAS and School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China)
	    </p>
	    <p>
	    	<b>Labels:</b> FL system; scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Increasingly complex and diverse deep neural network (DNN) models necessitate distributing the execution across multiple devices for training and inference tasks, and also require carefully planned schedules for performance. However, existing practices often rely on predefined schedules that may not fully exploit the benefits of emerging diverse model-aware operator placement strategies. Handcrafting high-efficiency schedules can be challenging due to the large and varying schedule space. This paper presents Tessel, an automated system that searches for efficient schedules for distributed DNN training and inference for diverse operator placement strategies. To reduce search costs, Tessel leverages the insight that the most efficient schedules often exhibit repetitive pattern (repetend) across different data inputs. This leads to a two-phase approach: repetend construction and schedule completion. By exploring schedules for various operator placement strategies, Tessel significantly improves both training and inference performance. Experiments with representative DNN models demonstrate that Tessel achieves up to 5.5× training performance speedup and up to 38% inference latency reduction.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://ieeexplore.ieee.org/abstract/document/10476450">Enhancing Collective Communication in MCM Accelerators for Deep Learning Training</a></b>
	    <p>
			Sabuj Laskar, Pranati Majhi, Sungkeun Kim, Farabi Mahmud, Abdullah Muzahid, Eun Jung Kim (Texas A&M University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML communication
	    </p>
		<p> 
			<b>Abstract:</b>
			With the widespread adoption of Deep Learning (DL) models, the demand for DL accelerator hardware has risen. On top of that, DL models are becoming massive in size. To accommodate those models, multi-chip-module (MCM) emerges as an effective approach for implementing large-scale DL accelerators. While MCMs have shown promising results for DL inference, its potential for Deep Learning Training remains largely unexplored. Current approaches fail to fully utilize available links in a mesh interconnection network of an MCM accelerator. To address this issue, we propose two novel AllReduce algorithms for mesh-based MCM accelerators - RingBiOdd and Three Tree Overlap (TTO). RingBiOdd is a ring-based algorithm that enhances the bandwidth of AllReduce by creating two unidirectional rings using bidirectional interconnects. On the other hand, TTO is a tree-based algorithm that improves AllReduce performance by overlapping data chunks. TTO constructs three topology-aware disjoint trees and runs different steps of the AllReduce operation in parallel. We present a detailed design and implementation of the proposed approaches. Our experimental results over seven DL models indicate that RingBiOdd achieves 50% and 8% training time reduction over unidirectional Ring AllReduce and MultiTree. Furthermore, TTO demonstrates 33% and 29% training time reduction over state-ofthe-art MultiTree and Bidirectional Ring AllReduce, respectively.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://ieeexplore.ieee.org/abstract/document/10476424">LibPreemptible: Enabling Fast, Adaptive, and Hardware-Assisted User-Space Scheduling</a></b>
	    <p>
			Yueying Li (Cornell University)<br>
			Nikita Lazarev (Massachusetts Institute of Technology)<br>
			David Koufaty (Intel Labs) <br>
			Tenny Yin (Cornell University) <br> 
			Andy Anderson (Intel Labs) <br>
			Zhiru Zhang (Cornell University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system; scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			With the widespread adoption of Deep Learning (DL) models, the demand for DL accelerator hardware has risen. On top of that, DL models are becoming massive in size. To accommodate those models, multi-chip-module (MCM) emerges as an effective approach for implementing large-scale DL accelerators. While MCMs have shown promising results for DL inference, its potential for Deep Learning Training remains largely unexplored. Current approaches fail to fully utilize available links in a mesh interconnection network of an MCM accelerator. To address this issue, we propose two novel AllReduce algorithms for mesh-based MCM accelerators - RingBiOdd and Three Tree Overlap (TTO). RingBiOdd is a ring-based algorithm that enhances the bandwidth of AllReduce by creating two unidirectional rings using bidirectional interconnects. On the other hand, TTO is a tree-based algorithm that improves AllReduce performance by overlapping data chunks. TTO constructs three topology-aware disjoint trees and runs different steps of the AllReduce operation in parallel. We present a detailed design and implementation of the proposed approaches. Our experimental results over seven DL models indicate that RingBiOdd achieves 50% and 8% training time reduction over unidirectional Ring AllReduce and MultiTree. Furthermore, TTO demonstrates 33% and 29% training time reduction over state-ofthe-art MultiTree and Bidirectional Ring AllReduce, respectively.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://ieeexplore.ieee.org/abstract/document/10476471">Ursa: Lightweight Resource Management for Cloud-Native Microservices</a></b>
	    <p>
			Yanqi Zhang, Zhuangzhuang Zhou (Cornell University) <br> 
			Sameh Elnikety (Microsoft Research) <br>
			Christina Delimitrou (MIT)
	    </p>
	    <p>
	    	<b>Labels:</b> microservice; scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			With the widespread adoption of Deep Learning (DL) models, the demand for DL accelerator hardware has risen. On top of that, DL models are becoming massive in size. To accommodate those models, multi-chip-module (MCM) emerges as an effective approach for implementing large-scale DL accelerators. While MCMs have shown promising results for DL inference, its potential for Deep Learning Training remains largely unexplored. Current approaches fail to fully utilize available links in a mesh interconnection network of an MCM accelerator. To address this issue, we propose two novel AllReduce algorithms for mesh-based MCM accelerators - RingBiOdd and Three Tree Overlap (TTO). RingBiOdd is a ring-based algorithm that enhances the bandwidth of AllReduce by creating two unidirectional rings using bidirectional interconnects. On the other hand, TTO is a tree-based algorithm that improves AllReduce performance by overlapping data chunks. TTO constructs three topology-aware disjoint trees and runs different steps of the AllReduce operation in parallel. We present a detailed design and implementation of the proposed approaches. Our experimental results over seven DL models indicate that RingBiOdd achieves 50% and 8% training time reduction over unidirectional Ring AllReduce and MultiTree. Furthermore, TTO demonstrates 33% and 29% training time reduction over state-ofthe-art MultiTree and Bidirectional Ring AllReduce, respectively.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://ieeexplore.ieee.org/document/10476433">LightPool: A NVMe-oF-based High-performance and Lightweight Storage Pool Architecture for Cloud-Native Distributed Database</a></b>
	    <p>
			Jiexiong Xu, Yiquan Chen (Zhejiang University, Alibaba Group) <br>
			Yijing Wang, Wenhui Shi, Guoju Fang (Alibaba Group) <br> 
			Yi Chen (Zhejiang University)
	    </p>
	    <p>
	    	<b>Labels:</b> storage, distributed database, nvme
	    </p>
		<p> 
			<b>Abstract:</b>
			Emerging cloud-native distributed databases rely on local NVMe SSDs to provide high-performance and highavailable data services to many cloud applications. However, the database clusters suffer from low utilization of local storage because of the imbalance between CPU and storage capacities within each node. For instance, the OceanBase distributed database cluster, with hundreds of PB local storage capacity, only utilizes around 40% of its local storage. Although disaggregated storage (EBS) can enhance storage utilization by provisioning the CPU and storage independently on demand, they suffer from performance bottlenecks and high costs. In this paper, we propose LightPool, a high-performance and lightweight storage pool architecture large-scale deployed in the OceanBase clusters, enhancing storage resource utilization. The key idea of LightPool is aggregating cluster storage into a storage pool and enabling unified management. In particular, LightPool adopts NVMe-oF to enable high-performance storage resource sharing among cluster nodes and integrate the storage pool with Kubernetes to achieve flexible management and allocation of storage resources. Furthermore, we design the hot-upgrade and hot-migration mechanisms to enhance the availability of LightPool. We have deployed LightPool on over 8500 nodes in production clusters. Statistics show that LightPool can improve storage resource utilization from about 40% to 65%. Experimental results show that the extra latency from LightPool is only about 2.1 μs compared to local storage. Compared to OpenEBS, LightPool enhances bandwidth up to 190.9% in microbenchmarks and throughput up to 6.9% in real-world applications. LightPool is the best practice to deploy NVMe-oF (NVMe/TCP) in the production environment. We also discuss important lessons and experiences learned from the development of LightPool.
		</p>
		<p></p>
	</td>
  </tr>

</table>



<a name="HPDC'24"></a>
3. [HPDC'24](http://hpdc.org/2024/program.html)

<table>
  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658658">ETS: Deep Learning Training Iteration Time Prediction based on Execution Trace Sliding Window</a></b>
	    <p>
			Zichao Yang, Hao GUO, Heng Wu, Yuewen WU, Hua Zhong, Wenbo ZHANG, (University of Chinese Academy of Sciences) <br>
			Chuan Zhou (Minzu University of China) <br>
			Yan Liu (Inspur Software Co., Ltd.)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system
	    </p>
		<p> 
			<b>Abstract:</b>
			Deep learning (DL) has become essential across various computer science domains. Accurately predicting iteration time for DL models in diverse cloud data center environments is critical for making high-quality scheduling decisions. Existing approaches neglect the sequential features inherent in the runtime execution, leading to issues such as overlooking DL framework overhead and struggling to handle diverse sizes of DL models, resulting in either low accuracy or slow convergence of the prediction model. This paper introduces ETS, a novel iteration time prediction method utilizing execution trace sliding windows. Our observation reveals that DL models exhibit a highly sequential runtime execution nature. Building upon this insight, we leverage sliding windows to extract a novel type of sequential features from the runtime execution trace. These features comprehensively capture DL framework overhead and address the diversity challenge in DL model sizes. By combining a best-practice method to train a prediction model, we achieve high accuracy and rapid convergence simultaneously. Experimental validation on over 14,000 DL model configurations demonstrates ETS's effectiveness in predicting the iteration time of DL models, achieving a mere 5.9% prediction error with a training time at the 10-minute level, and improving scheduling outcomes by reducing job completion time by 17%.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658687">FASOP: Fast yet Accurate Automated Search for Optimal Parallelization of Transformers on Heterogeneous GPU Clusters</a></b>
	    <p>
			Sunyeol Hwang, Eungyeong Lee, Hongseok Oh, Youngmin Yi (University of Seoul)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system
	    </p>
		<p> 
			<b>Abstract:</b>
			Transformer-based large language models have recently shown remarkable performance, but their significantly large parameters require efficient training, which is commonly realized by utilizing both data- and model-parallel deep learning on a GPU cluster. To minimize the training time, the optimal degrees of data and model parallelisms and the optimal model partitioning should be searched. When heterogeneous GPU clusters are used to utilize as many GPUs as possible, it becomes more challenging. In this work, we propose a framework named FASOP that automatically and rapidly finds the (near-)optimal degrees of parallelisms and model partitioning of Transformer-based models on heterogeneous GPU clusters, with an accurate estimation of pipelining latency and communications. Moreover, it can search for optimal cluster configurations that minimize the training time while satisfying the cost of GPU clusters. The proposed model partitioning algorithm in FASOP is three orders of magnitude faster than Dynamic Programming in the state-of-the-art for GPT-2 1.5B on a mixed set of 32 GPUs with A100 and A10, leading to a few seconds instead of several hours. And, FASOP shows only 8.7% mean absolute error in training time estimation for GPT-2 1.5B. With a fast yet accurate search, FASOP achieved up to 1.37× speedup compared to Megatron-LM.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658688">Loki: A System for Serving ML Inference Pipelines with Hardware and Accuracy Scaling</a></b>
	    <p>
			Sohaib Ahmad , Hui Guan , Ramesh K. Sitaraman (University of Massachusetts Amherst)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			The rapid adoption of machine learning (ML) has underscored the importance of serving ML models with high throughput and resource efficiency. Traditional approaches to managing increasing query demands have predominantly focused on hardware scaling, which involves increasing server count or computing power. However, this strategy can often be impractical due to limitations in the available budget or compute resources. As an alternative, accuracy scaling offers a promising solution by adjusting the accuracy of ML models to accommodate fluctuating query demands. Yet, existing accuracy scaling techniques target independent ML models and tend to underperform while managing inference pipelines. Furthermore, they lack integration with hardware scaling, leading to potential resource inefficiencies during low-demand periods. To address the limitations, this paper introduces Loki, a system designed for serving inference pipelines effectively with both hardware and accuracy scaling. Loki incorporates an innovative theoretical framework for optimal resource allocation and an effective query routing algorithm, aimed at improving system accuracy and minimizing latency deadline violations. Our empirical evaluation demonstrates that through accuracy scaling, the effective capacity of a fixed-size cluster can be enhanced by more than 2.7× compared to relying solely on hardware scaling. When compared with state-of-the-art inference-serving systems, Loki achieves up to a 10× reduction in Service Level Objective (SLO) violations, with minimal compromises on accuracy and while fulfilling throughput demands.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658657">ESG: Pipeline-Conscious Efficient Scheduling of DNN Workflows on Serverless Platforms with Shareable GPUs</a></b>
	    <p>
			Xinning Hui (North Carolina State University) <br>
			Yuanchao Xu (University of California, Santa Cruz) <br>
			Zhishan Guo , Xipeng Shen (North Carolina State University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Recent years have witnessed increasing interest in machine learning inferences on serverless computing for its auto-scaling and cost effective properties. Existing serverless computing, however, lacks effective job scheduling methods to handle the schedule space dramatically expanded by GPU sharing, task batching, and intertask relations. Prior solutions have dodged the issue by neglecting some important factors, leaving some large performance potential locked. This paper presents ESG, a new scheduling algorithm that directly addresses the difficulties. ESG treats sharable GPU as a first-order factor in scheduling. It employs an optimality-guided adaptive method by combining A*-search and a novel dual-blade pruning to dramatically prune the scheduling space without compromising the quality. It further introduces a novel method, dominator-based SLO distribution, to ensure the scalability of the scheduler. The results show that ESG can significantly improve the SLO hit rates (61%-80%) while saving 47%-187% costs over prior work.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658654">ElasticRoom: Multi-Tenant DNN Inference Engine via Co-design with Resource-constrained Compilation and Strong Priority Scheduling</a></b>
	    <p>
			Lixian Ma , Haoruo Chen , En Shao , Leping Wang (State Key Lab of Processors, Institute of Computing Technology) <br>
			Quan Chen (Shanghai Jiao Tong University) <br>
			Guangming Tan (State Key Lab of Processors, Institute of Computing Technology)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			GPU partition mechanisms in run-time software have been widely used in job scheduler and multi-tenant computing system to improve resource utilization and throughput. The latency requirements of different DNN requests, such as real-time and best-effort requests, often exhibit variations in computational systems that handle batch tasks for DNN inference. However, the existing GPU partition mechanisms and state-of-the-art scheduling approaches face challenges in effectively promising both high throughput and low latency for real-time requests. The current limitation lies in the inability of existing GPU partition mechanisms to enhance GPU resource utilization and ensure job priority simultaneously.
			<br>
			In this paper, we present an innovative multi-tenant DNN inference engine, ElasticRoom, which relies on the co-design with resource-constrained compilation and strong priority scheduling to achieve high GPU utilization and low latency of real-time requests simultaneously. To ensure portability across diverse manufacturers' accelerator hardware, ElasticRoom does not rely on any customization or pre-set features in the hardware or operating system. To quantify the ability of DNN inference computing systems to process and meet performance requirements for a batch of real-time DNN inference requests within a valid time, we define the concept of Goodput for each batch of inference requests. The performance of ElasticRoom was assessed on both NVIDIA GPUs (A100) and AMD GPUs (MI100), revealing significant enhancements in Goodput ranging from 14% to 49% compared to well-established state-of-the-art methods.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658693">Near-Optimal Wafer-Scale Reduce</a></b>
	    <p>
			Piotr Luczynski , Lukas Gianinazzi , Patrick Iff , Leighton Wilson , Daniele De Sensi , Torsten Hoefler (Department of Computer Science, ETH Zurich)
	    </p>
	    <p>
	    	<b>Labels:</b> Forkjoin, scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			Efficient Reduce and AllReduce communication collectives are a critical cornerstone of high-performance computing (HPC) applications. We present the first systematic investigation of Reduce and AllReduce on the Cerebras Wafer-Scale Engine (WSE). This architecture has been shown to achieve unprecedented performance both for machine learning workloads and other computational problems like FFT. We introduce a performance model to estimate the execution time of algorithms on the WSE and validate our predictions experimentally for a wide range of input sizes. In addition to existing implementations, we design and implement several new algorithms specifically tailored to the architecture. Moreover, we establish a lower bound for the runtime of a Reduce operation on the WSE. Based on our model, we automatically generate code that achieves near-optimal performance across the whole range of input sizes. Experiments demonstrate that our new Reduce and AllReduce algorithms outperform the current vendor solution by up to 3.27×. Additionally, our model predicts performance with less than 4% error. The proposed communication collectives increase the range of HPC applications that can benefit from the high throughput of the WSE. Our model-driven methodology demonstrates a disciplined approach that can lead the way to further algorithmic advancements on wafer-scale architectures.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658656">Efficient all-to-all Collective Communication Schedules for Direct-connect Topologies</a></b>
	    <p>
			Prithwish Basu , Liangyu Zhao , Jason Fantl , Siddharth Pal (RTX BBN Technologies)<br>
			Arvind Krishnamurthy (University of Washington) <br>
			Joud Khoury (RTX BBN Technologies)
	    </p>
	    <p>
	    	<b>Labels:</b> Forkjoin, scheduling
	    </p>
		<p> 
			<b>Abstract:</b>
			The all-to-all collective communications primitive is widely used in machine learning (ML) and high performance computing (HPC) workloads, and optimizing its performance is of interest to both ML and HPC communities. All-to-all is a particularly challenging workload that can severely strain the underlying interconnect bandwidth at scale. This paper takes a holistic approach to optimize the performance of all-to-all collective communications on supercomputer-scale direct-connect interconnects. We address several algorithmic and practical challenges in developing efficient and bandwidth-optimal all-to-all schedules for any topology and lowering the schedules to various runtimes and interconnect technologies. We also propose a novel topology that delivers near-optimal all-to-all performance.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658690">ScaleDFS: Accelerating Decentralized and Private File Sharing via Scaling Directed Acyclic Graph Processing</a></b>
	    <p>
			Mansub Song , Lan Anh Nguyen , Sunggon Kim , Hyeonsang Eom (Seoul National University)<br>
			Yongseok Son (Chung-Ang University)
	    </p>
	    <p>
	    	<b>Labels:</b> decentralized storage
	    </p>
		<p> 
			<b>Abstract:</b>
			This paper introduces a novel file system, ScaleDFS, designed to accelerate decentralized file sharing for a private network, leveraging the potential of scaling file management based on directed acyclic graph (DAG) on modern hardware. Specifically, in ScaleDFS, we first design a DAG builder that parallelizes the construction of DAG nodes for a file while preserving critical orders to speed up the uploading process. Second, we introduce a DAG reader that retrieves leaf DAG nodes in parallel without graph traversal assisted by a devised DAG cache to accelerate the downloading process. Finally, we present a DAG remover that rapidly identifies obsolete DAG nodes/data and removes them in parallel to mitigate the garbage collection overhead without compromising consistency. We implement ScaleDFS based on IPFS and demonstrate that ScaleDFS outperforms IPFS by up to 3.7×, 1.8×, and 12.6× in realistic file, private blockchain, and gateway workloads, respectively.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658679">EvoStore: Towards Scalable Storage of Evolving Learning Models</a></b>
	    <p>
			Robert Underwood (Argonne National Laboratory)<br>
			Meghana Madhyastha, Randal Burns (Johns Hopkins University)<br>
			Bogdan Nicolae (Argonne National Laboratory)
	    </p>
	    <p>
	    	<b>Labels:</b> scalable storage, ML system
	    </p>
		<p> 
			<b>Abstract:</b>
			Deep Learning (DL) has seen rapid adoption in all domains. Since training DL models is expensive, both in terms of time and resources, application workflows that make use of DL increasingly need to operate with a large number of derived learning models, which are obtained through transfer learning and fine-tuning. At scale, thousands of such derived DL models are accessed concurrently by a large number of processes. In this context, an important question is how to design and develop specialized DL model repositories that remain scalable under concurrent access, while addressing key challenges: how to query the DL model architectures for specific patterns? How to load/store a subset of layers/tensors from a DL model? How to efficiently share unmodified layers/tensors between DL models derived from each other through transfer learning? How to maintain provenance and answer ancestry queries? State of art leaves a gap regarding these challenges. To fill this gap, we introduce EvoStore, a distributed DL model repository with scalable data and metadata support to store and access derived DL models efficiently. Large-scale experiments on hundreds of GPUs show significant benefits over state-of-art with respect to I/O and metadata performance, as well as storage space utilization.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658816">Fast, Accurate and Distributed Simulation of novel HPC systems incorporating ARM and RISC-V CPUs</a></b>
	    <p>
			N. Tampouratzis , I. Papaefstathiou (Exascale Performance Systems, Heraklion, Greece)
	    </p>
	    <p>
	    	<b>Labels:</b> hardware simulation, CPU simulation
	    </p>
		<p> 
			<b>Abstract:</b>
			The growing developments of HPC systems used in a plethora of domains (healthcare, financial services, government and defense, energy) triggers an urgent demand for simulation frameworks that can simulate, in an integrated manner, both processing and network components of an HPC system-under-design (SuD). The main problem, however, is that, currently, there is a shortage of simulation frameworks that can handle the simulation of actual HPC systems, including the hardware, complete software stack and network dynamics in an integrated manner. In this work we start from the first known, open-source, fully-distributed Cloud simulation framework, COSSIM, and, as part of the RED-SEA1 and Vitamin-V2 European projects, we extend it so as to be able to accurately simulate HPC systems. The extended simulator has been evaluated when executing the very-widely used HPCG & LAMMPS benchmarks on both ARM & RISC-V architectures; the results demonstrate that the presented approach has up to 95% accuracy in the reported SuD aspects.
		</p>
		<p></p>
	</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658832">A runtime infrastructure for the Continuum of Computing</a></b>
	    <p>
			Edoardo Tinto, Tullio Vardanega (University of Padova, Padova, Italy)
	    </p>
	    <p>
	    	<b>Labels:</b> continuum of computing
	    </p>
		<p> 
			<b>Abstract:</b>
			Devices at the Edge of the network are experiencing a considerable increase in computational resources. At the same time, connectivity becomes more pervasive. These phenomena jointly facilitate the emergence of a new computational model, increasingly referred to as the Continuum of Computing. This model aims at including Edge resources in Cloud-like (and Cloud-inclusive) resource pooling to accommodate computations that need reduced latency, increased privacy, and general mobility. This model has the potential to enhance the power and the reach of high-performance computing (HPC) applications, making them extend up to the Edge of the network. However, managing a pool of resources that span across both Cloud and Edge nodes poses new challenges. Moving data across the network generates latency and security issues, while national policies may outright limit data mobility. This suggests moving computation towards data instead of the usual opposite. Enabling migrating computation is one of key traits of the envisioned Continuum of Computing. The vast heterogeneity in the technological stacks and the lack of uniform standards, however, hinder the deployment of applications in the Continuum. The availability of a common runtime environment across all host nodes of the Continuum is an obvious way to circumvent those problems, reviving the write-once-run-anywhere promise in that context. The ability to move computations opportunistically after user-specific performance objectives is another key trait of the Continuum model, which also is a foundation to spatial computing, a context-aware and space-aware computing paradigm. How to effectively orchestrate migrating computations so that they can deliver value added to their users is still an open question. There is a general understanding that Cloud-native orchestrators perform poorly when shifting towards the Edge, due to exceedingly restrictive (Cloud-centric) assumptions underneath their orchestration model. The matter of efficient orchestration in the Continuum is paramount in the envisioned model. To showcase the feasibility and viability of a Continuum-worthy runtime infrastructure, we singled out two emerging technologies: Rust and WebAssembly. The Rust programming language's highlight is its statically-checked memory safety. WebAssembly's highlights are solid guarantees of isolation and a portable bytecode format for applications compiled for its Instruction Set Architecture (ISA). To this project, WebAssembly components written in Rust constitute the candidate building blocks for the Continuum infrastructure, centred on memory-safe and sand-boxed execution capsules. In addition to that, this project aims to develop and deploy Continuum-worthy orchestration capabilities that leverage seamless migration.
			<br>
			The initial results of this project suggest that applications written in Rust and executing as WebAssembly components offer greater isolation and memory safety compared to containerized applications. Moreover, this novel approach might easily support live migration, consisting of the migration of an executing application into a different hosting node, preserving the state of the computation. Live migration prevents re-execution, and, at the time of writing, is largely unsupported in modern industrially applied containerized solutions. Supporting migrating computations might benefit multiple application scenarios. For example, the ability to migrate computation instead of freezing it during a low-energy phase may be of interest to energy-harvesting systems, Similarly, urgent science and Internet of Things (IoT) applications might want to move across Cloud and Edge nodes opportunistically, seeking optimal trade-offs between heavy and low-latency types of computation.
		</p>
		<p></p>
	</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658833">Efficient Stream Join Processing: Novel Approaches and Challenges</a></b>
	    <p>
			Adeel Aslam, Giovanni Simonini (University of Modena Reggio Emilia, Italy, Modena, Italy)
	    </p>
	    <p>
	    	<b>Labels:</b> Forkjoin, streaming
	    </p>
		<p> 
			<b>Abstract:</b>
			Stream join is a fundamental data operator for processing real-time data, but it faces computational challenges during stream inequality join (theta join operators) due to frequent updates in indexing data structures. To tackle this problem, we identify three key insights: 1) identifying skewed data distributions in real-time and implementing dedicated indexing structures for skewed keys to reduce index update costs; 2) leveraging optimized data structures, including insert-efficient mutable and search-efficient immutable structures to optimize the search stream join process and 3) adopting learned indexes instead of conventional ones, which can provide up to 4x better performance.
			<br>
			In this Ph.D. work, we propose novel solutions for distributed and multi-core stream join processing, including an indexing solution that uses a space-efficient dedicated filter and a two-stage data structure that effectively holds and processes sliding window items (bounded streaming contents). We are also exploring the adoption and benefits of learned indexes for real-time stream join processing. Despite non-trivial challenges like state management for distributed processing, processing guarantees, and efficient concurrency mechanisms, experiments on distributed stream processing systems show superior performance compared to state-of-the-art solutions.
		</p>
		<p></p>
	</td>
  </tr>

</table>


<a name="EuroSys'25"></a>
4. [EuroSys'25](https://2025.eurosys.org/accepted-papers.html#pagetop)

<table>



  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/abs/10.1145/3625549.3658833">Flex: Fast, Accurate DNN Inference on Low-Cost Edges Using Heterogeneous Accelerator Execution</a></b>
	    <p>
			Tanmoy Sen (University of Virginia)<br>
			Haiying Shen (University of Virginia)<br> 
			Anand Iyer (Georgia Tech)
	    </p>
	    <p>
	    	<b>Labels:</b> DNN inference, ML system, edge
	    </p>
			<p> 
				<b>Abstract:</b>
				Stream join is a fundamental data operator for processing real-time data, but it faces computational challenges during stream inequality join (theta join operators) due to frequent updates in indexing data structures. To tackle this problem, we identify three key insights: 1) identifying skewed data distributions in real-time and implementing dedicated indexing structures for skewed keys to reduce index update costs; 2) leveraging optimized data structures, including insert-efficient mutable and search-efficient immutable structures to optimize the search stream join process and 3) adopting learned indexes instead of conventional ones, which can provide up to 4x better performance.
				<br>
				In this Ph.D. work, we propose novel solutions for distributed and multi-core stream join processing, including an indexing solution that uses a space-efficient dedicated filter and a two-stage data structure that effectively holds and processes sliding window items (bounded streaming contents). We are also exploring the adoption and benefits of learned indexes for real-time stream join processing. Despite non-trivial challenges like state management for distributed processing, processing guarantees, and efficient concurrency mechanisms, experiments on distributed stream processing systems show superior performance compared to state-of-the-art solutions.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://mivenhan.github.io/publication/2025bless/2025bless.pdf">Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing</a></b>
	    <p>
			Shulai Zhang, Quan Chen, Weihao Cui, Han Zhao, Chunyu Xue (Shanghai Jiao Tong University)</br>
			Zhen Zheng (Microsoft)<br>
			Wei Lin (Alibaba Group)<br>
			Minyi Guo (Shanghai Jiao Tong University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, GPU sharing
	    </p>
			<p> 
				<b>Abstract:</b>
				Data centers now allow multiple applications that have lightweight workloads to share a GPU. Existing temporal or spatial sharing systems struggle to provide efficient and accurate quota assignments. We observe that the performance of the multi-user system is often underestimated because of the existence of unused GPU “bubbles” and can be enhanced by squeezing the bubbles. Based on this observation, we design Bless, a bubble-less spatial-temporal sharing GPU system that fine-tunes the GPU resource allocation to improve multi-user performance. Bless leverages precise computing resource management and fine-grained kernel scheduling to ensure stringent quota guarantees and reduce latency fairly for applications with varying GPU quotas. We implement and evaluate Bless with multiple applications and workloads. Our result shows that Bless achieves 21.1% − 37.3% average latency reduction over the state-of-the-art while guaranteeing the promised quota for all applications.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2409.19488">A House United Within Itself: SLO-Awareness for On-Premises Containerized ML Inference Clusters via Faro</a></b>
	    <p>
			Beomyeol Jeon (University of Illinois Urbana-Champaign)<br>
			Chen Wang, Diana Arroyo, Alaa Youssef (IBM Research)<br>
			Indranil Gupta (University of Illinois Urbana-Champaign)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, SLO-aware inference scheduling
	    </p>
			<p> 
				<b>Abstract:</b>
				This paper tackles the challenge of running multiple ML inference jobs (models) under time-varying workloads, on a constrained on-premises production cluster. Our system Faro takes in latency Service Level Objectives (SLOs) for each job, auto-distills them into utility functions, "sloppifies" these utility functions to make them amenable to mathematical optimization, automatically predicts workload via probabilistic prediction, and dynamically makes implicit cross-job resource allocations, in order to satisfy cluster-wide objectives, e.g., total utility, fairness, and other hybrid variants. A major challenge Faro tackles is that using precise utilities and high-fidelity predictors, can be too slow (and in a sense too precise!) for the fast adaptation we require. Faro's solution is to "sloppify" (relax) its multiple design components to achieve fast adaptation without overly degrading solution quality. Faro is implemented in a stack consisting of Ray Serve running atop a Kubernetes cluster. Trace-driven cluster deployments show that Faro achieves 2.3×-23× lower SLO violations compared to state-of-the-art systems.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2410.05004">Fast State Restoration in LLM Serving with HCache</a></b>
	    <p>
			Shiwei Gao, Youmin Chen, Jiwu Shu (Tsinghua University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system,
	    </p>
			<p> 
				<b>Abstract:</b>
				The growing complexity of LLM usage today, e.g., multi-round conversation and retrieval-augmented generation (RAG), makes contextual states (i.e., KV cache) reusable across user requests. Given the capacity constraints of GPU memory, only a limited number of contexts can be cached on GPU for reusing. Existing inference systems typically evict part of the KV cache and restore it by recomputing it from the original tokens or offloading it to host storage for later retrieval, both of which introduce substantial computational or I/O overheads. We propose HCache, a novel LLM state restoration method. Its key idea is to restore LLM states from intermediate activations and thus utilize computational and I/O resources with low overhead. We enhance HCache with two techniques, including i) a bubble-free restoration scheduler that integrates resource-complementary methods to optimize the balance between computation and IO tasks; and ii) a chunk-based storage manager to address the layout mismatch issue (i.e., layer-before-token saving versus token-before-layer restoration). Our evaluations, conducted using real-world tasks, show that HCache reduces the TTFT by up to 1.93X compared to KV offload while consuming 1.92-2.40X less storage space; compared to token recomputation, HCache achieves up to 5.73X reduction in TTFT.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b>Multiplexing Dynamic Deep Learning Workloads with SLO-awareness in GPU Clusters</b>
	    <p>
			Wenyan Chen (University of Macau; Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences)<br>
			Chengzhi Lu (Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences; University of Chinese Academy of Sciences; University of Macau)<br>
			Huanle Xu (University of Macau, Macau SAR, China)<br>
			Kejiang Ye (Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences)<br>
			ChengZhong Xu (University of Macau)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system,
	    </p>
			<p> 
				<b>Abstract:</b>
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="">HybridFlow: A Flexible and Efficient RLHF Framework</a></b>
	    <p>
			Guangming Sheng (The University of Hong Kong)<br>
			Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin (ByteDance)<br>
			Chuan Wu (The University of Hong Kong)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, RL
	    </p>
			<p> 
				<b>Abstract:</b>Reinforcement Learning from Human Feedback (RLHF) is widely used in Large Language Model (LLM) alignment. Traditional RL can be modeled as a dataflow, where each node represents computation of a neural network (NN) and each edge denotes data dependencies between the NNs. RLHF complicates the dataflow by expanding each node into a distributed LLM training or generation program, and each edge into a many-to-many multicast. Traditional RL frameworks execute the dataflow using a single controller to instruct both intra-node computation and inter-node communication, which can be inefficient in RLHF due to large control dispatch overhead for distributed intra-node computation. Existing RLHF systems adopt a multi-controller paradigm, which can be inflexible due to nesting distributed computation and data communication. We propose HybridFlow, which combines single-controller and multi-controller paradigms in a hybrid manner to enable flexible representation and efficient execution of the RLHF dataflow. We carefully design a set of hierarchical APIs that decouple and encapsulate computation and data dependencies in the complex RLHF dataflow, allowing efficient operation orchestration to implement RLHF algorithms and flexible mapping of the computation onto various devices. We further design a 3D-HybridEngine for efficient actor model resharding between training and generation phases, with zero memory redundancy and significantly reduced communication overhead. Our experimental results demonstrate 1.53×∼20.57× throughput improvement when running various RLHF algorithms using HybridFlow, as compared with state-of-the-art baselines. HybridFlow source code will be available at https://github.com/volcengine/verl
			</p>
			<p><a href="https://github.com/volcengine/verl">open source</a></p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b>JABAS: Joint Adaptive Batching and Automatic Scaling for DNN Training on Heterogeneous GPUs</b>
	    <p>
			Gyeongchan Yun, Junesoo Kang, Hyunjoon Jeong, Sanghyeon Eom (UNIST)<br>
			Minsung Jang (Samsung SDS)<br>
			Young-ri Choi (UNIST)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, heterogeneous GPU
	    </p>
			<p> 
				<b>Abstract:</b>
			</p>
			<p><a href="https://github.com/unist-ssl/JABAS">open source</a></p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://open.bu.edu/items/e280aa29-da44-495c-8094-29ddf2e7f944">CAPSys: Contention-aware task placement for data stream processing</a></b>
	    <p>
			Gyeongchan Yun, Junesoo Kang, Hyunjoon Jeong, Sanghyeon Eom (UNIST)<br>
			Minsung Jang (Samsung SDS)<br>
			Young-ri Choi (UNIST)
	    </p>
	    <p>
	    	<b>Labels:</b> task placement, scheduling, streaming
	    </p>
			<p> 
				<b>Abstract:</b> In the context of streaming dataflow queries, the task placement problem aims to identify a mapping of operator tasks to physical resources in a distributed cluster. We show that task placement not only significantly affects query performance but also the convergence and accuracy of auto-scaling controllers. We propose CAPSys, an adaptive resource controller for dataflow stream processors, that considers auto-scaling and task placement in concert. CAPSys relies on Contention-Aware Placement Search (CAPS), a new placement strategy that ensures compute-intensive, I/O-intensive, and network-intensive tasks are balanced across available resources. We integrate CAPSys with Apache Flink and show that it consistently achieves higher throughput and lower backpressure than Flink’s strategies, while it also improves the convergence of the DS2 auto-scaling controller under variable workloads. When compared with the state-of-the-art ODRP placement strategy, CAPSys computes the task placement in orders of magnitude lower time and achieves up to 6× higher throughput.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://open.bu.edu/items/e280aa29-da44-495c-8094-29ddf2e7f944">CAPSys: Contention-aware task placement for data stream processing</a></b>
	    <p>
			Gyeongchan Yun, Junesoo Kang, Hyunjoon Jeong, Sanghyeon Eom (UNIST)<br>
			Minsung Jang (Samsung SDS)<br>
			Young-ri Choi (UNIST)
	    </p>
	    <p>
	    	<b>Labels:</b> task placement, scheduling, streaming
	    </p>
			<p> 
				<b>Abstract:</b> In the context of streaming dataflow queries, the task placement problem aims to identify a mapping of operator tasks to physical resources in a distributed cluster. We show that task placement not only significantly affects query performance but also the convergence and accuracy of auto-scaling controllers. We propose CAPSys, an adaptive resource controller for dataflow stream processors, that considers auto-scaling and task placement in concert. CAPSys relies on Contention-Aware Placement Search (CAPS), a new placement strategy that ensures compute-intensive, I/O-intensive, and network-intensive tasks are balanced across available resources. We integrate CAPSys with Apache Flink and show that it consistently achieves higher throughput and lower backpressure than Flink’s strategies, while it also improves the convergence of the DS2 auto-scaling controller under variable workloads. When compared with the state-of-the-art ODRP placement strategy, CAPSys computes the task placement in orders of magnitude lower time and achieves up to 6× higher throughput.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2312.05516">Stateful Large Language Model Serving with Pensieve</a></b>
	    <p>
			Lingfan Yu, Jinkun Lin, Jinyang Li (New York University)
	    </p>
	    <p>
	    	<b>Labels:</b> LLM serving, ML system, stateful LLM
	    </p>
			<p> 
				<b>Abstract:</b> Large Language Models (LLMs) are wildly popular today and it is important to serve them efficiently. Existing LLM serving systems are stateless across requests. Consequently, when LLMs are used in the common setting of multi-turn conversations, a growing log of the conversation history must be processed alongside any request by the serving system at each turn, resulting in repeated processing. 
				<br>
				In this paper, we design 𝑃𝑒𝑛𝑠𝑖𝑒𝑣𝑒, a system optimized for multi-turn conversation LLM serving. 𝑃𝑒𝑛𝑠𝑖𝑒𝑣𝑒 maintains the conversation state across requests by caching previously processed history to avoid duplicate processing. 𝑃𝑒𝑛𝑠𝑖𝑒𝑣𝑒’s multi-tier caching strategy can utilize both GPU and CPU memory to efficiently store and retrieve cached data. 𝑃𝑒𝑛𝑠𝑖𝑒𝑣𝑒 also generalizes the recent PagedAttention kernel to support attention between multiple input tokens with a GPU cache spread over non-contiguous memory. Our evaluation shows that 𝑃𝑒𝑛𝑠𝑖𝑒𝑣𝑒 can achieve 1.14-3.0× the throughput of vLLM and TensorRT-LLM and significantly reduce latency.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2312.05516">FlowCheck: Decoupling Checkpointing and Training of Large-Scale Models</a></b>
	    <p>
				Zimeng Huang (Shanghai Jiao Tong University & Alibaba Cloud)<br>
				Hao Nie (Alibaba Cloud & Peking University)<br>
				Haonan Jia (Alibaba Cloud)<br>
				Bo Jiang (Shanghai Jiao Tong University)<br>
				Junchen Guo, Jianyuan Lu, Rong Wen (Alibaba Cloud)<br>
				Biao Lyu (Zhejiang University & Alibaba Cloud)<br>
				Shunmin Zhu (Hangzhou Feitian Cloud & Alibaba Cloud)<br>
				Xinbing Wang (Shanghai Jiao Tong University)
	    </p>
	    <p>
	    	<b>Labels:</b> DL training, ML system
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p><a href="https://github.com/AlibabaResearch/flowcheck-eurosys25">open source</a></p>
			<p></p>
		</td>
  </tr>



  <tr>
  	<td>
	    * <b><a href="https://web.njit.edu/~dingxn/papers/vsched.pdf">Optimizing Task Scheduling in Cloud VMs with Accurate vCPU Abstraction</a></b>
	    <p>
				Edward Guo (Hofstra University)<br>
				Weiwei Jia (The University of Rhode Island)<br>
				Xiaoning Ding (New Jersey Institute of Technology)<br>
				Jianchen Shan (Hofstra University)
	    </p>
	    <p>
	    	<b>Labels:</b> task scheduling, cloud computing
	    </p>
			<p> 
				<b>Abstract:</b> The paper shows that task scheduling in Cloud VMs hasn’t evolved quickly to handle the dynamic vCPU resources. The existing vCPU abstraction cannot accurately depict the vCPU dynamics in capacity, activity, and topology, and these mismatches can mislead the scheduler, causing performance degradation and system anomalies. The paper proposes a novel solution, vSched, which probes accurate vCPU abstraction through a set of lightweight microbenchmarks (vProbers) without modifying the hypervisor, and leverages the probed information to optimize task scheduling in cloud VMs with three new techniques: biased vCPU selection, intra-VM harvesting, and relaxed work conservation. Our evaluation of vSched’s implementation in x86 Linux Kernel demonstrates that it can effectively improve both system throughput and workload latency across various VM types in the dynamic multi-cloud environment.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://cs.nju.edu.cn/changxu/1_publications/25/EUROSYS25.pdf">Understanding the Linux Kernel, Visually</a></b>
	    <p>
				Hanzhi Liu, Yanyan Jiang, Chang Xu, Hanzhi Liu (Nanjing University)
	    </p>
	    <p>
	    	<b>Labels:</b> linux kernel, kernel debugging
	    </p>
			<p> 
				<b>Abstract:</b> Understanding the Linux kernel is challenging due to its large and complex program state. While existing kernel debugging tools provide full access to kernel states at arbitrary levels of detail, developers often spend a significant amount of time sifting through redundant information to find what is truly useful. Additionally, the textual results provided by traditional debuggers are often insufficient for expressing high-dimensional information in a readable manner. This paper presents Visualinux, the first debugging framework that can simplify the program state of the Linux kernel to a level that can be visually understood with low programming complexity and effort. Visualinux includes a domainspecific language for specifying simplifications of a kernel object graph, an SQL-like domain-specific language for customizing the simplified object graph, and a panel-based interactive debugger. Evaluation results show that Visualinux can visualize various complex kernel components and efficiently assist developers in diagnosing sophisticated kernel bugs.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2212.13228">DPack: Efficiency-Oriented Privacy Budget Scheduling</a></b>
	    <p>
				Pierre Tholoniat, Kelly Kostopoulou (Columbia University)<br>
				Mosharaf Chowdhury (University of Michigan)<br>
				Asaf Cidon (Columbia University)<br>
				Roxana Geambasu (Columbia University)<br>
				Mathias Lécuyer (University of British Columbia)<br>
				Junfeng Yang (Columbia University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, privacy scheduling
	    </p>
			<p> 
				<b>Abstract:</b> Machine learning (ML) models can leak information about users, and differential privacy (DP) provides a rigorous way to bound that leakage under a given budget. This DP budget can be regarded as a new type of compute resource in workloads of multiple ML models training on user data. Once it is used, the DP budget is forever consumed. Therefore, it is crucial to allocate it most efficiently to train as many models as possible. This paper presents the scheduler for privacy that optimizes for efficiency. We formulate privacy scheduling as a new type of multidimensional knapsack problem, called privacy knapsack, which maximizes DP budget efficiency. We show that privacy knapsack is NP-hard, hence practical algorithms are necessarily approximate. We develop an approximation algorithm for privacy knapsack, DPack, and evaluate it on microbenchmarks and on a new, synthetic private-ML workload we developed from the Alibaba ML cluster trace. We show that DPack: (1) often approaches the efficiency-optimal schedule, (2) consistently schedules more tasks compared to a state-of-the-art privacy scheduling algorithm that focused on fairness (1.3-1.7x in Alibaba, 1.0-2.6x in microbenchmarks), but (3) sacrifices some level of fairness for efficiency. Therefore, using DPack, DP ML operators should be able to train more models on the same amount of user data while offering the same privacy guarantee to their users.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2405.16444">CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion</a></b>
	    <p>
				Jiayi Yao (Chinese University of Hong Kong (Shenzhen))<br>
				Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng (University of Chicago)<br>
				Qizheng Zhang (Stanford University)<br>
				Kuntai Du (University of Chicago)<br>
				Shan Lu (Microsoft Research and University of Chicago)<br>
				Junchen Jiang (University of Chicago)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, prefilling KV cache
	    </p>
			<p> 
				<b>Abstract:</b> Large language models (LLMs) often incorporate multiple text chunks in their inputs to provide the necessary contexts. To speed up the prefill of the long LLM inputs, one can pre-compute the KV cache of a text and re-use the KV cache when the context is reused as the prefix of another LLM input. However, the reused text chunks are not always the input prefix, and when they are not, their precomputed KV caches cannot be directly used since they ignore the text's cross-attention with the preceding text in the LLM input. Thus, the benefits of reusing KV caches remain largely unrealized.
				<br>
				This paper tackles just one question: when an LLM input contains multiple text chunks, how to quickly combine their precomputed KV caches in order to achieve the same generation quality as the expensive full prefill (i.e., without reusing KV cache)? We present CacheBlend, a scheme that reuses the pre-computed KV caches, regardless prefix or not, and selectively recomputes the KV values of a small subset of tokens to partially update each reused KV cache. In the meantime,the small extra delay for recomputing some tokens can be pipelined with the retrieval of KV caches within the same job,allowing CacheBlend to store KV caches in slower devices with more storage capacity while retrieving them without increasing the inference delay. By comparing CacheBlend with the state-of-the-art KV cache reusing schemes on three open-source LLMs of various sizes and four popular benchmark datasets of different tasks, we show that CacheBlend reduces time-to-first-token (TTFT) by 2.2-3.3X and increases the inference throughput by 2.8-5X, compared with full KV recompute, without compromising generation quality or incurring more storage cost.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    *** <b><a href="https://arxiv.org/abs/2405.16444">SpotHedge: Serving AI Models on Spot Instances</a></b>
	    <p>
				Ziming Mao, Tian Xia, Zhanghao Wu, Wei-Lin Chiang, Tyler Griggs, Romil Bhardwaj, Zongheng Yang (UC Berkeley)<br>
				Scott Shenker (ICSI AND UC Berkeley)<br>
				<b>Ion Stoica (UC Berkeley)</b>
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, spot instance, preemptive scheduling
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2405.16444">Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization</a></b>
	    <p>
				Zhanda Zhu (University of Toronto, CentML, Vector Institute)<br>
				Christina Giannoula (University of Toronto)<br>
				Muralidhar Andoorveedu (CentML)<br>
				Qidong Su (University of Toronto, CentML, Vector Institute)<br>
				Karttikeya Mangalam (UC Berkeley)<br>
				Bojian Zheng (Independent Researcher)<br>
				Gennady Pekhimenko (CentML, University of Toronto, Vector Institute)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, LLM training
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2410.03480">SeBS-Flow: Benchmarking Serverless Cloud Function Workflows</a></b>
	    <p>
				Larissa Schmid (Karlsruhe Institute of Technology)<br>
				Marcin Copik, Alexandru Calotoiu, Laurin Brandner (ETH Zurich)<br>
				Anne Koziolek (Karlsruhe Institute of Technology)<br>
				Torsten Hoefler (ETH Zurich)
	    </p>
	    <p>
	    	<b>Labels:</b> Serverless, FaaS, workflow, benchmarking
	    </p>
			<p> 
				<b>Abstract:</b> Serverless computing has emerged as a prominent paradigm, with a significant adoption rate among cloud customers. While this model offers advantages such as abstraction from the deployment and resource scheduling, it also poses limitations in handling complex use cases due to the restricted nature of individual functions. Serverless workflows address this limitation by orchestrating multiple functions into a cohesive application. However, existing serverless workflow platforms exhibit significant differences in their programming models and infrastructure, making fair and consistent performance evaluations difficult in practice. To address this gap, we propose the first serverless workflow benchmarking suite SeBS-Flow, providing a platform-agnostic workflow model that enables consistent benchmarking across various platforms. SeBS-Flow includes six real-world application benchmarks and four microbenchmarks representing different computational patterns. We conduct comprehensive evaluations on three major cloud platforms, assessing performance, cost, scalability, and runtime deviations. We make our benchmark suite open-source, enabling rigorous and comparable evaluations of serverless workflows over time.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2312.05215">DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs</a></b>
	    <p>
				Xiaozhe Yao (ETH Zurich)<br>
				Qinghao Hu (MIT)<br>
				Ana Klimovic (ETH Zurich)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, LLM serving
	    </p>
			<p> 
				<b>Abstract:</b> Fine-tuning large language models (LLMs) greatly improves model quality for downstream tasks. However, serving many fine-tuned LLMs concurrently is challenging due to the sporadic, bursty, and varying request patterns of different LLMs. To bridge this gap, we present DeltaZip, an LLM serving system that efficiently serves multiple full-parameter fine-tuned models concurrently by aggressively compressing model deltas by up to 10x while maintaining high model quality. The key insight behind this design is that fine-tuning results in small-magnitude changes to the pre-trained model. By co-designing the serving system with the compression algorithm, DeltaZip achieves 2x to 12x improvement in throughput compared to the state-of-the-art systems.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://zenodo.org/records/14862956">MEPipe: Democratizing LLM Training with Memory-Efficient Slice-Level Pipeline Scheduling on Cost-Effective Accelerators</a></b>
	    <p>
				Zhenbo Sun, Shengqi Chen, Yuanwei Wang, Jian Sha (Tsinghua University)<br>
				Guanyu Feng (Zhipu AI)<br>
				Wenguang Chen (Tsinghua University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, LLM training, pipeline scheduling
	    </p>
			<p> 
				<b>Abstract:</b> The training of large language models (LLMs) typically needs costly GPUs, such as NVIDIA A100 or H100. They possess substantial high-bandwidth on-chip memory and rapid interconnects like NVLinks. The exorbitant expenses associated with LLM training pose not just an economic challenge but also a eocietal one, as it inrestricts the ability to train LLMs from scratch to a select few organizations.
				<br>
				There is a significant interest in democratizing access to LLM training. This paper explores a potential solution by employing innovative parallel strategies on more affordable accelerators. Budget-friendly options like NVIDIA RTX 4090, while considerably less expensive and comparable in computational power to A100, are hindered by their limited memory capacity and reduced interconnect bandwidth, making the effective training of LLMs challenging.
				<br>
				Conventional parallel strategies often result in high communication costs or excessive memory usage. Our paper introduces MEPIPE, a novel approach that includes a slice-level scheduling method for sequence pipeline parallelism. This method minimizes memory consumption without incurring additional communication overhead. Besides, MEPIPE utilizes fine-grained weight gradient computation to reduce idle time and mitigate imbalanced computation among slices.
				<br>
				MEPIPE has demonstrated up to speedup (on average) on clusters equipped with 64 NVIDIA 4090 GPUs when training Llama models of varying sizes. 35\% Model FLOPS Utilization (MFU) is achieved in training Llama 13B model, being more cost-effective than A100 clusters.
			</p>
			<p><a href="https://zenodo.org/records/14862956">Artifacts</a></p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b>Towards VM Rescheduling Optimization Through Deep Reinforcement Learning</b>
	    <p>
				Xianzhong Ding (University of California, Merced)<br>
				Yunkai Zhang (University of California, Berkeley)<br>
				Binbin Chen (ByteDance)<br>
				Donghao Ying (UC Berkeley)<br>
				Tieying Zhang, Jianjun Chen, Lei Zhang (ByteDance)<br>
				Alberto Cerpa (University of California, Merced)<br>
				Wan Du (University of California Merced)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, RL, scheduling
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    ** <b><a href="https://www.microsoft.com/en-us/research/publication/tuna-tuning-unstable-and-noisy-cloud-applications/#:~:text=Autotuning%20plays%20a%20pivotal%20role,value%20stores%2C%20and%20operating%20systems.">TUNA: Tuning Unstable and Noisy Cloud Applications</a></b>
	    <p>
				Johannes Freischuetz, Konstantinos Kanellis (University of Wisconsin-Madison)<br>
				Brian Kroth (Microsoft)<br>
				Shivaram Venkataraman (University of Wisconsin-Madison)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, autotuning
	    </p>
			<p> 
				<b>Abstract:</b> Autotuning plays a pivotal role in optimizing the performance of systems, particularly in large-scale cloud deployments, and has been used to improve the performance of a number of systems including databases, key-value stores, and operating systems. We find that one of the main challenges in performing autotuning in the cloud arises from performance variability or noise in system measurements. We first investigate the extent to which noise slows down autotuning and find that as little as 5% noise can lead to a 2.5x slowdown in converging to the best-performing configuration We also measure the magnitude of noise in cloud computing settings and find that, while some components (CPU, disk) have almost no performance variability there are still sources of significant variability (caches, memory). Additionally, we find that variability leads to autotuning finding unstable configurations, where for some workloads as many as 63.3% of configurations selected as “best” during tuning can degrade by 30% or more when deployed. Using this as motivation, this paper proposes a novel approach to improve the efficiency of autotuning systems by (a) detecting and removing outlier configurations, and (b) using ML-based approaches to provide a more stable true signal of de-noised experiment results to the optimizer. The resulting system, TUNA (Tuning Unstable and Noisy Cloud Applications) enables faster convergence and robust configurations. We find that configurations learned using TUNA perform better and with lower standard deviations during deployment, as compared to traditional sampling methodologies. Tuning PostgreSQL running an enterprise production workload, we find that TUNA can lead to 1.88x lower running time on average with 2.58𝑥 lower standard deviation compared to traditional sampling methodologies.  TUNA will be incorporated into the MLOS(opens in new tab) project and has both artifacts(opens in new tab) and multiple(opens in new tab) datasets(opens in new tab) available.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b>SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs</b>
	    <p>
				Ruibo FAN, Xiangrui YU, Peijie Dong, Zeyu Li, Gu Gong (Data Science and Analytics Thrust, HKUST(GZ))<br>
				QIANG WANG (Harbin Institute of Technology (Shenzhen))<br>
				Wei Wang (Hong Kong University of Science and Technology)<br>
				Xiaowen Chu (Data Science and Analytics Thrust, HKUST(GZ))<br>
				Wei Wang (Hong Kong University of Science and Technology)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, LLM inference, GPU
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b>Eva: Cost-Efficient Cloud-Based Cluster Scheduling</b>
	    <p>
				Tzu-Tao Chang (University of Wisconsin-Madison)<br>
				Shivaram Venkataraman (University of Wisconsin-Madison)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, scheduling
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    * <b>NeuStream: Bridging Deep Learning Serving and Stream Processing</b>
	    <p>
				Haochen Yuan (Peking University)<br>
				Yuanqing Wang (Peking University and Microsoft Research)<br>
				Wenhao Xie (Peking University)<br>
				Yu Cheng (Peking University and Microsoft Research)<br>
				Ziming Miao, Lingxiao Ma, Jilong Xue (Microsoft Research)<br>
				Zhi Yang (Peking University)
	    </p>
	    <p>
	    	<b>Labels:</b> streaming, DL serving
	    </p>
			<p> 
				<b>Abstract:</b> 
			</p>
			<p></p>
		</td>
  </tr>

</table>







<a name="NSDI'24"></a>
5. [NSDI'24](https://www.usenix.org/conference/nsdi24/technical-sessions)

<table>

  <tr>
  	<td>
	    *** <b><a href="https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao">Can’t Be Late: Optimizing Spot Instance Savings under Deadlines</a></b>
	    <p>
				Zhanghao Wu, Wei-Lin Chiang, Ziming Mao, and Zongheng Yang, (University of California, Berkeley)<br>
				Eric Friedman and Scott Shenker (University of California, Berkeley,
and ICSI)<br>
				Ion Stoica (University of California, Berkeley)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, spot instance, preemptive scheduling
	    </p>
	    <p><b><i>Awarded Outstanding Paper!</i></b></p>
			<p> 
				<b>Abstract:</b> Cloud providers offer spot instances alongside on-demand instances to optimize resource utilization. While economically appealing, spot instances’ preemptible nature causes them ill-suited for deadline-sensitive jobs. To allow jobs to meet deadlines while leveraging spot instances, we propose a simple idea: use on-demand instances judiciously as a backup resource. However, due to the unpredictable spot instance availability, determining when to switch between spot and on-demand to minimize cost requires careful policy design. In this paper, we first provide an in-depth characterization of spot instances (e.g., availability, pricing, duration), and develop a basic theoretical model to examine the worst and average-case behaviors of baseline policies (e.g., greedy). The model serves as a foundation to motivate our design of a simple and effective policy, Uniform Progress, which is parameter-free and requires no assumptions on spot availability. Our empirical study, based on three-month-long real spot availability traces on AWS, demonstrates that it can (1) outperform the greedy policy by closing the gap to the optimal policy by 2× in both average and bad cases, and (2) further reduce the gap when limited future knowledge is given. These results hold in a variety of conditions ranging from loose to tight deadlines, low to high spot availability, and on single or multiple instances. By implementing this policy on top of SkyPilot, an intercloud broker system, we achieve 27%-84% cost savings across a variety of representative real-world workloads and deadlines. The spot availability traces are open-sourced for future research.
			</p>
			<p><a href="https://github.com/skypilot-org/spot-traces">open source</a></p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    *** <b><a href="https://www.usenix.org/conference/nsdi24/presentation/duan">Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances</a></b>
	    <p>
				Jiangfei Duan (The Chinese University of Hong Kong)<br>
				Ziang Song (ByteDance)<br>
				Xupeng Miao and Xiaoli Xi (Carnegie Mellon University)<br>
				Dahua Lin (The Chinese University of Hong Kong)<br>
				Harry Xu (University of California, Los Angeles)<br>
				Minjia Zhang (Microsoft)<br>
				Zhihao Jia, Carnegie Mellon University
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, DNN training, spot instance, preemptive scheduling
	    </p>
			<p> 
				<b>Abstract:</b> Deep neural networks (DNNs) are becoming progressively large and costly to train. This paper aims to reduce DNN training costs by leveraging preemptible instances on modern clouds, which can be allocated at a much lower price when idle but may be preempted by the cloud provider at any time. Prior work that supports DNN training on preemptive instances employs a reactive approach to handling instance preemptions and allocations after their occurrence, which only achieves limited performance and scalability.
				<br>
				We present Parcae, a system that enables cheap, fast, and scalable DNN training on preemptible instances by proactively adjusting the parallelization strategy of a DNN training job to adapt to predicted resource changes before instance preemptions and allocations really happen, which significantly reduces the cost of handling these events. Parcae optimizes liveput, a novel metric that measures the expected training throughput of a DNN job under various possible preemption scenarios. Compared to existing reactive, throughput-optimized systems, Parcae's proactive, live-optimized solution considers both the throughput of a job and its robustness under preemptions. To optimize liveput, Parcae supports lightweight instance migration and uses an availability predictor to forecast future preemptions. It then uses a liveput optimizer to discover an optimal strategy to parallelize DNN training under predicted preemptions. We evaluate Parcae on a variety of DNNs and preemption traces and show that Parcae outperforms existing spot-instance DNN training systems by up to 10×. More importantly, Parcae achieves near-optimal performance for training large DNNs under frequent preemptions, in which case existing approaches cannot make any progress.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/wang-zibo">Autothrottle: A Practical Bi-Level Approach to Resource Management for SLO-Targeted Microservices</a></b>
	    <p>
				Zibo Wang (University of Science and Technology of China and Microsoft Research)<br>;
				inghe Li (ETH Zurich)<br>
				Chieh-Jan Mike Liang (Microsoft Research)<br>
				Feng Wu (University of Science and Technology of China)<br>
				Francis Y. Yan (Microsoft Research)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, serverless, microservices, SLO-targeted, resource scheduling
	    </p>
	    <p><b><i>Awarded Outstanding Paper!</i></b></p>
			<p> 
				<b>Abstract:</b> Achieving resource efficiency while preserving end-user experience is non-trivial for cloud application operators. As cloud applications progressively adopt microservices, resource managers are faced with two distinct levels of system behavior: end-to-end application latency and per-service resource usage. Translating between the two levels, however, is challenging because user requests traverse heterogeneous services that collectively (but unevenly) contribute to the end-to-end latency. We present Autothrottle, a bi-level resource management framework for microservices with latency SLOs (service-level objectives). It architecturally decouples application SLO feedback from service resource control, and bridges them through the notion of performance targets. Specifically, an application-wide learning-based controller is employed to periodically set performance targets—expressed as CPU throttle ratios—for per-service heuristic controllers to attain. We evaluate Autothrottle on three microservice applications, with workload traces from production scenarios. Results show superior CPU savings, up to 26.21% over the best-performing baseline and up to 93.84% over all baselines.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/zhang-zili-jolteon">Jolteon: Unleashing the Promise of Serverless for Serverless Workflows</a></b>
	    <p>
				Zili Zhang, Chao Jin, and Xin Jin (School of Computer Science, Peking University)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, serverless, microservices, workflow
	    </p>
			<p> 
				<b>Abstract:</b> Serverless computing promises automatic resource provisioning to relieve the burden of developers. Yet, developers still have to manually configure resources on current serverless platforms to satisfy application-level requirements. This is because cloud applications are orchestrated as serverless workflows with multiple stages, exhibiting a complex relationship between resource configuration and application requirements.
				<br>
				We propose Jolteon, an orchestrator to unleash the promise of automatic resource provisioning for serverless workflows. At the core of Jolteon is a stochastic performance model that combines the benefits of whitebox modeling to capture the execution characteristics of serverless computing and blackbox modeling to accommodate the inherent performance variability. We formulate a chance constrained optimization problem based on the performance model, and exploit sampling and convexity to find optimal resource configurations that satisfy user-defined cost or latency bounds. We implement a system prototype of Jolteon and evaluate it on AWS Lambda with a variety of serverless workflows. The experimental results show that Jolteon outperforms the state-of-the-art solution, Orion, by up to 2.3× on cost and 2.1× on latency.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/agarwal-saksham">Harmony: A Congestion-free Datacenter Architecture</a></b>
	    <p>
				Saksham Agarwal, Qizhe Cai, Rachit Agarwal, and David Shmoys (Cornell University)<br>
				Amin Vahdat (Google)
	    </p>
	    <p>
	    	<b>Labels:</b> congestion-free, datacenter, cloud computing, congestion control
	    </p>
			<p> 
				<b>Abstract:</b> Datacenter networks today provide best-effort delivery—messages may observe unpredictable queueing, delays, and drops due to switch buffer overflows within the network. Such weak guarantees reduce the set of assumptions that system designers can rely upon from the network, thus introducing inefficiency and complexity in host hardware and software.
				<br>
				We present Harmony, a datacenter network architecture that provides powerful "congestion-free" message delivery guarantees—each message, once transmitted by the sender, observes bounded queueing at each switch in the network. Thus, network delays are bounded in failure-free scenarios, and congestion-related drops are completely eliminated. We establish, both theoretically and empirically, that Harmony provides such powerful guarantees with near-zero overheads compared to best-effort delivery networks: it incurs a tiny additive latency overhead that diminishes with message sizes, while achieving near-optimal network utilization.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/hu">Characterization of Large Language Model Development in the Datacenter</a></b>
	    <p>
				Qinghao Hu (Shanghai AI Laboratory and S-Lab Nanyang Technological University) <br>
				Zhisheng Ye (Shanghai AI Laboratory and Peking University)<br>
				Zerui Wang (Shanghai AI Laboratory and Shanghai Jiao Tong University)<br>
				Guoteng Wang (Shanghai AI Laboratory)<br>
				Meng Zhang and Qiaoling Chen (Shanghai AI Laboratory and S-Lab, Nanyang Technological University)<br>
				Peng Sun (Shanghai AI Laboratory and SenseTime Research)<br>
				Dahua Lin (Shanghai AI Laboratory and CUHK)<br> 
				Xiaolin Wang and Yingwei Luo (Peking University)<br>
				Yonggang Wen and Tianwei Zhang (Nanyang Technological University)
	    </p>
	    <p>
	    	<b>Labels:</b> LLM deployment, resource scheduling
	    </p>
			<p> 
				<b>Abstract:</b> Large Language Models (LLMs) have presented impressive performance across several transformative tasks. However, it is non-trivial to efficiently utilize large-scale cluster resources to develop LLMs, often riddled with numerous challenges such as frequent hardware failures, intricate parallelization strategies, and imbalanced resource utilization. In this paper, we present an in-depth characterization study of a six-month LLM development workload trace collected from our GPU datacenter Acme. Specifically, we investigate discrepancies between LLMs and prior task-specific Deep Learning (DL) workloads, explore resource utilization patterns, and identify the impact of various job failures. Our analysis summarizes hurdles we encountered and uncovers potential opportunities to optimize systems tailored for LLMs. Furthermore, we introduce our system efforts: (1) fault-tolerant pretraining, which enhances fault tolerance through LLM-involved failure diagnosis and automatic recovery. (2) decoupled scheduling for evaluation, which achieves timely performance feedback via trial decomposition and scheduling optimization.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng">MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs</a></b>
	    <p>
				Ziheng Jiang and Haibin Lin (ByteDance)<br>
				Yinmin Zhong (Peking University)<br>
				Qi Huang, Yangrui Chen, Zhi Zhang, Yanghua Peng, Xiang Li, Cong Xie, Shibiao Nong, Yulu Jia, Sun He, Hongmin Chen, Zhihao Bai, Qi Hou, Shipeng Yan, Ding Zhou, Yiyao Sheng, Zhuo Jiang, Haohan Xu, Haoran Wei, Zhang Zhang, Pengfei Nie, Leqi Zou, Sida Zhao, Liang Xiang, Zherui Liu, Zhe Li, Xiaoying Jia, and Jianxi Ye (ByteDance)<br>
				Xin Jin (Peking University)<br>
				Xin Liu (ByteDance)
	    </p>
	    <p>
	    	<b>Labels:</b> LLM traning, resource scheduling
	    </p>
			<p> 
				<b>Abstract:</b> We present the design, implementation and engineering experience in building and deploying MegaScale, a production system for training large language models (LLMs) at the scale of more than 10,000 GPUs. Training LLMs at this scale brings unprecedented challenges to training efficiency and stability. We take a full-stack approach that co-designs the algorithmic and system components across model block and optimizer design, computation and communication overlapping, operator optimization, data pipeline, and network performance tuning. Maintaining high efficiency throughout the training process (i.e., stability) is an important consideration in production given the long extent of LLM training jobs. Many hard stability issues only emerge at large scale, and in-depth observability is the key to address them. We develop a set of diagnosis tools to monitor system components and events deep in the stack, identify root causes, and derive effective techniques to achieve fault tolerance and mitigate stragglers. MegaScale achieves 55.2% Model FLOPs Utilization (MFU) when training a 175B LLM model on 12,288 GPUs, improving the MFU by 1.34x compared to Megatron-LM. We share our operational experience in identifying and fixing failures and stragglers. We hope by articulating the problems and sharing our experience from a systems perspective, this work can inspire future LLM systems research.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/zu">Resiliency at Scale: Managing Google’s TPUv4 Machine Learning Supercomputer</a></b>
	    <p>
				Yazhou Zu, Alireza Ghaffarkhah, Hoang-Vu Dang, Brian Towles, Steven Hand, Safeen Huda, Adekunle Bello, Alexander Kolbasov, Arash Rezaei, Dayou Du, Steve Lacy, Hang Wang, Aaron Wisner, Chris Lewis, and Henri Bahini (Google)
	    </p>
	    <p>
	    	<b>Labels:</b> TPUv4, resource scheduling, ML supercomputer
	    </p>
			<p> 
				<b>Abstract:</b> TPUv4 (Tensor Processing Unit) is Google’s 3rd generation accelerator for machine learning training, deployed as a 4096-node supercomputer with a custom 3D torus interconnect. In this paper, we describe our experience designing and operating the software infrastructure that allows TPUv4 supercomputers to operate at scale, including features for automatic fault resiliency and hardware recovery. We adopt a software-defined networking (SDN) approach to manage TPUv4’s high-bandwidth inter-chip interconnect (ICI) fabric, using optical circuit switching to dynamically configure routes to work around machine, chip and link failures. Our infrastructure detects failures and automatically triggers reconfiguration to minimize disruption to running workloads, as well as initiating remediation and repair workflows for the affected components. Similar techniques interface with maintenance and upgrade workflows for both hardware and software. Our dynamic reconfiguration approach allows our TPUv4 supercomputers to achieve 99.98% system availability, gracefully handling hardware outages experienced by ~1% of the training jobs.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/zeng">Accelerating Neural Recommendation Training with Embedding Scheduling</a></b>
	    <p>
				Chaoliang Zeng, Xudong Liao, Xiaodian Cheng, Han Tian, Xinchen Wan, Hao Wang, and Kai Chen (iSING Lab, Hong Kong University of Science and Technology)
	    </p>
	    <p>
	    	<b>Labels:</b> recommendation system, DL training, scheduling, embedding scheduling
	    </p>
			<p> 
				<b>Abstract:</b> Deep learning recommendation models (DLRM) are extensively adopted to support many online services. Typical DLRM training frameworks adopt the parameter server (PS) in CPU servers to maintain memory-intensive embedding tables, and leverage GPU workers with embedding cache to accelerate compute-intensive neural network computation and enable fast embedding lookups. However, such distributed systems suffer from significant communication overhead caused by the embedding transmissions between workers and PS. Prior work reduces the number of cache embedding transmissions by compromising model accuracy, including oversampling hot embeddings or applying staleness-tolerant updates.
				<br>
				This paper reveals that many of such transmissions can be avoided given the predictability and infrequency natures of in-cache embedding accesses in distributed training. Based on this observation, we explore a new direction to accelerate distributed DLRM training without compromising model accuracy, i.e., embedding scheduling—with the core idea of proactively determining "where embeddings should be trained" and "which embeddings should be synchronized" to increase the cache hit rate and decrease unnecessary updates, thus achieving a low communication overhead. To realize this idea, we design Herald, a real-time embedding scheduler consisting of two main components: an adaptive location-aware inputs allocator to determine where embeddings should be trained and an optimal communication plan generator to determine which embeddings should be synchronized. Our experiments with real-world workloads show that Herald reduces 48%-89% embedding transmissions, leading up to 2.11× and up to 1.61× better performance with TCP and RDMA, respectively, over 100 Gbps Ethernet for end-to-end DLRM training.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/huang">ADISTMM: Accelerating Distributed Multimodal Model Training</a></b>
	    <p>
				Jun Huang (The Ohio State University)<br>
				Zhen Zhang (Amazon Web Services)<br>
				Shuai Zheng (Boson AI)<br>
				Feng Qin (The Ohio State University)<br>
				Yida Wang (Amazon Web Services)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, DL training
	    </p>
			<p> 
				<b>Abstract:</b> Multimodal model training takes multiple types of inputs to process with differently structured submodules, and aggregates outcomes from the submodules to learn the relationship among various types of inputs, e.g., correlating text to image for text-to-image generation. The differences of submodule architectures as well as their inputs lead to heterogeneity in terms of computation efficiency. Failing to account for such heterogeneity, existing distributed training systems treat all submodules as a monolithic entity and thus have sub-optimal performance. Moreover, the outcome aggregation phase introduces cross-sample dependencies with contrasting positive and negative sample pairs (i.e., contrastive loss). Such dependencies make the existing pipeline parallelism scheduling algorithms not applicable for multimodal training with contrastive loss.
				<br>
				To address the limitations of existing solutions, we propose DISTIMM. For a given multimodal model, DISTIMM exploits the heterogeneity among submodules, applying different distributed parallelism strategies for each submodule, e.g., using Tensor Parallelism for a computation-intensive submodule, and Data Parallelism for a submodule with a small number of parameters. DISTIMM balances the computation of parallelized submodules to reduce the computing resource idle time of waiting for the slowest submodule. DISTIMM further optimizes the locality of submodules by leveraging the heterogeneous bandwidth of interconnections among accelerators. To address the limitation of existing pipeline execution schedules, we propose a new pipeline execution primitive, called batch-sync instruction, and a corresponding schedule, called DISTIMM-Pipe. We build a prototype of DISTIMM and evaluate it with existing solutions on models with various sizes ranging from 1.1 billion to 26 billion parameters and observe 1.32-3.27 × speedup over Megatron-LM.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/lam">Accelerating Skewed Workloads With Performance Multipliers in the TurboDB Distributed Database</a></b>
	    <p>
				Jennifer Lam, Jeffrey Helt, and Wyatt Lloyd (Princeton University)<br>
				Haonan Lu (University at Buffalo)
	    </p>
	    <p>
	    	<b>Labels:</b> skewed workload, workload accelerating, distributed database
	    </p>
			<p> 
				<b>Abstract:</b> Distributed databases suffer from performance degradation under skewed workloads. Such workloads cause high contention, which is exacerbated by cross-node network latencies. In contrast, single-machine databases better handle skewed workloads because their centralized nature enables performance optimizations that execute contended requests more efficiently. Based on this insight, we propose a novel hybrid architecture that employs a single-machine database inside a distributed database and present TurboDB, the first distributed database that leverages this hybrid architecture to achieve up to an order of magnitude better performance than representative solutions under skewed workloads.
				<br>
				TurboDB introduces two designs to tackle the core challenges unique to its hybrid architecture. First, Hybrid Concurrency Control is a specialized technique that coordinates the single-machine and distributed databases to collectively ensure process-ordered serializability. Second, Phalanx Replication provides fault tolerance for the single-machine database without significantly sacrificing its performance benefits. We implement TurboDB using CockroachDB and Cicada as the distributed and single-machine databases, respectively. Our evaluation shows that TurboDB significantly improves the performance of CockroachDB under skewed workloads.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/wydrowski">Load is not what you should balance: Introducing Prequal</a></b>
	    <p>
				Bartek Wydrowski (Google Research)<br>
				Robert Kleinberg (Google Research and Cornell)<br>
				Stephen M. Rumble (Google (YouTube))<br>
				Aaron Archer (Google Research)
	    </p>
	    <p>
	    	<b>Labels:</b> load balancing, cloud computing
	    </p>
			<p> 
				<b>Abstract:</b> We present PReQuaL (Probing to Reduce Queuing and Latency), a load balancer for distributed multi-tenant systems. PReQuaL is designed to minimize real-time request latency in the presence of heterogeneous server capacities and non-uniform, time-varying antagonist load. To achieve this, PReQuaL actively probes server load and leverages the power of d choices paradigm, extending it with asynchronous and reusable probes. Cutting against received wisdom, PReQuaL does not balance CPU load, but instead selects servers according to estimated latency and active requests-in-flight (RIF). We explore the major design features of PReQuaL on a testbed system and describe our experience using it to balance load within YouTube, where it has been running for more than a year. PReQuaL has dramatically decreased tail latency, error rates, and resource use, enabling YouTube and other production systems at Google to run at much higher utilization.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/zhang-yiwen">Vulcan: Automatic Query Planning for Live ML Analytics</a></b>
	    <p>
				Yiwen Zhang and Xumiao Zhang (University of Michigan)<br>
				Ganesh Ananthanarayanan (Microsoft)<br>
				Anand Iyer (Georgia Institute of Technology)<br>
				Yuanchao Shu (Zhejiang University)<br>
				Victor Bahl (Microsoft Corporation)<br>
				Z. Morley Mao (University of Michigan and Google)<br>
				Mosharaf Chowdhury (University of Michigan)
	    </p>
	    <p>
	    	<b>Labels:</b> ML query, query planning
	    </p>
			<p> 
				<b>Abstract:</b> Live ML analytics have gained increasing popularity with large-scale deployments due to recent evolution of ML technologies. To serve live ML queries, experts nowadays still need to perform manual query planning, which involves pipeline construction, query configuration, and pipeline placement across multiple edge tiers in a heterogeneous infrastructure. Finding the best query plan for a live ML query requires navigating a huge search space, calling for an efficient and systematic solution.
				<br>
				In this paper, we propose Vulcan, a system that automatically generates query plans for live ML queries to optimize their accuracy, latency, and resource consumption. Based on the user query and performance requirements, Vulcan determines the best pipeline, placement, and query configuration for the query with low profiling cost; it also performs fast online adaptation after query deployment. Vulcan outperforms state-of-the-art ML analytics systems by 4.1×-30.1× in terms of search cost while delivering up to 3.3× better query latency.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/rajasekaran">CASSINI: Network-Aware Job Scheduling in Machine Learning Clusters</a></b>
	    <p>
				Sudarsanan Rajasekaran and Manya Ghobadi (Massachusetts Institute of Technology)<br>
				Aditya Akella (UT Austin)
	    </p>
	    <p>
	    	<b>Labels:</b> ML scheduling, resource scheduling, job scheduling
	    </p>
			<p> 
				<b>Abstract:</b> We present CASSINI, a network-aware job scheduler for machine learning (ML) clusters. CASSINI introduces a novel geometric abstraction to consider the communication pattern of different jobs while placing them on network links. To do so, CASSINI uses an Affinity graph that finds a series of time-shift values to adjust the communication phases of a subset of jobs, such that the communication patterns of jobs sharing the same network link are interleaved with each other. Experiments with 13 common ML models on a 24-server testbed demonstrate that compared to the state-of-the-art ML schedulers, CASSINI improves the average and tail completion time of jobs by up to 1.6x and 2.5x, respectively. Moreover, we show that CASSINI reduces the number of ECN marked packets in the cluster by up to 33x.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/feng-chengquan">LitePred: Transferable and Scalable Latency Prediction for Hardware-Aware Neural Architecture Search</a></b>
	    <p>
				Chengquan Feng (University of Science and Technology of China)<br>
				Li Lyna Zhang (Microsoft Research)<br>
				Yuanchi Liu (University of Science and Technology of China)<br>
				Jiahang Xu and Chengruidong Zhang (Microsoft Research)<br>
				Zhiyuan Wang (University of Science and Technology of China)<br>
				Ting Cao and Mao Yang (Microsoft Research)<br>
				Haisheng Tan (University of Science and Technology of China)
	    </p>
	    <p>
	    	<b>Labels:</b> edge computing, latency prediction, neural architecture search
	    </p>
			<p> 
				<b>Abstract:</b> Hardware-Aware Neural Architecture Search (NAS) has demonstrated success in automating the design of affordable deep neural networks (DNNs) for edge platforms by incorporating inference latency in the search process. However, accurately and efficiently predicting DNN inference latency on diverse edge platforms remains a significant challenge. Current approaches require several days to construct new latency predictors for each one platform, which is prohibitively time-consuming and impractical.
				<br>
				In this paper, we propose LitePred, a lightweight approach for accurately predicting DNN inference latency on new platforms with minimal adaptation data by transferring existing predictors. LitePred builds on two key techniques: (i) a Variational Autoencoder (VAE) data sampler to sample high-quality training and adaptation data that conforms to the model distributions in NAS search spaces, overcoming the out-of-distribution challenge; and (ii) a latency distribution-based similarity detection method to identify the most similar pre-existing latency predictors for the new target platform, reducing adaptation data required while achieving high prediction accuracy. Extensive experiments on 85 edge platforms and 6 NAS search spaces demonstrate the effectiveness of our approach, achieving an average latency prediction accuracy of 99.3% with less than an hour of adaptation cost. Compared with SOTA platform-specific methods, LitePred achieves up to 5.3% higher accuracy with a significant 50.6× reduction in profiling cost. Code and predictors are available at https://github.com/microsoft/Moonlit/tree/main/LitePred.
			</p>
			<p><a href="https://github.com/microsoft/Moonlit/tree/main/LitePred">open source</a></p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/cho">LDB: An Efficient Latency Profiling Tool for Multithreaded Applications</a></b>
	    <p>
				Inho Cho (MIT CSAIL)<br>
				Seo Jin Park (University of Southern California)<br>
				Ahmed Saeed (Georgia Tech)<br>
				Mohammad Alizadeh and Adam Belay (MIT CSAIL)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, latency profiling
	    </p>
			<p> 
				<b>Abstract:</b> Maintaining low tail latency is critical for the efficiency and performance of large-scale datacenter systems. Software bugs that cause tail latency problems, however, are notoriously difficult to debug. We present LDB, a new latency profiling tool that aims to overcome this challenge by precisely identifying the specific functions that are responsible for tail latency anomalies. LDB observes the latency of all functions in a running program. It uses a novel, software-only technique called stack sampling, where a busy-spinning stack scanner thread polls lightweight metadata recorded in the call stack, shifting tracing costs away from program threads. In addition, LDB uses event tagging to record requests, inter-thread synchronization, and context switching. This can be used, for example, to generate per-request timelines and to find the root cause of complex tail latency problems such as lock contention in multi-threaded programs. We evaluate LDB with three datacenter applications, finding latency problems in each. Our results further show that LDB produces actionable insights, has low overhead, and can rapidly analyze recordings, making it feasible to use in production settings.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    ** <b><a href="https://www.usenix.org/conference/nsdi24/presentation/peng">UFO: The Ultimate QoS-Aware Core Management for Virtualized and Oversubscribed Public Clouds</a></b>
	    <p>
				Yajuan Peng (Southern University of Science and Technology and Shenzhen Institutes of Advanced Technology, Chinese Academy of Science)<br>
				Shuang Chen and Yi Zhao (Shuhai Lab, Huawei Cloud)<br>
				Zhibin Yu (Shuhai Lab, Huawei Cloud, and Shenzhen Institutes of Advanced Technology, Chinese Academy of Science)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, QoS-Aware, CPU scheduling, over-committing, over-subscribed
	    </p>
			<p> 
				<b>Abstract:</b> Public clouds typically adopt (1) multi-tenancy to increase server utilization; (2) virtualization to provide isolation between different tenants; (3) oversubscription of resources to further increase resource efficiency. However, prior work all focuses on optimizing one or two elements, and fails to considerately bring QoS-aware multi-tenancy, virtualization and resource oversubscription together.
				<br>
				We find three challenges when the three elements coexist. First, the double scheduling symptoms are 10x worse with latency-critical (LC) workloads which are comprised of numerous sub-millisecond tasks and are significantly different from conventional batch applications. Second, inner-VM resource contention also exists between threads of the same VM when running LC applications, calling for inner-VM core isolation. Third, no application-level performance metrics can be obtained by the host to guide resource management in realistic public clouds.
				<br>
				To address these challenges, we propose a QoS-aware core manager dubbed UFO to specifically support co-location of multiple LC workloads in virtualized and oversubscribed public cloud environments. UFO solves the three above-mentioned challenges, by (1) coordinating the guest and host CPU cores (vCPU-pCPU coordination), and (2) doing fine-grained inner-VM resource isolation, to push core management in realistic public clouds to the extreme. Compared with the state-of-the-art core manager, it saves up to 50% (average of 22%) of physical cores under the same co-location scenario.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://www.usenix.org/conference/nsdi24/presentation/namyar-solving">Solving Max-Min Fair Resource Allocations Quickly on Large Graphs</a></b>
	    <p>
				Pooria Namyar (Microsoft and University of Southern California)<br>
				Behnaz Arzani and Srikanth Kandula (Microsoft)<br>
				Santiago Segarra (Microsoft and Rice University)<br>
				Daniel Crankshaw and Umesh Krishnaswamy (Microsoft)<br>
				Ramesh Govindan (University of Southern California)<br>
				Himanshu Raj (Microsoft)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, max-min fair resource allocation, resource scheduling, cluster scheduling
	    </p>
			<p> 
				<b>Abstract:</b> We consider the max-min fair resource allocation problem. The best-known solutions use either a sequence of optimizations or waterfilling, which only applies to a narrow set of cases. These solutions have become a practical bottleneck in WAN traffic engineering and cluster scheduling, especially at larger problem sizes. We improve both approaches: (1) we show how to convert the optimization sequence into a single fast optimization, and (2) we generalize waterfilling to the multi-path case. We empirically show our new algorithms Pareto-dominate prior techniques: they produce faster, fairer, and more efficient allocations. Some of our allocators also have theoretical guarantees: they trade off a bounded amount of unfairness for faster allocation. We have deployed our allocators in Azure's WAN traffic engineering pipeline, where we preserve solution quality and achieve a roughly 3× speedup.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://www.usenix.org/conference/nsdi24/presentation/friess">Cloudy with a Chance of Cyberattacks: Dangling Resources Abuse on Cloud Platforms</a></b>
	    <p>
				Jens Frieß (National Research Center for Applied Cybersecurity ATHENE and Technische Universität Darmstadt)<br>
				Tobias Gattermayer (National Research Center for Applied Cybersecurity ATHENE and Fraunhofer Institute for Secure Information Technology SIT)<br> 
				Nethanel Gelernter (IONIX)<br>
				Haya Schulmann (Goethe-Universität Frankfurt and National Research Center for Applied Cybersecurity ATHENE)<br>
				Michael Waidner (National Research Center for Applied Cybersecurity ATHENE and Technische Universität Darmstadt and Fraunhofer Institute for Secure Information Technology SIT)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, resource abuse, over-committing, resource hijack
	    </p>
			<p> 
				<b>Abstract:</b> Recent works showed that it is feasible to hijack resources on cloud platforms. In such hijacks, attackers can take over released resources that belong to legitimate organizations. It was proposed that adversaries could abuse these resources to carry out attacks against customers of the hijacked services, e.g., through malware distribution. However, to date, no research has confirmed the existence of these attacks.
				<br>
				We identify, for the first time, real-life hijacks of cloud resources. This yields a number of surprising and important insights. First, contrary to previous assumption that attackers primarily target IP addresses, our findings reveal that the type of resource is not the main consideration in a hijack. Attackers focus on hijacking records that allow them to determine the resource by entering freetext. The costs and overhead of hijacking such records are much lower than those of hijacking IP addresses, which are randomly selected from a large pool.
				<br>
				Second, identifying hijacks poses a substantial challenge. Monitoring resource changes, e.g., changes in content, is insufficient, since such changes could also be legitimate. Retrospective analysis of digital assets to identify hijacks is also arduous due to the immense volume of data involved and the absence of indicators to search for. To address this challenge, we develop a novel approach that involves analyzing data from diverse sources to effectively differentiate between malicious and legitimate modifications. Our analysis has revealed 20,904 instances of hijacked resources on popular cloud platforms. While some hijacks are short-lived (up to 15 days), 1/3 persist for more than 65 days.
				<br>
				We study how attackers abuse the hijacked resources and find that, in contrast to the threats considered in previous work, the majority of the abuse (75%) is blackhat search engine optimization. We also find fraudulent certificates and stolen cookies. We cluster the abuse resources and abuse content to identify about 1,800 individual attacking infrastructures.
			</p>
			<p></p>
		</td>
  </tr>

</table>


<a name="EuroSys'24"></a>
1. [EurySys'24](https://2024.eurosys.org/accepted-papers.html)


<table>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629565">Model Selection for Latency-Critical Inference Serving</a></b>
	    <p>
				Daniel Mendoza, Francisco Romero, Caroline Trippel (Stanford University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system
	    </p>
			<p> 
				<b>Abstract:</b> In an inference service system, model selection and scheduling (MS&S) schemes map inference queries to trained machine learning (ML) models, hosted on a finite set of workers, to solicit accurate predictions within strict latency targets. MS&S is challenged by both varying query load and stochastic query inter-arrival patterns; however, state-of-the-art MS&S approaches conservatively account for load exclusively.
				<br>
				In this paper, we first show that explicitly considering inter-arrival patterns creates opportunities to map queries to higher-accuracy (higher-latency) models during intermittent arrival lulls. We then propose RAMSIS, a framework for generating MS&S policies that exploits this finding. RAMSIS leverages a statistical problem model of query load and inter-arrival pattern to produce policies that maximize accuracy given some latency target. We evaluate RAMSIS-generated MS&S policies alongside state-of-the-art approaches. Notably, RAMSIS requires as low as 50.00% (on average 18.77%) fewer resources to achieve the same accuracy for an ImageNet image classification task given 26 trained models
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3627703.3629569">Enoki: High Velocity Linux Kernel Scheduler Development</a></b>
	    <p>
				Samantha Miller (University of Washington), Anirudh Kumar (University of Washington), Tanay Vakharia (University of Washington), Ang Chen (University of Michigan), Danyang Zhuo (Duke University), Thomas Anderson (University of Washington)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system
	    </p>
			<p> 
				<b>Abstract:</b> Kernel task scheduling is important for application performance, adaptability to new hardware, and complex user requirements. However, developing, testing, and debugging new scheduling algorithms in Linux, the most widely used cloud operating system, is slow and difficult. We developed Enoki, a framework for high velocity development of Linux kernel schedulers. Enoki schedulers are written in safe Rust, and the system supports live upgrade of new scheduling policies into the kernel, userspace debugging, and bidirectional communication with applications. A scheduler implemented with Enoki achieved near identical performance (within 1% on average) to the default Linux scheduler CFS on a wide range of benchmarks. Enoki is also able to support a range of research schedulers, specifically the Shinjuku scheduler, a locality aware scheduler, and the Arachne core arbiter, with good performance.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3627703.3629572">CDMPP: A Device-Model Agnostic Framework for Latency Prediction of Tensor Programs</a></b>
	    <p>
				Hanpeng Hu, Junwei Su, Juntao Zhao (The University of Hong Kong)<br>
				Yanghua Peng, Yibo Zhu, Haibin Lin (ByteDance Inc.)<br>
				Chuan Wu (The University of Hong Kong)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, latency prediction
	    </p>
			<p> 
				<b>Abstract:</b> Deep Neural Networks (DNNs) have shown excellent performance in a wide range of machine learning applications. Knowing the latency of running a DNN model or tensor program on a specific device is useful in various tasks, such as DNN graph- or tensor-level optimization and device selection. Considering the large space of DNN models and devices that impedes direct profiling of all combinations, recent efforts focus on building a predictor to model the performance of DNN models on different devices. However, none of the existing attempts have achieved a cost model that can accurately predict the performance of various tensor programs while supporting both training and inference accelerators. We propose CDMPP, an efficient tensor program latency prediction framework for both cross-model and cross-device prediction. We design an informative but efficient representation of tensor programs, called compact ASTs, and a pre-order-based positional encoding method, to capture the internal structure of tensor programs. We develop a domain-adaption-inspired method to learn domain-invariant representations and devise a KMeans-based sampling algorithm, for the predictor to learn from different domains (i.e., different DNN operators and devices). Our extensive experiments on a diverse range of DNN models and devices demonstrate that CDMPP significantly outperforms state-of-the-art baselines with 14.03% and 10.85% prediction error for cross-model and cross-device prediction, respectively, and one order of magnitude higher training efficiency. The implementation and the expanded dataset are available at https://github.com/joapolarbear/cdmpp.
			</p>
			<p><a href="https://github.com/joapolarbear/cdmpp">open source</a></p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/abs/10.1145/3627703.3629573">Concealing Compression-accelerated I/O for HPC Applications through In Situ Task Scheduling</a></b>
	    <p>
				Sian Jin (Indiana University)<br>
				Sheng Di (Argonne National Laboratory)<br>
				Frédéric Vivien (INRIA, France)<br>
				Daoce Wang (Indiana University)<br>
				Yves Robert (Ecole Normale Supérieure de Lyon, France)<br>
				Dingwen Tao (Indiana University)<br>
				Franck Cappello (Argonne National Laboratory)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, latency prediction
	    </p>
			<p> 
				<b>Abstract:</b> Lossy compression and asynchronous I/O are two of the most effective solutions for reducing storage overhead and enhancing I/O performance in large-scale high-performance computing (HPC) applications. However, current approaches have limitations that prevent them from fully leveraging lossy compression, and they may also result in task collisions, which restrict the overall performance of HPC applications. To address these issues, we propose an optimization approach for the task scheduling problem that encompasses computation, compression, and I/O. Our algorithm adaptively selects the optimal compression and I/O queue to minimize the performance degradation of the computation. We also introduce an intra-node I/O workload balancing mechanism that evenly distributes the workload across different processes. Additionally, we design a framework that incorporates fine-grained compression, a compressed data buffer, and a shared Huffman tree to fully benefit from our proposed task scheduling. Experimental results with up to 16 nodes and 64 GPUs from ORNL Summit, as well as real-world HPC applications, demonstrate that our solution reduces I/O overhead by up to 3.8× and 2.6× compared to non-compression and asynchronous I/O solutions, respectively.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    ** <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629575">Totoro: A Scalable Federated Learning Engine for the Edge</a></b>
	    <p>
				Cheng-Wei Ching (University of California Santa Cruz)<br>
				Xin Chen (Georgia Institute of Technology)<br>
				Taehwan Kim, Bo Ji (Virginia Tech)<br>
				Qingyang Wang (Louisiana State University)<br>
				Dilma Da Silva (Texas A&M University)<br>
				Liting Hu (University of California Santa Cruz and Virginia Tech)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, edge computing, FL
	    </p>
			<p> 
				<b>Abstract:</b> Federated Learning (FL) is an emerging distributed machine learning (ML) technique that enables in-situ model training and inference on decentralized edge devices. We propose Totoro, a novel scalable FL engine, that enables massive FL applications to run simultaneously on edge networks. The key insight is to explore a distributed hash table (DHT)-based peer-to-peer (P2P) model to re-architect the centralized FL system design into a fully decentralized one. In contrast to previous studies where many FL applications shared one centralized parameter server, Totoro assigns a dedicated parameter server to each individual application. Any edge node can act as any application's coordinator, aggregator, client selector, worker (participant device), or any combination of the above, thereby radically improving scalability and adaptivity. Totoro introduces three innovations to realize its design: a locality-aware P2P multi-ring structure, a publish/subscribe-based forest abstraction, and a bandit-based exploitation-exploration path planning model. Real-world experiments on 500 Amazon EC2 servers show that Totoro scales gracefully with the number of FL applications and N edge nodes, speeds up the total training time by 1.2 × -14.0×, achieves O (logN) hops for model dissemination and gradient aggregation with millions of nodes, and efficiently adapts to the practical edge networks and churns.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    ** <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629575">Totoro: A Scalable Federated Learning Engine for the Edge</a></b>
	    <p>
				Cheng-Wei Ching (University of California Santa Cruz)<br>
				Xin Chen (Georgia Institute of Technology)<br>
				Taehwan Kim, Bo Ji (Virginia Tech)<br>
				Qingyang Wang (Louisiana State University)<br>
				Dilma Da Silva (Texas A&M University)<br>
				Liting Hu (University of California Santa Cruz and Virginia Tech)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, edge computing, FL
	    </p>
			<p> 
				<b>Abstract:</b> Federated Learning (FL) is an emerging distributed machine learning (ML) technique that enables in-situ model training and inference on decentralized edge devices. We propose Totoro, a novel scalable FL engine, that enables massive FL applications to run simultaneously on edge networks. The key insight is to explore a distributed hash table (DHT)-based peer-to-peer (P2P) model to re-architect the centralized FL system design into a fully decentralized one. In contrast to previous studies where many FL applications shared one centralized parameter server, Totoro assigns a dedicated parameter server to each individual application. Any edge node can act as any application's coordinator, aggregator, client selector, worker (participant device), or any combination of the above, thereby radically improving scalability and adaptivity. Totoro introduces three innovations to realize its design: a locality-aware P2P multi-ring structure, a publish/subscribe-based forest abstraction, and a bandit-based exploitation-exploration path planning model. Real-world experiments on 500 Amazon EC2 servers show that Totoro scales gracefully with the number of FL applications and N edge nodes, speeds up the total training time by 1.2 × -14.0×, achieves O (logN) hops for model dissemination and gradient aggregation with millions of nodes, and efficiently adapts to the practical edge networks and churns.
			</p>
			<p></p>
		</td>
  </tr>



  <tr>
  	<td>
	    *** <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629578">Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications</a></b>
	    <p>
				Foteini Strati, Xianzhe Ma, Ana Klimovic (ETH Zurich)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, GPU sharing
	    </p>
			<p> 
				<b>Abstract:</b> GPUs are critical for maximizing the throughput-per-Watt of deep neural network (DNN) applications. However, DNN applications often underutilize GPUs, even when using large batch sizes and eliminating input data processing or communication stalls. DNN workloads consist of data-dependent operators, with different compute and memory requirements. While an operator may saturate GPU compute units or memory bandwidth, it often leaves other GPU resources idle. Despite the prevalence of GPU sharing techniques, current approaches are not sufficiently fine-grained or interference-aware to maximize GPU utilization while minimizing interference at the granularity of 10s of μs. We propose Orion, a system that transparently intercepts GPU kernel launches from multiple clients sharing a GPU. Orion schedules work on the GPU at the granularity of individual operators and minimizes interference by taking into account each operator's compute and memory requirements. We integrate Orion in PyTorch and demonstrate its benefits in various DNN workload collocation use cases. Orion significantly improves tail latency compared to state-of-the-art baselines for a high-priority inference job while collocating best-effort inference jobs to increase per-GPU request throughput by up to 7.3×, or while collocating DNN training, saving up to 1.49× in training costs compared to dedicated GPU allocation.
			</p>
			<p></p>
		</td>
  </tr>



  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629580">HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis</a></b>
	    <p>
				Shiwei Zhang (The University of Hong Kong)<br>
				Lansong Diao (Alibaba Group)<br>
				Chuan Wu (The University of Hong Kong)<br>
				Zongyan Cao, Siyu Wang, Wei Lin (Alibaba Group)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, SPMD parallelism, DNN training
	    </p>
			<p> 
				<b>Abstract:</b> Single-Program-Multiple-Data (SPMD) parallelism has recently been adopted to train large deep neural networks (DNNs). Few studies have explored its applicability on heterogeneous clusters, to fully exploit available resources for large model learning. This paper presents HAP, an automated system designed to expedite SPMD DNN training on heterogeneous clusters. HAP jointly optimizes the tensor sharding strategy, sharding ratios across heterogeneous devices and the communication methods for tensor exchanges for optimized distributed training with SPMD parallelism. We novelly formulate model partitioning as a program synthesis problem, in which we generate a distributed program from scratch on a distributed instruction set that semantically resembles the program designed for a single device, and systematically explore the solution space with an A-based search algorithm. We derive the optimal tensor sharding ratios by formulating it as a linear programming problem. Additionally, HAP explores tensor communication optimization in a heterogeneous cluster and integrates it as part of the program synthesis process, for automatically choosing optimal collective communication primitives and applying sufficient factor broadcasting technique. Extensive experiments on representative workloads demonstrate that HAP achieves up to 2.41x speed-up on heterogeneous clusters.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629581">TraceUpscaler: Upscaling Traces to Evaluate Systems at High Load</a></b>
	    <p>
				Sultan Mahmud Sajal, Timothy Zhu, Bhuvan Urgaonkar (The Pennsylvania State University)<br>
				Siddhartha Sen (Microsoft Research)
	    </p>
	    <p>
	    	<b>Labels:</b> trace, evaluation
	    </p>
			<p> 
				<b>Abstract:</b> Trace replay is a common approach for evaluating systems by rerunning historical traffic patterns, but it is not always possible to find suitable real-world traces at the desired level of system load. Experimenting with higher traffic loads requires upscaling a trace to artificially increase the load. Unfortunately, most prior research has adopted ad-hoc approaches for upscaling, and there has not been a systematic study of how the upscaling approach impacts the results. One common approach is to count the arrivals in a predefined time-interval and multiply these counts by a factor, but this requires generating new requests/jobs according to some model (e.g., a Poisson process), which may not be realistic. Another common approach is to divide all the timestamps in the trace by an upscaling factor to squeeze the requests into a shorter time period. However, this can distort temporal patterns within the input trace. This paper evaluates the pros and cons of existing trace upscaling techniques and introduces a new approach, TraceUpscaler, that avoids the drawbacks of existing methods. The key idea behind TraceUpscaler is to decouple the arrival timestamps from the request parameters/data and upscale just the arrival timestamps in a way that preserves temporal patterns within the input trace. Our work applies to open-loop traffic where requests have arrival timestamps that aren't dependent on previous request completions. We evaluate TraceUpscaler under multiple experimental settings using both real-world and synthetic traces. Through our study, we identify the trace characteristics that affect the quality of upscaling in existing approaches and show how TraceUpscaler avoids these pitfalls. We also present a case study demonstrating how inaccurate trace upscaling can lead to incorrect conclusions about a system's ability to handle high load.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629583">Blox: A Modular Toolkit for Deep Learning Schedulers</a></b>
	    <p>
				Saurabh Agarwal (University of Wisconsin-Madison)<br>
				Amar Phanishayee (Microsoft Research)<br>
				Shivaram Venkataraman (University of Wisconsin-Madison)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, DL scheduler
	    </p>
			<p> 
				<b>Abstract:</b> Deep Learning (DL) workloads have rapidly increased in popularity in enterprise clusters and several new cluster schedulers have been proposed in recent years to support these workloads. With rapidly evolving DL workloads, it is challenging to quickly prototype and compare scheduling policies across workloads. Further, as prior systems target different aspects of scheduling (resource allocation, placement, elasticity etc.), it is also challenging to combine these techniques and understand the overall benefits. To address these challenges we propose Blox, a modular toolkit which allows developers to compose individual components and realize diverse scheduling frameworks. We identify a set of core abstractions for DL scheduling, implement several existing schedulers using these abstractions, and verify the fidelity of these implementations by reproducing results from prior research. We also highlight how we can evaluate and compare existing schedulers in new settings: different workload traces, higher cluster load, change in DNN workloads and deployment characteristics. Finally, we showcase Blox's extensibility by composing policies from different schedulers, and implementing novel policies with minimal code changes. Blox is available at https://github.com/msr-fiddle/blox.
			</p>
			<p><a href="https://github.com/msr-fiddle/blox">open source</a></p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3629585">DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines
</a></b>
	    <p>
				Chenyu Jiang (The University of Hong Kong)<br>
				Zhen Jia (Amazon Web Services)<br>
				Shuai Zheng (Boson AI)<br>
				Yida Wang (Amazon Web Services)<br>
				Chuan Wu (The University of Hong Kong)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, DL scheduler
	    </p>
			<p> 
				<b>Abstract:</b> Multi-task model training has been adopted to enable a single deep neural network model (often a large language model) to handle multiple tasks (e.g., question answering and text summarization). Multi-task training commonly receives input sequences of highly different lengths due to the diverse contexts of different tasks. Padding (to the same sequence length) or packing (short examples into long sequences of the same length) is usually adopted to prepare input samples for model training, which is nonetheless not space or computation efficient. This paper proposes a dynamic micro-batching approach to tackle sequence length variation and enable efficient multi-task model training. We advocate pipelineparallel training of the large model with variable-length micro-batches, each of which potentially comprises a different number of samples. We optimize micro-batch construction using a dynamic programming-based approach, and handle micro-batch execution time variation through dynamic pipeline and communication scheduling, enabling highly efficient pipeline training. Extensive evaluation on the FLANv2 dataset demonstrates up to 4.39x higher training throughput when training T5, and 3.25x when training GPT, as compared with packing-based baselines. DynaPipe's source code is publicly available at https://github.com/awslabs/optimizing-multitask-training-through-dynamic-pipelines.
			</p>
			<p><a href="https://github.com/awslabs/optimizing-multitask-training-through-dynamic-pipelines">open source</a></p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3650074">GMorph: Accelerating Multi-DNN Inference via Model Fusion</a></b>
	    <p>
				Qizheng Yang, Tianyi Yang, Mingcan Xiang Lijun Zhang (University of Massachusetts Amherst)<br>
				Haoliang Wang (Adobe Research)<br>
				Marco Serafini, Hui Guan (University of Massachusetts Amherst)
	    </p>
	    <p>
	    	<b>Labels:</b> DNN training, multi-task learning, MTL, model fusion
	    </p>
			<p> 
				<b>Abstract:</b> AI-powered applications often involve multiple deep neural network (DNN)-based prediction tasks to support application-level functionalities. However, executing multi-DNNs can be challenging due to the high resource demands and computation costs that increase linearly with the number of DNNs. Multi-task learning (MTL) addresses this problem by designing a multi-task model that shares parameters across tasks based on a single backbone DNN. This paper explores an alternative approach called model fusion: rather than training a single multi-task model from scratch as MTL does, model fusion fuses multiple task-specific DNNs that are pre-trained separately and can have heterogeneous architectures into a single multi-task model. We materialize model fusion in a software framework called GMorph to accelerate multi-DNN inference while maintaining task accuracy. GMorph features three main technical contributions: graph mutations to fuse multi-DNNs into resource-efficient multi-task models, search-space sampling algorithms, and predictive filtering to reduce the high search costs. Our experiments show that GMorph can outperform MTL baselines and reduce the inference latency of multi-DNNs by 1.1-3× while meeting the target task accuracy.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3650083">ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling</a></b>
	    <p>
				Shaohuai Shi (Harbin Institute of Technology, Shenzhen)<br>
				Xinglin Pan (The Hong Kong University of Science and Technology (Guangzhou))<br>
				Qiang Wang (Harbin Institute of Technology, Shenzhen)<br>
				Chengjian Liu (Shenzhen Technology University)<br>
				Xiaozhe Ren, Zhongzhe Hu Yu Yang (Huawei Central Research Institute, Huawei Technologies)<br>
				Bo Li (The Hong Kong University of Science and Technology)<br>
				Xiaowen Chu (The Hong Kong University of Science and Technology (Guangzhou))
	    </p>
	    <p>
	    	<b>Labels:</b> DNN training, ML System
	    </p>
			<p> 
				<b>Abstract:</b> In recent years, large-scale models can be easily scaled to trillions of parameters with sparsely activated mixture-of-experts (MoE), which significantly improves the model quality while only requiring a sub-linear increase in computational costs. However, MoE layers require the input data to be dynamically routed to a particular GPU for computing during distributed training. The highly dynamic property of data routing and high communication costs in MoE make the training system low scaling efficiency on GPU clusters. In this work, we propose an extensible and efficient MoE training system, ScheMoE, which is equipped with several features. 1) ScheMoE provides a generic scheduling framework that allows the communication and computation tasks in training MoE models to be scheduled in an optimal way. 2) ScheMoE integrates our proposed novel all-to-all collective which better utilizes intra- and inter-connect bandwidths. 3) ScheMoE supports easy extensions of customized all-to-all collectives and data compression approaches while enjoying our scheduling algorithm. Extensive experiments are conducted on a 32-GPU cluster and the results show that ScheMoE outperforms existing state-of-the-art MoE systems, Tutel and Faster-MoE, by 9%-30%.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3650084">Erlang: Application-Aware Autoscaling for Cloud Microservices</a></b>
	    <p>
				Vighnesh Sachidananda (Google)<br>
				Anirudh Sivaraman (NYU)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, microservice, autoscaling
	    </p>
			<p> 
				<b>Abstract:</b> As cloud applications shift from monoliths to loosely coupled microservices, application developers must decide how many compute resources (e.g., number of replicated containers) to assign to each microservice within an application. This decision affects both (1) the dollar cost to the application developer and (2) the end-to-end latency perceived by the application user. Today, individual microservices are autoscaled independently by adding VMs whenever per-microservice CPU or memory utilization crosses a configurable threshold. However, an application user's end-to-end latency consists of time spent on multiple microservices and each microservice might need a different number of VMs to achieve an overall end-to-end latency.
				<br>
				We present Erlang, an autoscaler for microservice-based applications, which collectively allocates VMs to microservices with a global goal of minimizing dollar cost while keeping end-to-end application latency under a given target. Using 5 open-source applications, we compared Erlang to several utilization and machine learning based autoscalers. We evaluate Erlang across different compute settings on Google Kubernetes Engine (GKE) in which users manage compute resources, GKE standard, and a new mode of operation in which the cloud provider manages compute infrastructure, GKE Autopilot. Erlang meets a desired median or tail latency target on 53 of 63 workloads where it provides a cost reduction of 19.3%, on average, over the next cheapest autoscaler. Erlang is the most cost effective autoscaling policy for 48 of these 53 workloads. The cost savings from managing a cluster with Erlang result in Erlang paying for its training cost in a few days. On smaller applications, for which we can exhaustively search microservice configurations, we find that Erlang is optimal for 90% of cases and near optimal otherwise. Code for Erlang is available at https://github.com/vigsachi/erlang
			</p>
			<p><a href="https://github.com/vigsachi/erlang">open sources</a></p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3627703.3650088">ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs</a></b>
	    <p>
				Bing-Jyue Chen (UIUC)<br>
				Suppakit Waiwitlikhit (Stanford)<br>
				Ion Stoica (UC Berkeley)<br>
				Daniel Kang (UIUC)
	    </p>
	    <p>
	    	<b>Labels:</b> ML inference, work proof
	    </p>
			<p> 
				<b>Abstract:</b> Machine learning (ML) is increasingly used behind closed systems and APIs to make important decisions. For example, social media uses ML-based recommendation algorithms to decide what to show users, and millions of people pay to use ChatGPT for information every day. Because ML is deployed behind these closed systems, there are increasing calls for transparency, such as releasing model weights. However, these service providers have legitimate reasons not to release this information, including for privacy and trade secrets. To bridge this gap, recent work has proposed using zero-knowledge proofs (specifically a form called ZK-SNARKs) for certifying computation with private models but has only been applied to unrealistically small models.
				<br>
				In this work, we present the first framework, ZKML, to produce ZK-SNARKs for realistic ML models, including state-of-the-art vision models, a distilled GPT-2, and the ML model powering Twitter's recommendations. We accomplish this by designing an optimizing compiler from TensorFlow to circuits in the halo2 ZK-SNARK proving system. There are many equivalent ways to implement the same operations within ZK-SNARK circuits, and these design choices can affect performance by 24×. To efficiently compile ML models, ZKML contains two parts: gadgets (efficient constraints for low-level operations) and an optimizer to decide how to lay out the gadgets within a circuit. Combined, these optimizations enable proving on a wider range of models, faster proving, faster verification, and smaller proofs compared to prior work.
			</p>
			<p></p>
		</td>
  </tr>


</table>





<a name="SOSP'24"></a>
1. [SOSP'24](https://sigops.org/s/conferences/sosp/2024/accepted.html)

<table>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695963">Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving</a></b>
	    <p>
				Yinwei Dai, Rui Pan (Princeton University)<br>
				Anand Iyer (Georgia Tech)<br>
				Kai Li (Princeton University), Ravi Netravali (Princeton University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML system, DL inference
	    </p>
			<p> 
				<b>Abstract:</b> Machine learning (ML) inference platforms are tasked with balancing two competing goals: ensuring high throughput given many requests, and delivering low-latency responses to support interactive applications. Unfortunately, existing platform knobs (e.g., batch sizes) fail to ease this fundamental tension, and instead only enable users to harshly trade off one property for the other. This paper explores an alternate strategy to taming throughput-latency tradeoffs by changing the granularity at which inference is performed. We present Apparate, a system that automatically applies and manages early exits (EEs) in ML models, whereby certain inputs can exit with results at intermediate layers. To cope with the time-varying overhead and accuracy challenges that EEs bring, Apparate repurposes exits to provide continual feedback that powers several novel runtime monitoring and adaptation strategies. Apparate lowers median response latencies by 40.5--91.5% and 10.0--24.2% for diverse CV and NLP classification workloads, and median time-per-token latencies by 22.6--77.9% for generative scenarios, without affecting throughputs or violating tight accuracy constraints.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695969">Enabling Parallelism Hot Switching for Efficient Training of Large Language Models</a></b>
	    <p>
				Hao Ge, Fangcheng Fu, Haoyang Li, Xuanyu Wang, Sheng Lin, Yujie Wang, Xiaonan Nie, Hailin Zhang (Peking University)<br>
				Xupeng Miao (Carnegie Mellon University)<br>
				Bin Cui (Peking University)
	    </p>
	    <p>
	    	<b>Labels:</b> LLM training, parallelism hot switching
	    </p>
			<p> 
				<b>Abstract:</b> Training of large-scale deep learning models necessitates parallelizing the model and data across numerous devices, and the choice of parallelism strategy substantially depends on the training workloads such as memory consumption, computation cost, and communication cost. Current approaches generally assume uniform training workloads across samples in a given task. Thus, existing systems are designed to adopt a static parallelism strategy throughout one training process. Nevertheless, when training models with sequence inputs, this assumption fails due to the sequence length variation across samples. Consequently, training with a static parallelism strategy would result in sub-optimal performance.
				<br>
				In this paper, we first reveal the under-explored fact that the optimal parallelism strategy varies even for the sequences within a single mini-batch. Motivated by this, we present HotSPa, a novel system that adopts multiple parallelism strategies for efficient training with sequence inputs. To be specific, given a mini-batch of training sequences, HotSPa partitions them into multiple groups and applies different parallelism strategies to process each group individually. To enable the hot switching between strategies, HotSPa transfers model parameters and accumulated gradients among the devices on the fly. Significant solutions are proposed with the hope of seamless and rapid parallelism hot switching. Firstly, we design a graph compiler, which generates distributed computation graphs for different parallelism strategies simultaneously, and orchestrates them to share a single model storage backbone. Secondly, we develop a simple yet effective hot switch planner, which heuristically deduces communication plans to accelerate the transition of model partitioning given any pairs of strategies. Extensive experiments on large language model training demonstrate that HotSPa can be up to 2.99× faster than Megatron-LM and DeepSpeed that utilize static parallelism strategies. Source code is available: https://github.com/PKU-DAIR/Hetu.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    ** <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695978">Improving DNN Inference Throughput Using Practical, Per-Input Compute Adaptation</a></b>
	    <p>
				Anand Iyer (Georgia Tech)<br>
				Swapnil Gandhi (Stanford University)<br>
				Mingyu Guan (Georgia Tech)<br>
				Yinwei Dai, Rui Pan, Ravi Netravali (Princeton University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, DNN inference
	    </p>
			<p> 
				<b>Abstract:</b> Machine learning inference platforms continue to face high request rates and strict latency constraints. Existing solutions largely focus on compressing models to substantially lower compute costs (and time) with mild accuracy degradations. This paper explores an alternate (but complementary) technique that trades off accuracy and resource costs on a perinput granularity: early exit models, which selectively allow certain inputs to exit a model from an intermediate layer. Though intuitive, early exits face fundamental deployment challenges, largely owing to the effects that exiting inputs have on batch size (and resource utilization) throughout model execution. We present E3, the first system that makes early exit models practical for realistic inference deployments. Our key insight is to split and replicate blocks of layers in models in a manner that maintains a constant batch size throughout execution, all the while accounting for resource requirements and communication overheads. Evaluations with NLP and vision models show that E3 can deliver up to 1.74× improvement in goodput (for a fixed cost) or 1.78× reduction in cost (for a fixed goodput). Additionally, E3's goodput wins generalize to autoregressive LLMs (2.8--3.8×) and compressed models (1.67×).
			</p>
			<p></p>
		</td>
  </tr>



  <tr>
  	<td>
	    ** <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695948">LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism</a></b>
	    <p>
				Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, Xin Jin (Peking University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, LLM inference, LLM serving
	    </p>
			<p> 
				<b>Abstract:</b> The context window of large language models (LLMs) is rapidly increasing, leading to a huge variance in resource usage between different requests as well as between different phases of the same request. Restricted by static parallelism strategies, existing LLM serving systems cannot efficiently utilize the underlying resources to serve variable-length requests in different phases. To address this problem, we propose a new parallelism paradigm, elastic sequence parallelism (ESP), to elastically adapt to the variance across different requests and phases. Based on ESP, we design and build LoongServe, an LLM serving system that (1) improves computation efficiency by elastically adjusting the degree of parallelism in real-time, (2) improves communication efficiency by reducing key-value cache migration overhead and overlapping partial decoding communication with computation, and (3) improves GPU memory efficiency by reducing key-value cache fragmentation across instances. Our evaluation under diverse real-world datasets shows that LoongServe improves the throughput by up to 3.85× compared to chunked prefill and 5.81× compared to prefill-decoding disaggregation.
			</p>
			<p></p>
		</td>
  </tr>



  <tr>
  	<td>
	    *** <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695964">PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU</a></b>
	    <p>
				Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (Shanghai Jiao Tong University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, LLM inference, LLM serving
	    </p>
			<p> 
				<b>Abstract:</b> This paper introduces PowerInfer, a high-speed Large Language Model (LLM) inference engine on a personal computer (PC) equipped with a single consumer-grade GPU. The key principle underlying the design of PowerInfer is exploiting the high locality inherent in LLM inference, characterized by a power-law distribution in neuron activation. This distribution indicates that a small subset of neurons, termed hot neurons, are consistently activated across inputs, while the majority, cold neurons, vary based on specific inputs. PowerInfer exploits such an insight to design a GPU-CPU hybrid inference engine: hot-activated neurons are preloaded onto the GPU for fast access, while cold-activated neurons are computed on the CPU, thus significantly reducing GPU memory demands and CPU-GPU data transfers. PowerInfer further integrates adaptive predictors and neuron-aware sparse operators, optimizing the efficiency of neuron activation and computational sparsity. The evaluation shows that PowerInfer significantly outperforms llama.cpp by up to 11.69× while retaining model accuracy across various LLMs (including OPT-175B) on a single NVIDIA RTX 4090 GPU. For the OPT-30B model, PowerInfer achieves performance comparable to that of a high-end server-grade A100 GPU, reaching 82% of its token generation rate on a single consumer-grade RTX 4090 GPU.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    <b><a href="https://arxiv.org/abs/2405.14009">ReCycle: Pipeline Adaptation for the Resilient Distributed Training of Large DNNs</a></b>
	    <p>
				Swapnil Gandhi, Mark Zhao, Athinagoras Skiadopoulos, Christos Kozyrakis (Stanford University)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, pipeline adaption, DNN training, fault tolerant training
	    </p>
			<p> 
				<b>Abstract:</b> Training large Deep Neural Network (DNN) models requires thousands of GPUs over the course of several days or weeks. At this scale, failures are frequent and can have a big impact on training throughput. Utilizing spare GPU servers to mitigate performance loss becomes increasingly costly as model sizes grow. ReCycle is a system designed for efficient DNN training in the presence of failures, without relying on spare servers. It exploits the inherent functional redundancy in distributed training systems -- where servers across data-parallel groups store the same model parameters -- and pipeline schedule bubbles within each data-parallel group. When servers fails, ReCycle dynamically re-routes micro-batches to data-parallel peers, allowing for uninterrupted training despite multiple failures. However, this re-routing can create imbalances across pipeline stages, leading to reduced training throughput. To address this, ReCycle introduces two key optimizations that ensure re-routed micro-batches are processed within the original pipeline schedule's bubbles. First, it decouples the backward pass into two phases: one for computing gradients for the input and another for calculating gradients for the parameters. Second, it avoids synchronization across pipeline stages by staggering the optimizer step. Together, these optimizations enable adaptive pipeline schedules that minimize or even eliminate training throughput degradation during failures. We describe a prototype for ReCycle and show that it achieves high training throughput under multiple failures, outperforming recent proposals for fault-tolerant training such as Oobleck and Bamboo by up to 1.46× and 1.64×, respectively.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695970">Reducing Energy Bloat in Large Model Training</a></b>
	    <p>
				Jae-Won Chung (University of Michigan)<br>
				Yile Gu (University of Washington)<br>
				Insu Jang (University of Michigan)<br>
				Luoxi Meng (University of California, San Diego)<br>
				Nikhil Bansal, Mosharaf Chowdhury (University of Michigan)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, DNN training, energy saving
	    </p>
			<p> 
				<b>Abstract:</b> Training large AI models on numerous GPUs consumes a massive amount of energy, making power delivery one of the largest limiting factors in building and operating datacenters for AI workloads. However, we observe that not all energy consumed during training directly contributes to end-to-end throughput; a significant portion can be removed without slowing down training. We call this portion energy bloat.
				<br>
				In this work, we identify two independent sources of energy bloat in large model training and propose Perseus, a training system that mitigates both. To do this, Perseus obtains the time-energy tradeoff frontier of a large model training job using an efficient graph cut-based algorithm, and schedules computation energy consumption across time to reduce both types of energy bloat. Evaluation on large models, including GPT-3 and Bloom, shows that Perseus reduces the energy consumption of large model training by up to 30% without any throughput loss or hardware modification.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    ** <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695975">Tenplex: Dynamic Parallelism for Deep Learning using Parallelizable Tensor Collections</a></b>
	    <p>
				Marcel Wagenländer, Guo Li (Imperial College London)<br>
				Bo Zhao (Aalto University)<br>
				Luo Mai (University of Edinburgh)<br>
				Peter Pietzuch (Imperial College London)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, DNN training, energy saving
	    </p>
			<p> 
				<b>Abstract:</b> Deep learning (DL) jobs use multi-dimensional parallelism, i.e., combining data, model, and pipeline parallelism, to use large GPU clusters efficiently. Long-running jobs may experience changes to their GPU allocation: (i) resource elasticity during training adds or removes GPUs; (ii) hardware maintenance may require redeployment on different GPUs; and (iii) GPU failures force jobs to run with fewer devices. Current DL frameworks tie jobs to a set of GPUs and thus lack support for these scenarios. In particular, they cannot change the multi-dimensional parallelism of an already-running job in an efficient and model-independent way.
				<br>
				We describe Tenplex, a state management library for DL systems that enables jobs to change their parallelism dynamically after the GPU allocation is updated at runtime. Tenplex achieves this through a new abstraction, a parallelizable tensor collection (PTC), that externalizes the job state during training. After a GPU change, Tenplex uses the PTC to transform the job state: the PTC repartitions the dataset state under data parallelism and exposes it to GPU workers through a virtual file system; and the PTC obtains the model state as partitioned checkpoints and transforms them to reflect the new parallelization configuration. For efficiency, Tenplex executes PTC transformations in parallel with minimum data movement between GPU workers. Our experiments show that Tenplex enables DL jobs to support dynamic parallelization with low overhead.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695955">Scaling Deep Learning Computation over the Inter-Core Connected Intelligence Processor with T10</a></b>
	    <p>
				Marcel Wagenländer, Guo Li (Imperial College London)<br>
				Bo Zhao (Aalto University)<br>
				Luo Mai (University of Edinburgh)<br>
				Peter Pietzuch (Imperial College London)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, deep learning computing, inter-core communication
	    </p>
			<p> 
				<b>Abstract:</b> As AI chips incorporate numerous parallelized cores to scale deep learning (DL) computing, inter-core communication is enabled recently by employing high-bandwidth and low-latency interconnect links on the chip (e.g., Graphcore IPU). It allows each core to directly access the fast scratchpad memory in other cores, which enables new parallel computing paradigms. However, without proper support for the scalable inter-core connections in current DL compilers, it is hard for developers to exploit the benefits of this new architecture.
				<br>
				We present T10, the first DL compiler to exploit the inter-core communication bandwidth and distributed on-chip memory on AI chips. To formulate the computation and communication patterns of tensor operators in this new architecture, T10 introduces a distributed tensor abstraction rTensor. T10 maps a DNN model to execution plans with a generalized compute-shift pattern, by <b><i>partitioning DNN computation into sub-operators and mapping them to cores</i></b>, so that the cores can exchange data following predictable patterns. T10 makes globally optimized trade-offs between on-chip memory consumption and inter-core communication overhead, selects the best execution plan from a vast optimization space, and alleviates unnecessary inter-core communications. Our evaluation with a real inter-core connected AI chip, the Graphcore IPU, shows up to 3.3× performance improvement, and scalability support for larger models, compared to state-of-the-art DL compilers and vendor libraries.
			</p>
			<p></p>
		</td>
  </tr>



  <tr>
  	<td>
	    * <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695955">Tiered Memory Management: Access Latency is the Key!</a></b>
	    <p>
				Marcel Wagenländer, Guo Li (Imperial College London)<br>
				Bo Zhao (Aalto University)<br>
				Luo Mai (University of Edinburgh)<br>
				Peter Pietzuch (Imperial College London)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, deep learning computing, inter-core communication
	    </p>
			<p> 
				<b>Abstract:</b> The emergence of tiered memory architectures has led to a renewed interest in memory management. Recent works on tiered memory management innovate on mechanisms for access tracking, page migration, and dynamic page size determination; however, they all use the same page placement algorithm---packing the hottest pages in the default tier (one with the lowest hardware-specified memory access latency). This makes an implicit assumption that, despite serving the hottest pages, the access latency of the default tier is less than that of alternate tiers. This assumption is far from real: it is well-known in the computer architecture community that, in the realistic case of multiple in-flight requests, memory access latency can be significantly larger than the hardware-specified latency. We show that, even under moderate loads, the default tier access latency can inflate to be 2.5× larger than the latency of alternate tiers; and that, under this regime, performance of state-of-the-art memory tiering systems can be 2.3× worse than the optimal.
				<br>
				Colloid is a memory management mechanism that embodies the principle of balancing access latencies---page placement across tiers should be performed so as to balance their average (loaded) access latencies. To realize this principle, Colloid innovates on both per-tier memory access latency measurement mechanisms, and page placement algorithms that decide the set of pages to place in each tier. We integrate Colloid with three state-of-the-art memory tiering systems---HeMem, TPP and MEMTIS. Evaluation across a wide variety of workloads demonstrates that Colloid consistently enables the underlying system to achieve near-optimal performance.
			</p>
			<p></p>
		</td>
  </tr>


  <tr>
  	<td>
	    ** <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695961">Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor</a></b>
	    <p>
				Siran Liu (Peking University)<br>
				Chengxiang Qi (University of Chinese Academy of Sciences)<br>
				Ying Cao (Microsoft Research Asia)<br>
				Chao Yang (Peking University)<br>
				Weifang Hu, Xuanhua Shi (Huazhong University of Science and Technology)<br>
				Fan Yang, Mao Yang (Microsoft Research Asia)
	    </p>
	    <p>
	    	<b>Labels:</b> ML System, deep learning computing, inter-core communication
	    </p>
			<p> 
				<b>Abstract:</b> To speed up computation, deep neural networks (DNNs) usually rely on highly optimized tensor operators. Despite the effectiveness, tensor operators are often defined empirically with ad hoc semantics. This hinders the analysis and optimization across operator boundaries. FractalTensor is a programming framework that addresses this challenge. At the core, FractalTensor is a nested list-based abstract data type (ADT), where each element is a tensor with static shape or another FractalTensor (i.e., nested). DNNs are then de-fined by high-order array compute operators like map/reduce/scan and array access operators like window/stride on FractalTensor. This new way of DNN definition explicitly exposes nested data parallelism and fine-grained data access patterns, opening new opportunities for whole program analysis and optimization. To exploit these opportunities, from the FractalTensor-based code the compiler extracts a nested multi-dimensional dataflow graph called Extended Task Dependence Graph (ETDG), which provides a holistic view of data dependency across different granularity. The ETDG is then transformed into an efficient implementation through graph coarsening, data reordering, and access materialization. Evaluation on six representative DNNs like RNN and FlashAttention on NVIDIA A100 shows that Fractal-Tensor achieves speedup by up to 5.45x and 2.14x on average through a unified solution for diverse optimizations.
			</p>
			<p></p>
		</td>
  </tr>

  <tr>
  	<td>
	    **** <b><a href="https://dl.acm.org/doi/10.1145/3694715.3695947">Unifying serverless and microservice workloads with SigmaOS</a></b>
	    <p>
				Ariel Szekely, Adam Belay, Robert Morris, Frans Kaashoek (MIT)
	    </p>
	    <p>
	    	<b>Labels:</b> cloud computing, serverless computing, microservice, workload consolidation
	    </p>
			<p> 
				<b>Abstract:</b> Many cloud applications use both serverless functions, for bursts of stateless parallel computation, and container orchestration, for long-running microservices and tasks that need to interact. Ideally a single platform would offer the union of these systems' capabilities, but neither is sufficient to act as that single platform: serverless functions are lightweight but cannot act as servers with long-term state, while container orchestration offers general-purpose computation but instance start-up takes too long to support burst parallelism.
				<br>
				σOS is a new multi-tenant cloud operating system that combines the best of container orchestration and serverless in one platform with one API. σOS computations, called procs, can be long-running, stateful, and interact with each other, making them a good match for both serverless and microservice tasks. A key aspect of the σOS design is its cloud-centric API, which provides flexible management of computation, a novel abstraction for communication endpoints, σEPs---which allow procs of a tenant to communicate efficiently but prohibits procs from sending packets to other tenants---and a flexible naming system to name, for example, σEPs.
				<br>
				Quick proc start-up is important for serverless uses. A key enabling observation is that both serverless and microservice applications rely on cloud services for much of the work traditionally done by the local OS (e.g., access to durable storage and additional compute resources). σOS exploits this observation by providing only a small and generic local operating system image to each proc, which can be created much more quickly than a container orchestration instance since σOS need not install application-specific filesystem content or (due to σOS's σEPs) configure an isolated overlay network.
				<br>
				Microbenchmarks show that σOS can cold start a proc in 7.7 msec and can create 36,650 procs per second, distributing them over a 24-machine cluster. An evaluation of σOS with two microservice applications from DeathStarBench, a MapReduce application, and an image processing benchmark, shows that the σOS API supports both microservices and lambda-style computations, and provides better performance than corresponding versions on AWS Lambda and Kubernetes.
			</p>
			<p></p>
		</td>
  </tr>


</table>
