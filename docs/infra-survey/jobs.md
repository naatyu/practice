# AI infrastructure hiring evidence

Snapshot collected 2026-11-19. The [survey](README.md) explains how to interpret it.

## Employer-board sample

20 public employer feeds returned 4,552 listings across all functions. I screened their titles, matched descriptions for relevance, removed cross-location and seniority title variants where identifiable, and retained up to nine postings per employer across role families. I removed several adjacent research, evaluation, support and commercial roles on review. The resulting table has **91 postings across 17 employers**. This is a targeted sample, with technically explicit postings preferred, and cannot estimate market-wide prevalence. Some similarly titled positions have distinct teams or descriptions. Locations and roles can change.

Tags are case-insensitive keyword matches on the employer's job description. They include responsibilities, preferred qualifications and general context. A tag is evidence that the text mentions a topic, and the linked description determines whether it is required. The tags are deliberately broad and will miss synonyms.

- `par` sharding and parallelism, including FSDP.
- `net` NCCL, RDMA, InfiniBand, NVLink, collectives, NVSHMEM or RoCE.
- `gpu` accelerator kernels or tools such as CUDA, Triton, CUTLASS, HIP and ROCm.
- `perf` profiling, benchmarking or performance analysis.
- `serve` inference runtime names, KV cache, batching or prefill.
- `reliab` reliability, recovery, checkpointing, on-call or observability.
- `fleet` cluster scheduling or orchestration tooling.
- `data` data loading, dataset pipelines, storage or I/O.
- `precision` mixed precision, BF16, FP8, FP4 or quantization.
- `compile` compilers, torch.compile, XLA, MLIR or CUDA Graphs.
- `rl` rollouts or reinforcement learning.
- `remat` activation checkpointing, recomputation or activation offloading. Generic checkpointing matches `reliab`.

| Employer | Family | Posting | Location as listed | Mentions |
| --- | --- | --- | --- | --- |
| Anthropic | training | [Research Engineer / Scientist, Pre-training](https://job-boards.greenhouse.io/anthropic/jobs/5468947008) | Remote-Friendly (Travel-Required) / San Francisco, CA / Seattle, WA / New York City, NY | net, reliab, fleet, rl |
| Anthropic | inference | [Staff+ Software Engineer, Inference Velocity](https://job-boards.greenhouse.io/anthropic/jobs/5257650008) | Remote-Friendly (Travel-Required) / San Francisco, CA / Seattle, WA / New York City, NY | net, gpu, reliab, fleet, compile |
| Anthropic | fleet | [Staff Infrastructure Engineer, Cluster Infrastructure](https://job-boards.greenhouse.io/anthropic/jobs/5211297008) | London, UK | net, reliab, fleet |
| Anthropic | data | [Pre-training Data Infrastructure Engineer](https://job-boards.greenhouse.io/anthropic/jobs/4973067008) | San Francisco, CA | net, reliab, data |
| Anthropic | training | [Research Engineer, Pretraining](https://job-boards.greenhouse.io/anthropic/jobs/5119713008) | London, UK | net, reliab, fleet, rl |
| Anthropic | inference | [Performance Engineer, Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008) | San Francisco, CA / New York City, NY / Seattle, WA | net, perf, reliab, precision |
| Anthropic | inference | [Staff + Senior Software Engineer, Inference](https://job-boards.greenhouse.io/anthropic/jobs/5385998008) | Ontario, CAN | net, reliab, fleet |
| Anthropic | training | [Research Engineer, Pretraining Scaling](https://job-boards.greenhouse.io/anthropic/jobs/4938432008) | San Francisco, CA | net, reliab |
| Anyscale | inference | [Distributed LLM Inference Engineer](https://jobs.ashbyhq.com/anyscale/1cf38233-8aa0-47f8-9d85-65ce27bc3047) | San Francisco | gpu, serve, compile |
| Baseten | training | [Post-Training Research Engineer](https://jobs.ashbyhq.com/baseten/68328a94-f785-463b-a3d7-b0dfc11839b5) | San Francisco | par, net, gpu, perf, serve, fleet, data, rl |
| Baseten | inference | [Software Engineer - Voice AI (Inference Runtime)](https://jobs.ashbyhq.com/baseten/6e396eb7-acb3-436a-89ec-05e755c477f2) | San Francisco | perf, serve, fleet, data, rl |
| Baseten | kernels | [Software Engineer - GPU Kernels](https://jobs.ashbyhq.com/baseten/ddb5bc98-6116-49a2-802e-1c05398663f1) | San Francisco | gpu, perf, precision |
| Baseten | fleet | [Software Engineer - GPU Networking & Distributed Systems](https://jobs.ashbyhq.com/baseten/1f7d7fda-5540-4205-890b-cdbf774f0814) | San Francisco | par, net, perf, serve, reliab, fleet, data |
| Baseten | training | [Software Engineer - Training Infrastructure](https://jobs.ashbyhq.com/baseten/6f947409-2bc0-491a-81b3-507cdc95df85) | San Francisco | par, net, reliab, fleet, data, rl |
| Baseten | inference | [Software Engineer - Inference Performance](https://jobs.ashbyhq.com/baseten/7cb19a05-8e5b-44cf-b3e7-19949e2eff04) | San Francisco | gpu, perf, serve, precision |
| Baseten | inference | [Software Engineer - Inference Platform](https://jobs.ashbyhq.com/baseten/14c4a663-0b1f-4c11-93ff-1359741ee456) | San Francisco | perf, serve, reliab, fleet |
| Baseten | inference | [Software Engineer - Dedicated Inference](https://jobs.ashbyhq.com/baseten/fc6e5f2e-eb2d-4a6c-8a51-8422e8662bde) | San Francisco | reliab, fleet |
| Cohere | training | [Member of Technical Staff, Training Performance Engineer](https://jobs.ashbyhq.com/cohere/d42f5fd4-1ffc-45b9-957c-f09862db6af6) | London | net, gpu, perf, compile |
| Cohere | inference | [Senior/Staff Software Engineer (Back-End), Inference](https://jobs.ashbyhq.com/cohere/14aef6d9-763e-4f56-a498-206fb31dd0c6) | Canada | net, reliab, rl |
| Cohere | fleet | [Software Engineer, GPU Infrastructure](https://jobs.ashbyhq.com/cohere/735f26a3-9456-49e2-8f84-23804534cfe4) | Seattle | net, reliab, fleet |
| Cohere | training | [Member of Technical Staff, Training Infra Engineer](https://jobs.ashbyhq.com/cohere/a13207e7-dc82-473f-8ca4-e832452fe8c3) | Paris | net, fleet, compile |
| Cohere | inference | [Staff Software Engineer, Inference Infrastructure](https://jobs.ashbyhq.com/cohere/41f23dad-9da2-451a-bd1e-a1800437cb64) | San Francisco | net, fleet, data |
| Cohere | training | [Member of Technical Staff - Post-Training (Reinforcement Learning & Generalization)](https://jobs.ashbyhq.com/cohere/8a73f59d-41b7-42ed-8cc8-99afa0a2ba5f) | Europe | par, net, perf, rl |
| Cohere | training | [Member of Technical Staff, Post-Training](https://jobs.ashbyhq.com/cohere/554a9380-ab50-4338-88a9-c6b8ab19d92e) | London | net, fleet, compile |
| Cohere | training | [Member of Technical Staff - Post-training (Data Infrastructure)](https://jobs.ashbyhq.com/cohere/eef82766-7e31-430f-a102-bc33a5022597) | United States | net, data |
| Cohere | training | [Member of Technical Staff, Pre-Training Data](https://jobs.ashbyhq.com/cohere/859e2e47-02fb-4afe-bb8a-e83bf4d8c265) | Toronto | net, data |
| CoreWeave | training | [Senior Software Engineer (Post-Training)](https://coreweave.com/careers/job) | Bellevue, WA | net, reliab, fleet, rl |
| CoreWeave | inference | [Software Engineer, Inference](https://coreweave.com/careers/job) | Sunnyvale, CA / Bellevue, WA | net, gpu, perf, serve, reliab, fleet, precision |
| CoreWeave | kernels | [Senior Software Engineer - GPU Kernel Authoring & Optimization](https://coreweave.com/careers/job) | Sunnyvale, CA / Bellevue, WA | net, gpu, perf, serve, reliab, fleet, precision, compile |
| CoreWeave | inference | [Senior Deployment Engineer - LLM Inference & Optimization](https://coreweave.com/careers/job) | Bellevue, WA/ San Francisco, CA/  Sunnyvale, CA | net, gpu, perf, serve, precision |
| CoreWeave | inference | [Senior Software Engineer I, Inference](https://coreweave.com/careers/job) | Sunnyvale, CA / Bellevue, WA | net, perf, serve, precision |
| Crusoe | inference | [Staff Applied AI Inference Engineer](https://jobs.ashbyhq.com/Crusoe/0aae94c4-9fe6-4275-be3c-6c397345f7f4) | San Francisco, CA - US | gpu, perf, serve, fleet |
| Cursor | training | [Sr. Software Engineer, Pretraining Data](https://jobs.ashbyhq.com/cursor/db5d0c48-7c24-474d-a662-c960c3770d44) | Palo Alto, CA | net, reliab, fleet, data |
| Databricks | training | [Staff Software Engineer - AI Research Infrastructure](https://databricks.com/company/careers/open-positions/job) | New York City, New York; San Francisco, California | net, reliab, fleet, data |
| Databricks | inference | [Staff Software Engineer - GenAI inference](https://databricks.com/company/careers/open-positions/job) | San Francisco, California | par, net, gpu, perf, reliab, fleet, precision |
| Databricks | fleet | [Senior Software Engineer, Compute Infrastructure](https://databricks.com/company/careers/open-positions/job) | Mountain View, California | net, fleet |
| Databricks | inference | [Staff Software Engineer- Foundation Model Inference](https://databricks.com/company/careers/open-positions/job) | San Francisco, California | net, serve, reliab, fleet |
| Lambda | fleet | [Senior HPC Engineer - Fleet Engineering](https://jobs.ashbyhq.com/Lambda/d4e6f207-73c3-43d1-aa4b-d840ad2254d4) | San Francisco Office (Fremont St) | net, perf, reliab, fleet |
| Lambda | fleet | [Senior HPC Platform Hardware Engineer](https://jobs.ashbyhq.com/Lambda/3fede5ab-bd58-4539-b846-f613c8bc81ff) | San Jose Office (Zanker) | net, perf, reliab, data |
| Lambda | fleet | [Staff HPC Systems Architect](https://jobs.ashbyhq.com/Lambda/614dc105-61e2-4f78-827f-05f145211125) | San Jose Office (First St) | net, perf, fleet |
| Lambda | fleet | [Staff HPC Network Architect](https://jobs.ashbyhq.com/Lambda/efe606d6-eaa2-4e0a-a204-38002ae68818) | San Jose Office (First St) | net, perf, reliab, data |
| Modal | training | [Member of Technical Staff - Research, Post-Training](https://jobs.ashbyhq.com/modal/7db771eb-e379-47fe-974f-765a8780dc10) | New York | reliab, data, rl |
| Modal | inference | [Member of Technical Staff - Inference Runtime](https://jobs.ashbyhq.com/modal/6195f552-ca6f-4540-b7f6-e7e6054e47ad) | San Francisco | net, gpu, perf, reliab, data |
| Modal | inference | [Member of Technical Staff - Research, Inference](https://jobs.ashbyhq.com/modal/73c97bbc-8e27-4c5d-b38b-90b3afdb0d93) | New York | serve, reliab, data, precision |
| Nebius | inference | [Senior Machine Learning Engineer, LLM Inference Optimization](https://careers.nebius.com/) | London, United Kingdom | par, net, gpu, perf, serve, reliab, fleet, data, precision, rl |
| Nebius | kernels | [Lead Software Systems Engineer - GPU  Performance](https://careers.nebius.com/) | Remote - United States | net, fleet, data |
| Nebius | fleet | [Senior AI Infrastructure Systems Engineer (R&D / GPU / AI)](https://careers.nebius.com/) | United States | net, perf, fleet, data |
| Nebius | inference | [Senior Applied Scientist, Efficient LLM Inference & Model Optimization](https://careers.nebius.com/) | London, United Kingdom | par, net, gpu, perf, serve, fleet, data, precision |
| Nebius | fleet | [Senior HPC Cluster Engineer](https://careers.nebius.com/) | Madrid, Spain; Remote - Europe; United Kingdom | net, fleet, data |
| Nebius | inference | [Senior Site Reliability Engineer — Token Factory (Inference Platform)](https://careers.nebius.com/) | Amsterdam, Netherlands; Berlin, Germany; London, United Kingdom; Prague, Czech Republic; Remote - Europe | net, gpu, serve, reliab, fleet, data |
| Nebius | fleet | [Senior HPC Engineer, GPU Compute](https://careers.nebius.com/) | Amsterdam, Netherlands; Berlin, Germany; London, United Kingdom | net, fleet, data |
| Nebius | fleet | [Senior Systems HPC Engineer](https://careers.nebius.com/) | Amsterdam, Netherlands; Remote - Europe | net, fleet, data |
| Nebius | fleet | [GPU Cluster Architect](https://careers.nebius.com/) | India; Singapore | net, reliab, fleet, data |
| OpenAI | training | [Training Performance Engineer](https://jobs.ashbyhq.com/openai/6eb386ac-9056-4795-aa79-a27e105faf5c) | San Francisco | par, net, gpu, perf, reliab, fleet, data, compile |
| OpenAI | inference | [Software Engineer, Developer Productivity - Inference Runtime](https://jobs.ashbyhq.com/openai/9d48e2e6-41a9-4a90-8b3b-6cc960e95c2f) | San Francisco | net, reliab, rl |
| OpenAI | kernels | [Software Engineer, Kernel Performance & AI Tooling](https://jobs.ashbyhq.com/openai/e9627fa6-ac76-4899-9a93-9251419e61a0) | San Francisco | perf, reliab, compile, rl |
| OpenAI | fleet | [Software Engineer, Compute Infrastructure](https://jobs.ashbyhq.com/openai/ca300a6d-a2a7-4580-aad7-323fbdfee7b1) | San Francisco | net, perf, reliab, fleet, data |
| OpenAI | data | [Technical Lead - Training Runtime, Data(set) Movement](https://jobs.ashbyhq.com/openai/4cbbf0b1-7c0b-4c42-bf38-48377f847e91) | San Francisco | net, perf, reliab, data, rl |
| OpenAI | kernels | [Performance Modeling Engineer](https://jobs.ashbyhq.com/openai/19fc3e36-3bf3-4a7c-b65f-498d89220436) | San Francisco | perf, data |
| OpenAI | fleet | [Software Engineer, GPU Infrastructure - HPC](https://jobs.ashbyhq.com/openai/f58cb1eb-9642-4a4d-a14d-d7a57d583a11) | San Francisco | net, reliab |
| Prime Intellect | training | [Research Engineer - Distributed Training](https://jobs.ashbyhq.com/PrimeIntellect/8bd52610-175c-42a7-a7cd-b29c45f9d305) | San Francisco | par, gpu, perf, serve, compile, rl |
| Prime Intellect | inference | [Research Engineer - Inference](https://jobs.ashbyhq.com/PrimeIntellect/3103e1a4-2515-4ae0-a2f0-784e25d30cfe) | San Francisco | par, gpu, perf, serve, precision, rl |
| Prime Intellect | fleet | [Member of Technical Staff - GPU Infrastructure](https://jobs.ashbyhq.com/PrimeIntellect/297d925e-5a42-40bd-b02f-5c928d226f18) | San Francisco | par, net, gpu, perf, reliab, fleet, data |
| Prime Intellect | training | [Research Engineer - RL Infrastructure](https://jobs.ashbyhq.com/PrimeIntellect/07f6f7bc-5f5b-4565-8897-9876763198dd) | San Francisco | par, gpu, perf, serve, compile, rl |
| Prime Intellect | training | [Member of Technical Staff - Training Platform](https://jobs.ashbyhq.com/PrimeIntellect/8706578d-5a01-4270-9d43-ed9cd998a982) | San Francisco | par, net, serve, reliab, fleet, data |
| Prime Intellect | training | [Research Engineer - Pre-training](https://jobs.ashbyhq.com/PrimeIntellect/9dae2470-0e2c-4d42-8e7f-20b7f534440e) | San Francisco | perf, reliab |
| Prime Intellect | training | [Research Engineer - Pre-training Data](https://jobs.ashbyhq.com/PrimeIntellect/eecf9af5-46fb-4b62-b8c8-921098fd6514) | San Francisco | net, data |
| San Francisco Compute | fleet | [Principal/Staff HPC Network Engineer](https://jobs.ashbyhq.com/sfcompute/1b83badf-94ec-4563-88e9-bdae81e70626) | San Francisco, CA | net, reliab, fleet |
| San Francisco Compute | fleet | [HPC/ GPU Cluster Architect](https://jobs.ashbyhq.com/sfcompute/4b20fd99-8536-4a62-ad74-b7ba98f126be) | Remote | net, perf, reliab, fleet |
| San Francisco Compute | fleet | [HPC Data Center Ops Engineer](https://jobs.ashbyhq.com/sfcompute/2134c7b5-0190-41ea-be7c-3d827d52b4f1) | San Francisco, CA | net |
| Thinking Machines Lab | training | [Research Engineer, Infrastructure, Training Systems](https://jobs.ashbyhq.com/ThinkingMachines/9dabecde-1bdb-47c6-b943-ff7b0526657c) | San Francisco | par, net, perf, reliab, fleet, data, precision, compile |
| Thinking Machines Lab | inference | [Research Engineer, Infrastructure, Inference](https://jobs.ashbyhq.com/ThinkingMachines/4087e0f6-4295-419a-ba02-08e95a74ceea) | San Francisco | net, gpu, serve, reliab, fleet, compile |
| Thinking Machines Lab | kernels | [Research Engineer, Infrastructure, Kernels](https://jobs.ashbyhq.com/ThinkingMachines/2d7a9e99-836a-475b-9ae3-b6b1df8baa6d) | San Francisco | par, net, gpu, perf, precision, compile |
| Thinking Machines Lab | training | [Software Engineer, Training Infrastructure](https://jobs.ashbyhq.com/ThinkingMachines/5b2ee0dc-0fe5-4c5d-806d-e171187e460e) | San Francisco | net, perf, serve, reliab, fleet, rl |
| Thinking Machines Lab | inference | [Software Engineer, Inference](https://jobs.ashbyhq.com/ThinkingMachines/d7513f3d-6a49-485d-b3c7-ecb629ad48db) | San Francisco | net, reliab, precision, rl |
| Thinking Machines Lab | training | [Site Reliability Engineer, Post Training](https://jobs.ashbyhq.com/ThinkingMachines/a9469410-04c7-4e6a-b8b4-64c15933a2bf) | San Francisco | net, reliab, fleet, data, rl |
| Thinking Machines Lab | training | [Research Software Engineer, Post Training](https://jobs.ashbyhq.com/ThinkingMachines/4d51cb65-5e5b-42d2-8c28-7073b8574e4b) | San Francisco | net, data |
| Thinking Machines Lab | training | [Research, Pre-Training Data](https://jobs.ashbyhq.com/ThinkingMachines/27ec4240-9f70-4ebe-abc0-a7336358a9a4) | San Francisco | net |
| Together AI | training | [Research Engineer, Large-Scale Training](https://job-boards.greenhouse.io/togetherai/jobs/5199554007) | San Francisco | par, net, gpu, perf, reliab, precision, rl |
| Together AI | inference | [Senior Backend Engineer, Inference Platform](https://job-boards.greenhouse.io/togetherai/jobs/4835763007) | San Francisco | net, gpu, perf, serve, reliab, fleet, data, rl |
| Together AI | fleet | [AI Infrastructure System Engineer](https://job-boards.greenhouse.io/togetherai/jobs/5180155007) | Bangalore, India | net, gpu, reliab, fleet, data, rl |
| Together AI | inference | [Machine Learning Engineer - Inference](https://job-boards.greenhouse.io/togetherai/jobs/4385540007) | San Francisco | net, gpu, serve, reliab, data, compile, rl |
| Together AI | fleet | [AI Infrastructure Systems Engineer](https://job-boards.greenhouse.io/togetherai/jobs/5138540007) | San Francisco | net, gpu, reliab, fleet, data, rl |
| Together AI | inference | [(Senior/Junior AI Inference Reliability Engineer, Compute Infra](https://job-boards.greenhouse.io/togetherai/jobs/5268939007) | San Francisco | net, gpu, serve, reliab, fleet, rl |
| Together AI | fleet | [Staff Engineer, Distributed Storage and HPC & AI Infrastructure](https://job-boards.greenhouse.io/togetherai/jobs/5226095007) | Amsterdam | net, perf, reliab, fleet, data, rl |
| Together AI | inference | [(Staff / Senior Staff) AI Inference Reliability Engineer, Compute Infra](https://job-boards.greenhouse.io/togetherai/jobs/5266079007) | San Francisco | net, gpu, serve, reliab, fleet, rl |
| xAI | training | [Model Engineer, RL Training Framework](https://job-boards.greenhouse.io/xai/jobs/5186992007) | Palo Alto, CA | net, perf, reliab, rl |
| xAI | inference | [Software Engineer, Training/Inference (C++)](https://job-boards.greenhouse.io/xai/jobs/4533894007) | Palo Alto, CA; Seattle, WA | net, gpu, perf, serve, reliab, fleet, precision, rl |
| xAI | kernels | [Software Engineer, Kernels/CUDA (C++)](https://job-boards.greenhouse.io/xai/jobs/5052040007) | Palo Alto, CA; Seattle, WA | net, gpu, perf, fleet |
| xAI | training | [Model Engineer, Post-Training & RL](https://job-boards.greenhouse.io/xai/jobs/5114737007) | Palo Alto, CA | net, rl |
| xAI | inference | [Sr. Model Engineer, RL Inference](https://job-boards.greenhouse.io/xai/jobs/5180223007) | Palo Alto, CA | net, perf, serve, precision |

## Direct career-page checks

These employers supplement the board sample. I excluded these rows from the automated mention counts. NVIDIA graphics and general-purpose GPU roles outside ML were excluded.

| Employer | Family | Posting or dated listing |
| --- | --- | --- |
| Microsoft | Training | [Principal Software Engineer, AI Training Systems (CoreAI)](https://careers.microsoft.com/v2/global/en/locations/seattle-area.html), shown November 17 |
| NVIDIA | Inference | [Senior Deep Learning Software Engineer, Inference](https://jobs.nvidia.com/careers/job/893391034202) |
| NVIDIA | Kernels | [Senior Software Engineer, CUDA Deep Learning Systems](https://jobs.nvidia.com/careers/job/893395142941) |
| NVIDIA | Inference | [AI Computing Software Development Engineer, TensorRT-LLM](https://jobs.nvidia.com/careers/job/893396942415) |
| Google | Compilers | [Software Engineer III, Machine Learning Compiler, TPU Scaling](https://www.google.com/about/careers/applications/jobs/results/107943431245832902-software-engineer-iii-machine-learning-compiler-tpu-scaling) |
| Google | Compilers | [Staff ML Compiler Engineer, TPU Performance Optimizations](https://www.google.com/about/careers/applications/jobs/results/83143356728648390-staff-ml-compiler-engineer-tpu-performance-optimizations) |
| AMD | Inference | [Senior GPU Inference Performance Engineer](https://careers.amd.com/careers-home/jobs/87853) |
| AMD | Kernels and compilers | [Triton Compiler and Kernel Software Engineer](https://careers.amd.com/careers-home/jobs/92626) |
| Apple | Training | [Apple Foundation Models Senior ML/RL Training Infrastructure Engineer](https://jobs.apple.com/en-us/details/200634821-4170/aiml-apple-foundation-model-senior-ml-rl-training-infrastructure-engineer) |

## How I read selected postings

| Posting | Responsibility | What the description treats as experience or qualifications |
| --- | --- | --- |
| [OpenAI Training Performance](https://openai.com/careers/training-performance-engineer-san-francisco/) | Improve distributed training throughput across kernels, collective communication, I/O and sharding | Performance debugging across the training stack is central to the role. |
| [OpenAI Training Runtime Data Movement](https://openai.com/careers/technical-lead-training-runtime-data%28set%29-movement-san-francisco/) | Own data access interfaces, storage contracts, versioning, reproducibility and performance | Prior ownership of production datasets, storage, loading or distributed training infrastructure is a fit signal. |
| [Together AI Large-Scale Training](https://job-boards.greenhouse.io/togetherai/jobs/5199554007) | Design and improve large-scale training infrastructure | GPU architecture, mixed precision and distributed-training fundamentals appear in the core criteria. CUDA/Triton kernel authoring and collective implementation are listed as additional experience. |
| [Microsoft AI Training Systems](https://careers.microsoft.com/v2/global/en/locations/seattle-area.html) | Improve training across pretraining, fine-tuning and RL with sharding, recomputation, overlap, recovery and reproducible benchmarks | The formal minimum qualifications are broader than the techniques named under responsibilities. |
| [Anthropic Inference Velocity](https://job-boards.greenhouse.io/anthropic/jobs/5257650008) | Build development workflows and accelerator validation across CUDA, TPU and Trainium | The posting illustrates a systems productivity role adjacent to serving runtime optimization. |
| [Google TPU Performance Optimizations](https://www.google.com/about/careers/applications/jobs/results/83143356728648390-staff-ml-compiler-engineer-tpu-performance-optimizations) | Tune ML compiler performance for TPUs | Compiler and accelerator specialization is a distinct route within ML infrastructure. |
