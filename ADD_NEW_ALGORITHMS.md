# Adding New Continual Learning Algorithms

New Continual Learning algorithms can be added into CLiMB by adding Algorithm-specific modules in `src/cl_algorithms`. 
Algorithm specific initializations are made in `src/train/train_upstream_continual_learning`. 
However, certain algorithms may also need modifications to the model classes and/or TaskTrainer class.

For instance, in `src/cl_algorithms/experience_replay.py`, we defined a `ExperienceReplayMemory` [class](src/cl_algorithms/experience_replay.py#L12).
An empty ReplayMemory is [instantiated](src/train/train_upstream_continual_learning.py#L161) in `src/train/train_upstream_continual_learning`.
When training on a new CL task, the replay memory is passed into the `TaskTrainer.train()` to do a [periodic replay step](src/train/visionlanguage_tasks/train_snli_ve.py#L214).
Finally, after learning each task, the replay memory is [updated](src/train/train_upstream_continual_learning.py#L286) with a memory buffer for the new task.

On the model side, algorithms like `freeze-bottom-k` and `freeze-encoder` can be implemented by adding algorithm-specific methods [directly](src/modeling/vilt.py#L123) to the model.
