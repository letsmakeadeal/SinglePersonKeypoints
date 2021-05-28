Test:
- Install necessary requirements
- Weights for model you can find in:
  - (https://wandb.ai/letsmakeadeal/Single%20person%20keypoints/runs/3c3sbe34/files?workspace=user-letsmakeadeal) for (128, 128) resolution.
  - (https://wandb.ai/letsmakeadeal/Single%20person%20keypoints/runs/1h5use6a/files?workspace=user-letsmakeadeal) for (256, 256) resolution.  
- To test network on your's own pictures, you should create directory with this pictures, and run python ./test.py path_to_dir checkpoint_path --train_dir_order 0.
- If you want to run (128, 128) network you should pass "--big_resolution 0" to ./test.py
- 

