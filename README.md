# Halide Auto-scheduler Error Reproduction

Reproduction of the following error when using the Adams2019 auto-scheduler:
```
terminate called after throwing an instance of 'Halide::Error'
  what():  Internal Error at /Halide/apps/autoscheduler/FunctionDAG.cpp:349 triggered by user code at : Condition failed: num_live == 0: Destroying a Layout without returning all the BoundContents. 1 are still live
```

1. Build the docker image: 
   ```
   docker build -t halide_autoscheduler_error .
   ```

   The following `docker build --build-arg` values can be overridden:
   - `HALIDE_REPO=https://github.com/halide/Halide.git`
   - `HALIDE_COMMIT=95f154af71f1895119b2c57270ab8eaccb9cbef0`

2. Run `autoscheduler_error.py` in the docker container:
   ```
   docker run -ti --mount type=bind,source=$PWD,target=/work halide_autoscheduler_error python autoscheduler_error.py
   ```