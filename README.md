# Halide Auto-scheduler Error Reproduction

1. Build the docker image: 
   ```
   docker build -t halide_autoscheduler_error .
   ```

   The following `docker build --build-arg` values can be overridden:
   - `PYTHON_VERSION=3.7.6`
   - `HALIDE_REPO=https://github.com/halide/Halide.git`
   - `HALIDE_COMMIT=95f154af71f1895119b2c57270ab8eaccb9cbef0`
   
2. Run `autoscheduler_error.py` in the docker container:
   ```
   docker run -ti --mount type=bind,source=$PWD,target=/work halide_autoscheduler_error python autoscheduler_error.py
   ```