set REGION=us-east1
set BUCKET=gs://steven-ml-bucket/steeldefect
set JOB_DIR=%BUCKET%/output
set DATA_DIR=%BUCKET%/data
set MODEL_DIR=%BUCKET%/models
set MODEL_NAME=UNET_vgg19_800_128_4_original
set PREPROCESSED_DIR=%DATA_DIR%/preprocessed

REM using command to generate whl file:
REM python setup.py bdist_wheel

gcloud ai-platform jobs submit training steeldefect_25 ^
    --runtime-version 1.14 ^
    --job-dir %JOB_DIR%  ^
    --package-path trainer/ ^
    --module-name trainer.task ^
    --region %REGION% ^
	--python-version 3.5 ^
    --scale-tier BASIC_GPU ^
	--packages ../package/dist/steeldefect-0.1-py3-none-any.whl ^
    -- ^
    --model-dir %MODEL_DIR% ^
    --model-name %MODEL_NAME% ^
	--data-dir %DATA_DIR% ^
	--preprocessed-dir %PREPROCESSED_DIR% ^
    --continue-training False ^
	--batch-size 12 ^
	--num-epochs 10 ^
	--fast-verification False ^
	--show-progress False

