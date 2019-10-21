set JOB_DIR=.\output
set DATA_DIR=C:\Work\gitsrc\Kaggle\severstal-steel-defect-detection
set MODEL_DIR=C:\Work\gitsrc\Kaggle\severstal-steel-defect-detection\models
set MODEL_NAME=UNET_vgg19_800_128_4_10152258
set PREPROCESSED_DIR=C:\Work\gitsrc\ml\Kaggle\SteelDefects\data

gcloud ai-platform local train ^
    --job-dir %JOB_DIR%  ^
    --package-path trainer ^
    --module-name trainer.task ^
    -- ^
    --model-dir %MODEL_DIR% ^
    --model-name %MODEL_NAME% ^
	--data-dir %DATA_DIR% ^
	--preprocessed-dir %PREPROCESSED_DIR% ^
    --continue-training True ^
	--batch-size 12 ^
	--num-epochs 10 ^
	--fast-verification True
	

