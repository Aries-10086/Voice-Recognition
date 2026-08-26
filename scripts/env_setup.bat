@echo off
REM ============================================================
REM CrossLingual Voice Clone - Environment Setup
REM ============================================================
set "PROJECT_DIR=E:\Work_Work\voiceclone"
set "MODEL_DIR=D:\CodingPackage\models"
set "CONDA_ENV=D:\CodingPackage\Anaconda3\envs\voice1"

echo Model cache: %MODEL_DIR%
echo Conda env: %CONDA_ENV%

set "HF_HOME=%MODEL_DIR%\huggingface"
set "TORCH_HOME=%MODEL_DIR%\torch"
set "NLTK_DATA=%MODEL_DIR%\nltk_data"
set "TRANSFORMERS_CACHE=%MODEL_DIR%\huggingface\hub"
set "CTRANSLATE2_MODELS=%MODEL_DIR%\ctranslate2"

call "%CONDA_ENV%\Scripts\activate.bat"
echo Ready. Run: python scripts/run_pipeline.py --input audio.wav --target-lang en
cmd /k
