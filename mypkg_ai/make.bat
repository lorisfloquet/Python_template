:: This file is for Windows only, use Makefile for Linux.
@echo off
setlocal

if "%1"=="" goto no_target

if "%1"=="init" goto init
if "%1"=="test" goto test
if "%1"=="lib" goto lib
if "%1"=="test-cov" goto test_cov
if "%1"=="run" goto run
if "%1"=="run-v" goto run_v
if "%1"=="clean" goto clean
if "%1"=="archive" goto archive
goto end

:init
pip install -r requirements.txt
goto end

:lib
pip install -e .
goto end

:test
pytest -m "not slow"
goto end

:test_cov
pytest -m "not slow" --cov=mypkg_ai --cov-report=term-missing
goto end

:run
python mypkg_ai/main.py
goto end

:run_v
python mypkg_ai/main.py -v
goto end

:clean
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (.coverage) do @if exist "%%f" del /q "%%f"
goto end

:archive
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
for /r . %%f in (.coverage) do @if exist "%%f" del /q "%%f"
powershell -Command "Compress-Archive -Path * -DestinationPath mypkg_ai.zip -CompressionLevel Optimal -Force"
powershell -Command "Remove-Item -Path mypkg_ai.zip\*.git\* -Recurse -Force"
powershell -Command "Remove-Item -Path mypkg_ai.zip\.vscode\* -Recurse -Force"
goto end

:no_target
echo No target specified.
goto end

:end
endlocal
