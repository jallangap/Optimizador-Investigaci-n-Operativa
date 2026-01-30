@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ===========================================
echo   INICIANDO PROYECTO - INVESTIGACION OPERATIVA
echo ===========================================

REM Flag opcional: play.bat --force
set FORCE=0
if /I "%~1"=="--force" set FORCE=1

REM Detectar launcher de Python
where py >nul 2>nul
if %errorlevel%==0 (
    set PY_LAUNCH=py -3
) else (
    set PY_LAUNCH=python
)

REM 1) Crear venv si no existe
if not exist ".\venv\Scripts\activate.bat" (
    echo [INFO] No existe venv. Creando entorno virtual...
    %PY_LAUNCH% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

REM 2) Activar venv
call ".\venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo activar venv.
    pause
    exit /b 1
)

REM 3) Calcular hash del requirements.txt (para evitar reinstalar siempre)
set REQ_HASH=
for /f "usebackq tokens=*" %%H in (`certutil -hashfile "requirements.txt" SHA256 ^| findstr /R /C:"^[0-9A-F][0-9A-F]*$"`) do (
    set REQ_HASH=%%H
)

set HASH_FILE=.\venv\.requirements.sha256
set OLD_HASH=

if exist "%HASH_FILE%" (
    set /p OLD_HASH=<"%HASH_FILE%"
)

REM 4) Chequeo rápido de dependencias (si faltan, instalamos)
python -c "import PyQt5, matplotlib; from google import genai" >nul 2>nul
set IMPORT_OK=%errorlevel%

set NEED_INSTALL=0
if %FORCE%==1 set NEED_INSTALL=1
if not exist "%HASH_FILE%" set NEED_INSTALL=1
if /I not "!OLD_HASH!"=="!REQ_HASH!" set NEED_INSTALL=1
if not "!IMPORT_OK!"=="0" set NEED_INSTALL=1

if "!NEED_INSTALL!"=="1" (
    echo [INFO] Instalando/actualizando dependencias...
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Fallo instalando requirements.
        pause
        exit /b 1
    )
    echo !REQ_HASH! > "%HASH_FILE%"
) else (
    echo [INFO] Dependencias OK. Saltando instalacion.
)

REM 5) Ejecutar app
echo [INFO] Ejecutando aplicacion...
start "" pythonw main.py
exit
