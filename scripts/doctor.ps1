# SPDX-License-Identifier: Apache-2.0
# Windows counterpart of scripts/doctor.sh: reports the device, build prerequisites and which backends can be built.
# Output lines are "<status> <item>: <detail>" with status ok | missing | skip | info.
# Exits non-zero only when core prerequisites are missing. Runs on Windows PowerShell 5.1 and PowerShell 7.
#
#   powershell -NoProfile -File scripts/doctor.ps1
$ErrorActionPreference = 'SilentlyContinue'
$script:coreMissing = $false

function Report($status, $text) {
    '{0,-8} {1}' -f $status, $text
}

# PowerShell 7 defines $IsWindows; Windows PowerShell 5.1 runs only on Windows.
$onWindows = $true
if (Test-Path variable:IsWindows) { $onWindows = $IsWindows }

'== device'
if (-not $onWindows) {
    Report 'missing' 'target: this is not Windows; use scripts/doctor.sh'
    exit 1
}
$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
# The hardware's architecture comes from WMI: an emulated x64 process on an ARM64 machine still sees "AMD64" elsewhere.
$architectures = @{ 0 = 'x86'; 5 = 'ARM'; 9 = 'x64'; 12 = 'ARM64' }
$architecture = $architectures[[int]$processor.Architecture]
if (-not $architecture) { $architecture = $env:PROCESSOR_ARCHITECTURE }
$shellArchitecture = $env:PROCESSOR_ARCHITECTURE
Report 'info' ('os: Windows {0}' -f $architecture)
if ($architecture -eq 'ARM64' -and $shellArchitecture -ne 'ARM64') {
    Report 'info' ('shell: this PowerShell is an emulated {0} process; Python and builds started from it may be x64, not ARM64' -f $shellArchitecture)
}
$chip = $processor.Name
if ($chip) { Report 'info' ('chip: {0}' -f $chip.Trim()) }
$npu = Get-PnpDevice | Where-Object { $_.FriendlyName -like '*NPU*' -and $_.Status -eq 'OK' } |
    Select-Object -First 1 -ExpandProperty FriendlyName
if ($npu) { Report 'ok' ('npu: {0}' -f $npu) } else { Report 'skip' 'npu: none detected' }
$gpus = Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }
if ($gpus) { Report 'info' ('gpu: {0}' -f ($gpus -join ', ')) }
if (Get-Command nvidia-smi) {
    $nvidia = (nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | Select-Object -First 1)
    if ($nvidia) { Report 'ok' ('nvidia: {0}' -f $nvidia) }
}
if (Get-Command vulkaninfo) {
    $vulkan = vulkaninfo --summary 2>$null | Select-String 'deviceName' | Select-Object -First 1
    if ($vulkan) { Report 'ok' ('vulkan: {0}' -f ($vulkan.ToString() -replace '.*=\s*', '')) }
}
if ($chip -match 'X1E|X1P|X Elite|X Plus') {
    Report 'info' 'hexagon arch hint: v73 (QNN hexagon_arch option; verify against your chip)'
}
Report 'info' 'presets: cmake --preset windows-arm64-debug | windows-x64-debug (then cmake --build --preset ... and ctest --preset ...)'

'== core'
foreach ($tool in 'cmake', 'git') {
    $command = Get-Command $tool
    if ($command) { Report 'ok' ('{0}: {1}' -f $tool, ((& $tool --version) | Select-Object -First 1)) }
    else { Report 'missing' ('{0}: not found' -f $tool); $script:coreMissing = $true }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$studio = $null
$otherStudio = $null
if (Test-Path $vswhere) {
    # The presets name the Visual Studio 2022 generator (version 17), so only 2022 counts.
    $studio = & $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName
    if (-not $studio) { $otherStudio = & $vswhere -latest -products '*' -property displayName }
}
if ($studio) { Report 'ok' ('cc: {0} (use a preset, or -G "Visual Studio 17 2022")' -f $studio) }
elseif ($otherStudio) { Report 'missing' ('cc: found {0}, but the presets and docs use Visual Studio 2022 (generator ''Visual Studio 17 2022'') with the C++ tools; install it or pass your own generator' -f $otherStudio); $script:coreMissing = $true }
else { Report 'missing' 'cc: Visual Studio 2022 with the C++ tools not found'; $script:coreMissing = $true }
# The vendor-file test needs Git's bash. A bash.exe in System32 is the WSL launcher, which sees Linux paths.
$gitBash = $null
$gitCommand = Get-Command git
if ($gitCommand) {
    $candidate = Join-Path (Split-Path (Split-Path $gitCommand.Source)) 'bin\bash.exe'
    if (Test-Path $candidate) { $gitBash = $candidate }
}
if ($gitBash) { Report 'ok' ('bash: {0} (Git Bash, used by the no_vendor_files test)' -f $gitBash) }
else { Report 'missing' 'bash: Git Bash not found next to git; the no_vendor_files test needs it (System32\bash.exe is WSL)'; $script:coreMissing = $true }

'== backends (SDKs are installed by you under their own licenses)'
$qnn = $env:QNN_SDK_ROOT
if (-not $qnn) {
    $found = Get-ChildItem 'C:\Qualcomm\AIStack\QAIRT' -Directory | Where-Object {
        Test-Path (Join-Path $_.FullName 'include\QNN\QnnInterface.h') } | Sort-Object Name | Select-Object -Last 1
    if ($found) { Report 'skip' ('qnn: found {0}; set QNN_SDK_ROOT to it (newer QAIRT releases are needed for .dlc models)' -f $found.FullName) }
    else { Report 'skip' 'qnn: set QNN_SDK_ROOT to your QAIRT SDK directory' }
} elseif (-not (Test-Path (Join-Path $qnn 'include\QNN\QnnInterface.h'))) {
    Report 'missing' 'qnn: QNN_SDK_ROOT set but include\QNN\QnnInterface.h not found'
} elseif (-not (Test-Path (Join-Path $qnn 'include\QNN\System\QnnSystemDlc.h'))) {
    Report 'missing' ('qnn: {0} is too old: it lacks System\QnnSystemDlc.h (the DLC API); use a newer QAIRT' -f $qnn)
} else {
    Report 'ok' ('qnn: {0}' -f $qnn)
}

function CheckRoot($variable, $name, $headers, $hint) {
    $root = [Environment]::GetEnvironmentVariable($variable)
    if (-not $root) { Report 'skip' ('{0}: set {1}{2}' -f $name, $variable, $hint); return }
    foreach ($header in $headers) {
        if (Test-Path (Join-Path $root $header)) { Report 'ok' ('{0}: {1}' -f $name, $root); return }
    }
    Report 'missing' ('{0}: header not found under {1}' -f $name, $variable)
}
CheckRoot 'ONNXRUNTIME_ROOT' 'onnxruntime' @('include\onnxruntime_c_api.h', 'include\onnxruntime\onnxruntime_c_api.h') ''
CheckRoot 'OPENVINO_ROOT' 'openvino' @('include\openvino\c\openvino.h', 'runtime\include\openvino\c\openvino.h') ' (an x64 install; there is no Windows ARM64 build)'
CheckRoot 'NCNN_ROOT' 'ncnn' @('include\ncnn\net.h') ' (an install built with NCNN_VULKAN=ON)'
CheckRoot 'TENSORRT_ROOT' 'tensorrt' @('include\NvInfer.h') ''
Report 'skip' 'coreml: requires macOS'

if ($script:coreMissing) { exit 1 }
exit 0
