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

# Vendor versions: scripts/vendors.tsv holds the tested and minimum versions and the download pages (docs/vendors.md).
$vendors = @{}
$vendorsFile = Join-Path $PSScriptRoot 'vendors.tsv'
if (Test-Path $vendorsFile) {
    foreach ($line in Get-Content $vendorsFile) {
        if ($line.StartsWith('#') -or -not $line.Trim()) { continue }
        $field = $line -split "`t"
        $vendors[$field[0]] = @{ tested = $field[3]; minimum = $field[4]; url = $field[6] }
    }
}
function VendorValue($backend, $key) {
    $value = $vendors[$backend][$key]
    if ($value -and $value -ne '-') { $value }
}
function AsVersion($text) {
    if ($text -match '^\d+(\.\d+){1,3}') { try { [version]$Matches[0] } catch { } }
}
function DirVersion($path) {
    $name = Split-Path ($path.TrimEnd('\', '/')) -Leaf
    if ($name -match '^\d+(\.\d+)+') { $Matches[0] }
}
# Where to get a backend's SDK; printed when it is missing, so the user (never this script) downloads it.
function VendorHint($backend) {
    $url = VendorValue $backend 'url'
    $minimum = VendorValue $backend 'minimum'
    if ($url) {
        $suffix = if ($minimum) { " (minimum $minimum)" } else { '' }
        Report 'info' ('{0}: get it from {1}{2}; you install it and accept its licence' -f $backend, $url, $suffix)
    }
}
# Older than the minimum is reported as missing; a version other than the tested one is information, not an error.
function VendorCheck($backend, $have) {
    $tested = VendorValue $backend 'tested'
    $minimum = VendorValue $backend 'minimum'
    if (-not $have) {
        $suffix = if ($tested) { " (tested with $tested)" } else { '' }
        Report 'info' ('{0}: version not detected{1}' -f $backend, $suffix)
        return
    }
    $haveVersion = AsVersion $have
    $minimumVersion = AsVersion $minimum
    if ($haveVersion -and $minimumVersion -and $haveVersion -lt $minimumVersion) {
        Report 'missing' ('{0}: version {1} is older than the minimum {2}' -f $backend, $have, $minimum)
    } elseif ($tested -and $tested -ne 'submodule' -and -not $have.StartsWith($tested)) {
        Report 'info' ('{0}: version {1}; tested with {2}, so other versions are unverified' -f $backend, $have, $tested)
    } else {
        Report 'info' ('{0}: version {1}' -f $backend, $have)
    }
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
$npu = Get-PnpDevice | Where-Object { $_.FriendlyName -match '\bNPU\b' -and $_.Status -eq 'OK' } |
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
$studio2026 = $null
if (Test-Path $vswhere) {
    # The presets name the Visual Studio 2022 generator (version 17), so only 2022 counts.
    $studio = & $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName
    if (-not $studio) {
        $studio2026 = & $vswhere -latest -version '[18.0,19.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName
    }
    if (-not $studio -and -not $studio2026) { $otherStudio = & $vswhere -latest -products '*' -property displayName }
}
if ($studio) { Report 'ok' ('cc: {0} (use a preset, or -G "Visual Studio 17 2022")' -f $studio) }
elseif ($studio2026) { Report 'ok' ('cc: {0} (the presets name Visual Studio 2022; add -G "Visual Studio 18 2026" to the preset configure)' -f $studio2026) }
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
    VendorCheck 'qnn' (DirVersion $qnn)
} else {
    Report 'ok' ('qnn: {0}' -f $qnn)
    VendorCheck 'qnn' (DirVersion $qnn)
}
if ($qnn) {
    # Another QAIRT next to the selected one that is newer: the usual cause of "too old" when a newer one is installed.
    $qnnHave = AsVersion (DirVersion $qnn)
    $qnnNewest = Get-ChildItem (Split-Path ($qnn.TrimEnd('\', '/')) -Parent) -Directory | Where-Object {
        (DirVersion $_.FullName) -and (Test-Path (Join-Path $_.FullName 'include\QNN\QnnInterface.h')) } |
        Sort-Object { AsVersion (DirVersion $_.FullName) } | Select-Object -Last 1
    if ($qnnHave -and $qnnNewest -and $qnnHave -lt (AsVersion (DirVersion $qnnNewest.FullName))) {
        Report 'info' ('qnn: a newer QAIRT {0} is installed at {1}; set QNN_SDK_ROOT to it to use it' -f (DirVersion $qnnNewest.FullName), $qnnNewest.FullName)
    }
} else {
    VendorHint 'qnn'
}

function CheckRoot($variable, $name, $headers, $hint, $versionOf) {
    $root = [Environment]::GetEnvironmentVariable($variable)
    if (-not $root) { Report 'skip' ('{0}: set {1}{2}' -f $name, $variable, $hint); VendorHint $name; return }
    foreach ($header in $headers) {
        if (Test-Path (Join-Path $root $header)) {
            Report 'ok' ('{0}: {1}' -f $name, $root)
            $detected = if ($versionOf) { & $versionOf $root } else { DirVersion $root }
            VendorCheck $name $detected
            return
        }
    }
    Report 'missing' ('{0}: header not found under {1}' -f $name, $variable)
    VendorHint $name
}
# ONNX Runtime releases carry VERSION_NUMBER; NCNN's platform.h defines NCNN_VERSION_STRING; NvInferVersion.h defines NV_TENSORRT_*.
$ortVersion = { param($root) $file = Join-Path $root 'VERSION_NUMBER'; if (Test-Path $file) { (Get-Content $file -Raw).Trim() } }
$ncnnVersion = {
    param($root)
    $match = Select-String -Path (Join-Path $root 'include\ncnn\platform.h') -Pattern '^#define NCNN_VERSION_STRING "(.*)"' | Select-Object -First 1
    if ($match) { $match.Matches[0].Groups[1].Value }
}
$trtVersion = {
    param($root)
    $file = Join-Path $root 'include\NvInferVersion.h'
    if (Test-Path $file) {
        $parts = foreach ($part in 'MAJOR', 'MINOR', 'PATCH', 'BUILD') {
            $match = Select-String -Path $file -Pattern ('^#define NV_TENSORRT_{0} (\d+)' -f $part) | Select-Object -First 1
            if ($match) { $match.Matches[0].Groups[1].Value }
        }
        if ($parts.Count -ge 2) { $parts -join '.' }
    }
}
CheckRoot 'ONNXRUNTIME_ROOT' 'onnxruntime' @('include\onnxruntime_c_api.h', 'include\onnxruntime\onnxruntime_c_api.h') '' $ortVersion
CheckRoot 'OPENVINO_ROOT' 'openvino' @('include\openvino\c\openvino.h', 'runtime\include\openvino\c\openvino.h') ' (an x64 install; there is no Windows ARM64 build)'
CheckRoot 'NCNN_ROOT' 'ncnn' @('include\ncnn\net.h') ' (an install built with NCNN_VULKAN=ON)' $ncnnVersion
CheckRoot 'TENSORRT_ROOT' 'tensorrt' @('include\NvInfer.h') '' $trtVersion
$llamaRoot = $env:UAIRT_LLAMACPP_ROOT
if (-not $llamaRoot) { $llamaRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'third_party\llama.cpp' }
if (Test-Path (Join-Path $llamaRoot 'include\llama.h')) {
    Report 'ok' ('llamacpp: {0}' -f $llamaRoot)
    $llamaVersion = git -C $llamaRoot describe --tags --always
    if ($llamaVersion) { Report 'info' ('llamacpp: version {0} (a pinned submodule; bumping it needs ctest and a model run, see docs/vendors.md)' -f $llamaVersion) }
} else {
    Report 'skip' 'llamacpp: run git submodule update --init third_party/llama.cpp (or set UAIRT_LLAMACPP_ROOT)'
}
Report 'skip' 'coreml: requires macOS'

if ($script:coreMissing) { exit 1 }
exit 0
