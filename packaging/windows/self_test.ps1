$ErrorActionPreference = "Stop"
$bundled = Get-ChildItem dist\screen-translator\_internal\ocr_models\*.onnx -ErrorAction SilentlyContinue
if (-not $bundled) { Write-Error "The build has no bundled OCR models in _internal\ocr_models" }
"Bundled OCR models: $($bundled.Name -join ', ')"
$loader = Get-Command vulkan-1.dll -ErrorAction SilentlyContinue
"Vulkan loader on the DLL search path: $(if ($loader) { $loader.Source } else { 'none' })"
"Vulkan loader in System32: $(Test-Path "$env:SystemRoot\System32\vulkan-1.dll")"

# An unreachable proxy makes any OCR model download fail, so the test proves the first run works offline.
$env:HTTP_PROXY = "http://127.0.0.1:9"
$env:HTTPS_PROXY = "http://127.0.0.1:9"
Remove-Item -Recurse -Force (Join-Path $env:SCREEN_TRANSLATOR_HOME "ocr-models") -ErrorAction SilentlyContinue
$log = Join-Path $env:SCREEN_TRANSLATOR_HOME "screen-translator.log"
$test = Start-Process dist\screen-translator\screen-translator.exe -ArgumentList "--self-test" -Wait -PassThru
Get-Content $log
if ($test.ExitCode -ne 0) { exit $test.ExitCode }
if (-not (Select-String -Path $log -Pattern "^Translation:" -Quiet)) {
    Write-Error "The self-test did not translate anything; is the test model in $env:SCREEN_TRANSLATOR_HOME\models?"
}
