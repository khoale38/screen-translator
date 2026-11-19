$ErrorActionPreference = "Stop"
$loader = Get-Command vulkan-1.dll -ErrorAction SilentlyContinue
"Vulkan loader on the DLL search path: $(if ($loader) { $loader.Source } else { 'none' })"
$log = Join-Path $env:SCREEN_TRANSLATOR_HOME "screen-translator.log"
$test = Start-Process dist\screen-translator\screen-translator.exe -ArgumentList "--self-test" -Wait -PassThru
Get-Content $log
if ($test.ExitCode -ne 0) { exit $test.ExitCode }
if (-not (Select-String -Path $log -Pattern "^Translation:" -Quiet)) {
    Write-Error "The self-test did not translate anything; is the test model in $env:SCREEN_TRANSLATOR_HOME\models?"
}
