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
# Windows' built-in voice reads a line, so speech recognition is checked on real speech.
Add-Type -AssemblyName System.Speech
$sample = Join-Path $env:RUNNER_TEMP "speech-sample.wav"
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$voice.SetOutputToWaveFile($sample, $format)
$voice.Speak("Get in the car before the police arrive.")
$voice.Dispose()
$env:ST_SPEECH_SAMPLE = $sample

$log = Join-Path $env:SCREEN_TRANSLATOR_HOME "screen-translator.log"
$test = Start-Process dist\screen-translator\screen-translator.exe -ArgumentList "--self-test" -PassThru
$null = $test.Handle  # without this, ExitCode stays empty once the process exits
if (-not $test.WaitForExit(20 * 60 * 1000)) {
    Get-Content $log -ErrorAction SilentlyContinue
    Stop-Process -Id $test.Id -Force
    Write-Error "The self-test was still running after 20 minutes; the log above shows how far it got."
}
Get-Content $log
if ($test.ExitCode -ne 0) { exit $test.ExitCode }
if (-not (Select-String -Path $log -Pattern "^Speech \(" -Quiet)) {
    Write-Error "The self-test did not recognize speech; is the speech model in $env:SCREEN_TRANSLATOR_HOME\models\speech?"
}
if (-not (Select-String -Path $log -Pattern "^Side by side:" -Quiet)) {
    Write-Error "The self-test did not translate side by side"
}
if (-not (Select-String -Path $log -Pattern "^Online: Claude answered" -Quiet)) {
    Write-Error "The self-test could not reach Claude"
}
if (-not (Select-String -Path $log -Pattern "^Pronouns: " -Quiet)) {
    Write-Error "The self-test did not work out a relationship"
}
if (-not (Select-String -Path $log -Pattern "^Dub: Natural voice" -Quiet)) {
    Write-Error "The self-test did not speak with a natural voice; are they in $env:SCREEN_TRANSLATOR_HOME\models\voices?"
}
if (-not (Select-String -Path $log -Pattern "^Model hy-mt1.5-1.8b loaded" -Quiet)) {
    Write-Error "The self-test did not load HY-MT; is it in $env:SCREEN_TRANSLATOR_HOME\models?"
}
if (-not (Select-String -Path $log -Pattern "^Translation:" -Quiet)) {
    Write-Error "The self-test did not translate anything; is the test model in $env:SCREEN_TRANSLATOR_HOME\models?"
}
