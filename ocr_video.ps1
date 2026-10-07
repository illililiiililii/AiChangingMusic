param([Parameter(Mandatory=$true)][string]$FramesDir)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.RandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [WindowsRuntimeSystemExtensions]
$null = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages
$awaiter = [WindowsRuntimeSystemExtensions].GetMethods([Reflection.BindingFlags]::Public -bor [Reflection.BindingFlags]::Static) | Where-Object { $_.Name -eq 'GetAwaiter' -and $_.IsGenericMethodDefinition -and $_.GetParameters()[0].ParameterType.Name -like 'IAsyncOperation*' } | Select-Object -First 1
function Invoke-Async([object]$AsyncTask,[Type]$As) { return $awaiter.MakeGenericMethod($As).Invoke($null,@($AsyncTask)).GetResult() }
$language = [Windows.Globalization.Language]::new('ko')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) { throw '이 Windows에 한국어 OCR 언어 데이터가 없습니다.' }
$results = [System.Collections.Generic.List[object]]::new()
foreach ($image in (Get-ChildItem -LiteralPath $FramesDir -Filter '*.jpg' -File | Sort-Object Name)) {
    $index = [int]([IO.Path]::GetFileNameWithoutExtension($image.Name) -replace '\D','')
    $file = Invoke-Async ([Windows.Storage.StorageFile]::GetFileFromPathAsync($image.FullName)) ([Windows.Storage.StorageFile])
    $stream = Invoke-Async ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Invoke-Async ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Invoke-Async ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Invoke-Async ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $lineTexts = @($result.Lines | ForEach-Object { $_.Text })
    $results.Add([pscustomobject]@{index=$index; text=($lineTexts -join "`n")})
    $stream.Dispose()
}
ConvertTo-Json -InputObject @($results) -Depth 4 -Compress
