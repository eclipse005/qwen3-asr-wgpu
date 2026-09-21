# L1/L2 sweep: every model size x every fixture, with the phase breakdown the
# engine prints on stderr.  One GPU job at a time -- this script is serial.
#
#   pwsh -File scripts/bench.ps1                 # both sizes, all 6 fixtures
#   pwsh -File scripts/bench.ps1 -Size 0.6B      # one size
#   pwsh -File scripts/bench.ps1 -Repeats 3      # keep the best of N
#
# Output: one TSV row per run on stdout, plus scripts/bench-last.tsv.
param(
    [string]$Size = 'both',
    [int]$Repeats = 1,
    [string]$Tsv = "$PSScriptRoot\bench-last.tsv"
)

$ErrorActionPreference = 'Stop'
# The transcripts are UTF-8, and `transcribe`'s "vs <baseline>: MATCH|MISMATCH"
# line comes after one -- decoding it as the console's legacy code page turns a
# CJK transcript into a byte-soup whose last character can swallow the newline,
# merging that line into the transcript and hiding the verdict.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo 'target\release\transcribe.exe'

$models = [ordered]@{
    '0.6B' = 'D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf'
    '1.7B' = 'D:\Qwen3-ASR\models\Qwen3-ASR-1.7B-hf'
}
if ($Size -ne 'both') { $models = [ordered]@{ $Size = $models[$Size] } }

# wav, max_new, audio seconds
$cases = @(
    @('15s_en.wav', 512, 15.0),
    @('30s_zh.wav', 512, 30.0),
    @('90s_en.wav', 1024, 90.0),
    @('90s_ja.wav', 1024, 89.0),
    @('180s_zh.wav', 1024, 180.0),
    @('180s_en.wav', 1024, 180.0)
)
$fixtures = 'D:\Qwen3-ASR\fixtures'
$baseline = 'D:\qwen3-asr-rs\docs\baseline\texts'

$rows = @()
$hdr = "size`twav`trep`tRTFx`telapsed_s`tmel_ms`tenc_ms`tprefill_ms`tdecode_ms`tsubmit_ms`tread_ms`ttokens`tseq`tmatch"
Write-Host $hdr
$rows += $hdr

foreach ($size in $models.Keys) {
    $model = $models[$size]
    foreach ($case in $cases) {
        $wav = $case[0]; $maxNew = $case[1]; $audioS = $case[2]
        $wavBase = [System.IO.Path]::GetFileNameWithoutExtension($wav)
        $baseTxt = Join-Path $baseline ("python-hf_{0}_{1}.txt" -f $size, $wavBase)
        for ($r = 1; $r -le $Repeats; $r++) {
            $args = @('--model', $model, '--wav', (Join-Path $fixtures $wav), '--max-new', $maxNew)
            if (Test-Path $baseTxt) { $args += @('--baseline', $baseTxt) }
            $lines = & $exe @args 2>&1 | ForEach-Object { "$_" }
            $text = $lines -join "`n"

            $elapsed = if ($text -match 'elapsed=([\d.]+)s') { [double]$Matches[1] } else { $null }
            $rtfx = if ($text -match 'RTFx=([\d.]+)') { [double]$Matches[1] } else { '' }
            # `transcribe` prints exactly one "vs <baseline>: MATCH|MISMATCH" line.
            # Read it as a line rather than a regex: the path has colons in it.
            $vsLine = @($lines | Where-Object { "$_" -like 'vs *: *' }) | Select-Object -Last 1
            $match = if ("$vsLine" -like '*: MATCH') { 'MATCH' }
                     elseif ("$vsLine" -like '*: MISMATCH') { 'MISMATCH' }
                     else { 'no-baseline' }
            $ph = if ($text -match 'mel=(\d+)ms enc=(\d+)ms prefill=(\d+)ms decode=(\d+)ms \[host submit (\d+) / read (\d+)\] tokens=(\d+) seq=(\d+)') {
                @{ mel = $Matches[1]; enc = $Matches[2]; pre = $Matches[3]; dec = $Matches[4]
                   sub = $Matches[5]; rd = $Matches[6]; tok = $Matches[7]; seq = $Matches[8] }
            } else { $null }

            if ($match -eq 'MISMATCH') {
                Write-Host "!! MISMATCH $size $wav" -ForegroundColor Red
            }
            $row = if ($ph) {
                "{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}`t{7}`t{8}`t{9}`t{10}`t{11}`t{12}`t{13}" -f `
                    $size, $wav, $r, $rtfx, $elapsed, $ph.mel, $ph.enc, $ph.pre, $ph.dec, $ph.sub, $ph.rd, $ph.tok, $ph.seq, $match
            } else {
                "{0}`t{1}`t{2}`t{3}`t{4}`t `t`t`t`t`t`t`t{13}" -f $size, $wav, $r, $rtfx, $elapsed, $match
            }
            Write-Host $row
            $rows += $row
        }
    }
}
[System.IO.File]::WriteAllLines($Tsv, [string[]]$rows)
Write-Host "`nwrote $Tsv"
if ($rows -match 'MISMATCH') { exit 1 }
