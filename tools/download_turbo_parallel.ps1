$ErrorActionPreference='Stop'
$url='https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/whisper_large_v3_turbo/releases/v0.62.2/whisper_large_v3_turbo-qnn_context_binary-float-qualcomm_snapdragon_8gen3.zip'
$out='D:\mcp-agent-workspace\whisper-app\tools\whisper_large_v3_turbo_asset'
$total=2018863899
$chunk=67108864
$jobs=@()
for($i=0;$i -lt [math]::Ceiling($total/$chunk);$i++){
  $start=$i*$chunk
  $end=[math]::Min($total-1,(($i+1)*$chunk)-1)
  $part=Join-Path $out ("part_{0:D2}" -f $i)
  $args=@('-L','--fail','--retry','5','--retry-delay','2','--range',"$start-$end",$url,'-o',$part)
  $jobs += Start-Process -FilePath 'curl.exe' -ArgumentList $args -WindowStyle Hidden -PassThru
}
$jobs | Select-Object Id | Out-File (Join-Path $out 'parallel_pids.txt') -Encoding utf8
'STARTED' | Out-File (Join-Path $out 'parallel.log') -Encoding utf8
