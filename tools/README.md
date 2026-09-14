# PC tools

`download_models.ps1` checks the current `qai_hub_models` module and intentionally does not guess CLI export/fetch arguments.

`inspect_qnn_assets.ps1` checks only `$env:QNN_SDK_ROOT` and `$env:QAIRT_SDK_ROOT` for `libQnnHtp.so`.

Set the SDK root explicitly before inspection, for example:

```powershell
$env:QNN_SDK_ROOT = 'C:\path\to\qnn-sdk'
.\tools\inspect_qnn_assets.ps1
```
