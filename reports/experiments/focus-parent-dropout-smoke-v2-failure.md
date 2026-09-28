# Smoke v2: checksum guard stopped before optimizer

Actual recovered launcher: error after63.566654056seconds, cap900seconds,
submission_performed=false. Log traceback stops at the full-audit SHA check,
before augmentation reconstruction, diagnostics or optimizer construction.
No training checkpoint was produced. Log follower14887 and download56156 terminal.

All downloaded runtime file bytes exactly match the launched notebook's decoded
runtime dictionary. The embedded audit's SHA was
`8a0256c0973ee34c492b7e25c4021c205a0f3c633a57e6a0a58846fb8f01bf37`;
the expected original Windows file SHA was
`dd75e5643a84645e0bef2b6d7a3f80d09347b646fdcb61794d887437ca115225`.
The original contains31,791CRLF line endings. `read_text()` normalized these to
LF before compression. Local tests confirm parsed JSON and normalized text agree,
but raw bytes do not. This is a packaging defect, not a model-quality failure.

Separate v3 packages `read_bytes()` losslessly and updates the runtime file hash
to that same original byte identity. Worker, settings, data, checksum guards and
budget are unchanged. A new regression test requires exact original audit-byte
round trip, not merely equality with the previously normalized runtime.
Nine focused tests passed1.74s. No larger training authorized until smoke passes.
