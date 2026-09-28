# Verified FOCUS proposal feature cache

Kaggle `biohub-focus-adaptation-features-v1/1` completed. Worker156.861s;
launcher211.299s. Two prior clips replayed exact neural matrices and normalized
image hashes; flow tolerance1e-5um passed. All792 new adjacent pairs plus four
replay pairs saved and strictly reloaded, with unchanged encoder/head/flow.

Host verified834 required artifacts: frozen runtime bytes, original raw hashes,
movie roles, pair manifests and packet hashes, exact global node IDs/native
coordinates, finite32-channel FP32 features/positions, physical flow alignment
and every sparse target label. `focus-adaptation-features-v1-result.json` is
the verification receipt; worker resultSHA3b7d190c.... No optimization or score
improvement is implied by successful feature extraction.

The initial sequential local transfer was intentionally replaced by four
download workers. A broad recovery filter tried to validate an unrelated
repository LICENSE as a data artifact and returned an error after transferring
files. Recovery was restricted to834 required outputs/runtime/terminal files;
all834 were already complete and retained, then passed full hashes. No GPU
rerun, artifact deletion, prediction change or source/target access occurred.
