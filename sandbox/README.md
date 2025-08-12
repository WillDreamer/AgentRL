# Local sandbox setup with firejail

Install:

```bash
sudo apt-get update && sudo apt-get install -y firejail
pip install "fastapi[all]" uvicorn
```

Run:

```bash
cd sandbox
uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4
```

Test:

```bash
# test code exec
curl -X POST http://127.0.0.1:12345/faas/sandbox/ -H 'Content-Type: application/json' -d '{"code":"print(1+1)","language":"python","compile_timeout":1.0,"run_timeout":3.0}'
# test stdin
curl -X POST http://127.0.0.1:12345/faas/sandbox/ -H 'Content-Type: application/json' -d '{"code":"name = input(\"Your name:\"); print(f\"Hi, {name}!\")","stdin":"Alice","language":"python","compile_timeout":1.0,"run_timeout":3.0}'
# test via python code
SANDBOX_ENDPOINT=http://127.0.0.1:12345/faas/sandbox/ python local_sandbox.py
```
