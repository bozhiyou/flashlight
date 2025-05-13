# clone the repo
git clone https://github.com/bozhiyou/flashlight.git
cd flashlight
# switch to the benchmark branch
git checkout benchmark
# run the benchmark; add root path for monkeypatch utils
PYTHONPATH=. python benchmarks/attention.py